from __future__ import absolute_import

import argparse
import os
import random

import apache_beam as beam
# import apache_beam.runners.interactive.interactive_beam as ib
import apache_beam.transforms.sql
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

import beam__common
import fidscs_globals


def pl__2__get_keys__train_val_split_candidates(document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll):
    """
    document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll is the main table we use for training.
        It will ultimately provide which frame sequences correspond to individual tokens.

        document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll:
            beam.Row(
                DocumentID,
                ASLConsultantID,
                CameraPerspective,
                TargetVideoFilename,
                UtteranceSequence,
                TokenSequence,
                FrameSequence,
                TokenID
            )

    But our first measure is to build train and validation sets (for tokens).
      In order to split up train vs validation sets, we need to compare "apples to apples".
      That is, in order for a token (TokenID) to be considered a candidate for the split,
      we require at least two of the same (TokenID, CameraPerspective) from differing
      consultants (OR differing utterances OR token sequence, for the same consultant).  
      
      We would prefer more than two of these tuples, with the majority of said
      tuples being assigned to the training set and the remainder (at least one) being
      assigned to the validation set.  We would like to achieve a 90/10 split, ideally,
      but we will take what we get.


    returns:
        two pcollections of tuples of the form:
            (
                <TokenID>,
                <CameraPerspective>,
                <ASLConsultantID>,
                <TargetVideoFilename>,
                <UtteranceSequence>,
                <TokenSequence>
            )

        the first pcollection represents the basis for splitting into
            the training and validation sets

            the split is done by grouping all records keyed by (TokenID, CameraPerspective)

            the splitting algorithm will:
                1. split at .9/.1 ratio (train/validation) when there are at least 10 
                    such occurrences resulting from the (TokenID, CameraPerspective) grouping
                2. when there are less than 10, the validation set will have only a single
                    observation

            this guarantees that the validation set will have at least one observation 
                keyed by (TokenID, CameraPerspective) in common with the (initial) training set

        the second pcollection contains "dangling" observations, keyed by (TokenID, CameraPerspective)
            that have only a single observation associated with this pair; these instances will
            still be used for training but, since there is only a single instance, we have no way
            to validate it (since there are no other available instances to "send" to the validation set)
    """

    tcpctvustsfs = (
        document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll
        | "Beam PL: extract (TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence,FrameSequence) from dctvustsfs schemad pcoll" >> beam.Map(
                lambda dctvustsfs_row: (
                    dctvustsfs_row.TokenID,
                    dctvustsfs_row.CameraPerspective,
                    dctvustsfs_row.ASLConsultantID,
                    dctvustsfs_row.TargetVideoFilename,
                    dctvustsfs_row.UtteranceSequence,
                    dctvustsfs_row.TokenSequence,
                    dctvustsfs_row.FrameSequence
                )
            )
    )

    # for train-validation split, we want to key/group by (TokenID, CameraPerspective) with lists of unique (ASLConsultantID, TargetVideoFilename, UtteranceSequence, TokenSequence) > 1
    ctvusts_by_tcp = (
        tcpctvustsfs
        | "Beam PL: extract ((TokenID,CameraPerspective), (ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence)) from tcpctvustsfs" >> beam.Map(
                lambda tcpctvustsfs_row_tpl: (
                    (
                        tcpctvustsfs_row_tpl[0],
                        tcpctvustsfs_row_tpl[1]
                    ),
                    (
                        tcpctvustsfs_row_tpl[2],
                        tcpctvustsfs_row_tpl[3],
                        tcpctvustsfs_row_tpl[4],
                        tcpctvustsfs_row_tpl[5]
                    )
                )
            )
        | "Beam PL: select distinct ((TokenID,CameraPerspective), (ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence)) from ctvusts_by_tcp" >> beam.Distinct()
        | "Beam PL: group (ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence) by key (TokenID,CameraPerspective)" >> beam.GroupByKey() 
        # the above produces tuples of the form:
            # (
            #     (
            #         TokenID,
            #         CameraPerspective
            #     ),
            #     listof(
            #       (ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence)
            #     )
            # )
    )


    def flatten_ctvusts_by_tcp(ctvusts_by_tcp_tpl):
        return [
            (
                ctvusts_by_tcp_tpl[0][0],   # TokenID
                ctvusts_by_tcp_tpl[0][1],   # CameraPerspective
                ctvusts_tpl[0],             # ASLConsultantID
                ctvusts_tpl[1],             # TargetVideoFilename
                ctvusts_tpl[2],             # UtteranceSequence
                ctvusts_tpl[3]              # TokenSequence
            ) for ctvusts_tpl in ctvusts_by_tcp_tpl[1]
        ]

    ctvusts_by_tcp__gt_1 = (
        ctvusts_by_tcp
        | "Beam PL: filter candidate (TokenID,CameraPerspective) for test-validation split" >> beam.Filter(
                lambda list_ctvusts_by_tcp_tpl: len(set(list_ctvusts_by_tcp_tpl[1])) > 1
            )
        | "Beam PL: flatten filtered (TokenID,CameraPerspective) candidates for test-validation split" >> beam.FlatMap(flatten_ctvusts_by_tcp)

        # debug
        # | "Beam PL: print ctvusts_by_tcp__gt_1" >> beam.ParDo(beam__common.PipelinePcollPrinter("ctvusts_by_tcp__gt_1 entry"))
    )

    ctvusts_by_tcp__lte_1 = (
        ctvusts_by_tcp
        | "Beam PL: filter non-candidate (TokenID,CameraPerspective) for test-validation split" >> beam.Filter(
                lambda list_ctvusts_by_tcp_tpl: len(set(list_ctvusts_by_tcp_tpl[1])) <= 1
            )
        | "Beam PL: flatten filtered (TokenID,CameraPerspective) non-candidates for test-validation split" >> beam.FlatMap(flatten_ctvusts_by_tcp)

        # debug
        # | "Beam PL: print ctvusts_by_tcp__lte_1" >> beam.ParDo(beam__common.PipelinePcollPrinter("ctvusts_by_tcp__lte_1 entry"))
    )

    return (
        ctvusts_by_tcp__gt_1,   # candidates for train/val split
        ctvusts_by_tcp__lte_1,  # NON-candidates for train/val split
        tcpctvustsfs
    )


def pl__3__do_train_val_split_keys(train_val_split_candidates_keys):
    """
    train_val_split_candidates_keys (pcollection of):
        (
            <TokenID>,
            <CameraPerspective>,
            <ASLConsultantID>,
            <TargetVideoFilename>,
            <UtteranceSequence>,
            <TokenSequence>
        )

    returns:
        two pcollections of tuples of the form:
            (
                <TokenID>,
                <CameraPerspective>,
                <ASLConsultantID>,
                <TargetVideoFilename>,
                <UtteranceSequence>,
                <TokenSequence>
            )

        they represent the keys for training and validation splits, respectively
    """

    # first, we need to put train_val_split_candidates_keys back into ((TokenID, CameraPerspective), (ASLConsultantID, TargetVideoFilename, UtteranceSequence, TokenSequence)) form
    def rekey_ctvusts_by_tcp(ctvusts_by_tcp_tpl):
        return (
            (
                ctvusts_by_tcp_tpl[0],  # TokenID
                ctvusts_by_tcp_tpl[1]   # CameraPerspective
            ),
            (
                ctvusts_by_tcp_tpl[2],  # ASLConsultantID
                ctvusts_by_tcp_tpl[3],  # TargetVideoFilename
                ctvusts_by_tcp_tpl[4],  # UtteranceSequence
                ctvusts_by_tcp_tpl[5]   # TokenSequence
            )
        )

    # first, we need to put train_val_split_candidates_keys back into ((TokenID, CameraPerspective), (ASLConsultantID, TargetVideoFilename, UtteranceSequence, TokenSequence)) form
    def rekey_ctvusts_by_tcp(ctvusts_by_tcp_tpl):
        return (
            (
                ctvusts_by_tcp_tpl[0],  # TokenID
                ctvusts_by_tcp_tpl[1]   # CameraPerspective
            ),
            (
                ctvusts_by_tcp_tpl[2],  # ASLConsultantID
                ctvusts_by_tcp_tpl[3],  # TargetVideoFilename
                ctvusts_by_tcp_tpl[4],  # UtteranceSequence
                ctvusts_by_tcp_tpl[5]   # TokenSequence
            )
        )

    def val_train_split__train_val_split_candidates_keys__tpl(ctvusts_list__by__tcp__gt_1__tpl):
        """
        ctvusts_list__by__tcp__gt_1__tpl
            (
                (TokenID,CameraPerspective), # key
                listof(
                    (ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence)
                )
            )
        """
        ctvusts_list = ctvusts_list__by__tcp__gt_1__tpl[1].copy() # we need a copy since we want to shuffle
        random.shuffle(ctvusts_list)
        len_ctvusts_list = len(ctvusts_list)
        val_len_ctvusts_list = int(len_ctvusts_list*fidscs_globals.VALIDATION_SIZE_RATIO) if len_ctvusts_list > int(((1-fidscs_globals.VALIDATION_SIZE_RATIO)*100)/10) else 1
        train__ctvusts_list, val__ctvusts_list = ctvusts_list[val_len_ctvusts_list:], ctvusts_list[:val_len_ctvusts_list]
        return (
            (
                ctvusts_list__by__tcp__gt_1__tpl[0][0],    # TokenID
                ctvusts_list__by__tcp__gt_1__tpl[0][1]     # CameraPerspective
            ),
            (
                train__ctvusts_list,
                val__ctvusts_list
            )
        )

    val_train_split_basis__train_val_split_candidates_keys = (
        train_val_split_candidates_keys
        | "Beam PL: rekey train_val_split_candidates_keys for validation/train split" >> beam.Map(rekey_ctvusts_by_tcp)
        | "Beam PL: group (ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence) rekeyed by (TokenID,CameraPerspective)" >> beam.GroupByKey()
        # the above produces tuples of the form:
            # (
            #     (TokenID,CameraPerspective), # key
            #     listof(
            #       (ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence)
            #     )
            # )
        | "Beam PL: split rekeyed ctvusts_list_by_tcp__gt_1" >> beam.Map(val_train_split__train_val_split_candidates_keys__tpl)
        # the above produces tuples of the form:
            # (
            #     (TokenID,CameraPerspective), # key
            #     (
            #       test_list_of(ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence),
            #       val_list_of(ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence),
            #     )
            # )
    )

    train__train_val_split_candidates_keys = (
        val_train_split_basis__train_val_split_candidates_keys
        | "Beam PL: select train sublist from val_train_split_basis__train_val_split_candidates_keys" >> beam.Map(
                lambda val_train_split_basis__train_val_split_candidates_keys_tpl: [
                    (
                        val_train_split_basis__train_val_split_candidates_keys_tpl[0][0],  # TokenID
                        val_train_split_basis__train_val_split_candidates_keys_tpl[0][1],  # CameraPerspective
                        train_ctvusts_tpl[0],                                   # ASLConsultantID
                        train_ctvusts_tpl[1],                                   # TargetVideoFilename
                        train_ctvusts_tpl[2],                                   # UtteranceSequence
                        train_ctvusts_tpl[3]                                    # TokenSequence
                    ) for train_ctvusts_tpl in val_train_split_basis__train_val_split_candidates_keys_tpl[1][0] # index [1][0] points to train sublist
                ]
            )
        | "Beam PL: 'explode list_train__train_val_split_candidates_keys_tpl" >> beam.FlatMap(lambda list_train__train_val_split_candidates_keys_tpl: list_train__train_val_split_candidates_keys_tpl)
    )

    val__train_val_split_candidates_keys = (
        val_train_split_basis__train_val_split_candidates_keys
        | "Beam PL: select validation sublist from val_train_split_basis__train_val_split_candidates_keys" >> beam.Map(
                lambda val_train_split_basis__train_val_split_candidates_keys_tpl: [
                    (
                        val_train_split_basis__train_val_split_candidates_keys_tpl[0][0],  # TokenID
                        val_train_split_basis__train_val_split_candidates_keys_tpl[0][1],  # CameraPerspective
                        val_ctvusts_tpl[0],                                     # ASLConsultantID
                        val_ctvusts_tpl[1],                                     # TargetVideoFilename
                        val_ctvusts_tpl[2],                                     # UtteranceSequence
                        val_ctvusts_tpl[3]                                      # TokenSequence
                    ) for val_ctvusts_tpl in val_train_split_basis__train_val_split_candidates_keys_tpl[1][1] # index [1][1] points to validation sublist
                ]
            )
        | "Beam PL: 'explode list_val__train_val_split_candidates_keys_tpl" >> beam.FlatMap(lambda list_val__train_val_split_candidates_keys_tpl: list_val__train_val_split_candidates_keys_tpl)
    )

    return train__train_val_split_candidates_keys, val__train_val_split_candidates_keys


def pl__4__create_train_frame_sequences__assoc(train__ctvusts_by_tcp, tcpctvustsfs):
    """
    joins train__ctvusts_by_tcp to tcpctvustsfs

    inputs:
        train__ctvusts_by_tcp:

        tcpctvustsfs:

    returns:
        train_tcpctvustsfs:
            (
                <TokenID>,
                <CameraPerspective>,
                <ASLConsultantID>,
                <TargetVideoFilename>,
                <UtteranceSequence>,
                <TokenSequence>,
                <FrameSequence>
            )

        frame_sequences__by__tcpctvustsfs:
            (
                (
                    <TokenID>,
                    <CameraPerspective>,
                    <ASLConsultantID>,
                    <TargetVideoFilename>,
                    <UtteranceSequence>,
                    <TokenSequence>
                ),
                <FrameSequence>
            )
    """

    # join train__ctvusts_by_tcp to tcpctvustsfs
    train__ctvusts_by_tcp__keys = (
        train__ctvusts_by_tcp
        | "Beam PL: extract ((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence), '<train__ctvusts_by_tcp__has_key>') for join to tcpctvustsfs" >> beam.Map(
                lambda train__ctvusts_by_tcp_tpl : (
                    (
                        train__ctvusts_by_tcp_tpl[0], # TokenID
                        train__ctvusts_by_tcp_tpl[1], # CameraPerspective
                        train__ctvusts_by_tcp_tpl[2], # ASLConsultantID
                        train__ctvusts_by_tcp_tpl[3], # TargetVideoFilename
                        train__ctvusts_by_tcp_tpl[4], # UtteranceSequence
                        train__ctvusts_by_tcp_tpl[5]  # TokenSequence
                    ),
                    "<train__ctvusts_by_tcp__has_key>"
                )
            )
    )

    frame_sequences__by__tcpctvustsfs = (
        tcpctvustsfs
        | "Beam PL: extract ((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence), FrameSequence) for join to train__ctvusts_by_tcp/val__ctvusts_by_tcp" >> beam.Map(
                lambda tcpctvustsfs_tpl: (
                    (
                        tcpctvustsfs_tpl[0],  # TokenID
                        tcpctvustsfs_tpl[1],  # CameraPerspective
                        tcpctvustsfs_tpl[2],  # ASLConsultantID
                        tcpctvustsfs_tpl[3],  # TargetVideoFilename
                        tcpctvustsfs_tpl[4],  # UtteranceSequence
                        tcpctvustsfs_tpl[5]   # TokenSequence
                    ),
                    tcpctvustsfs_tpl[6]       # FrameSequence
                )
            )
    )

    train_tcpctvustsfs = (
        ({
            'has_key': train__ctvusts_by_tcp__keys,
            'frame_sequences': frame_sequences__by__tcpctvustsfs
        })
        | "Beam PL: join train__ctvusts_by_tcp to tcpctvustsfs" >> beam.CoGroupByKey()
        # the above produces tuples of the form:
            # (
            #     (
            #         <TokenID>,
            #         <CameraPerspective>,
            #         <ASLConsultantID>,
            #         <TargetVideoFilename>,
            #         <UtteranceSequence>,
            #         <TokenSequence>
            #     ),
            #     {
            #         'has_key': listof('<train__ctvusts_by_tcp__has_key>'),    # should have only one/single element
            #         'frame_sequences': listof(<FrameSequence>)                      # many
            #     }
            # )
        | "Beam PL: filter out mismatches from joined train__ctvusts_by_tcp to tcpctvustsfs" >> beam.Filter(
                lambda joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl: 
                    len(joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[1]['has_key'])>0 and \
                        len(joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[1]['frame_sequences'])>0
            )
        | "Beam PL: 'explode' listof(<FrameSequence>) from joined train__ctvusts_by_tcp to tcpctvustsfs to list of tuples" >> beam.Map(
                lambda joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl: [
                    (
                        joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][0], # TokenID
                        joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][1], # CameraPerspective
                        joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][2], # ASLConsultantID
                        joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][3], # TargetVideoFilename
                        joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][4], # UtteranceSequence
                        joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][5], # TokenSequence
                        frame_seq                                                   # FrameSequence
                    ) for frame_seq in sorted(joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[1]['frame_sequences'])
                ]
            )
        | "Beam PL: 'explode' listof((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence, FrameSequence)) from joined train__ctvusts_by_tcp to tcpctvustsfs" >> beam.FlatMap(
                lambda list_joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl: list_joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl
            )
    )

    return train_tcpctvustsfs, frame_sequences__by__tcpctvustsfs


def pl__5__write_train_frame_sequences__assoc_index_csv(train_frame_sequences__assoc):
    """
    train_frame_sequences__assoc:
        (
            <TokenID>,
            <CameraPerspective>,
            <ASLConsultantID>,
            <TargetVideoFilename>,
            <UtteranceSequence>,
            <TokenSequence>,
            <FrameSequence>
        )
    """
    sorted_train_frame_sequences__assoc_index_pcoll = beam__common.pl__X__sort_pcoll(train_frame_sequences__assoc, pcoll_label="train_frame_sequences__assoc_index")
    sorted_train_frame_sequences__assoc_index_csv_rows_pcoll = (
        sorted_train_frame_sequences__assoc_index_pcoll
        | "Beam PL: apply schema to sorted_train_frame_sequences__assoc_index" >> beam.Map(
                lambda sorted_train_frame_sequences__assoc_index_pcoll_row_tpl: beam.Row(
                    # SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX = [
                    #     'TokenID',
                    #     'CameraPerspective',
                    #     'ASLConsultantID',
                    #     'TargetVideoFilename',
                    #     'UtteranceSequence',
                    #     'TokenSequence',
                    #     'FrameSequence'
                    # ]
                    TokenID=int(sorted_train_frame_sequences__assoc_index_pcoll_row_tpl[0]),
                    CameraPerspective=int(sorted_train_frame_sequences__assoc_index_pcoll_row_tpl[1]),
                    ASLConsultantID=int(sorted_train_frame_sequences__assoc_index_pcoll_row_tpl[2]),
                    TargetVideoFilename=str(sorted_train_frame_sequences__assoc_index_pcoll_row_tpl[3]),
                    UtteranceSequence=int(sorted_train_frame_sequences__assoc_index_pcoll_row_tpl[4]),
                    TokenSequence=int(sorted_train_frame_sequences__assoc_index_pcoll_row_tpl[5]),
                    FrameSequence=int(sorted_train_frame_sequences__assoc_index_pcoll_row_tpl[6])
                )
            )
        | beam.Map(lambda sorted_train_frame_sequences__assoc_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(sorted_train_frame_sequences__assoc_index_schemad_pcoll_row))
    )
    return beam__common.pl__X__write_pcoll_to_csv(
        sorted_train_frame_sequences__assoc_index_csv_rows_pcoll, 
        "TRAIN-FRAME-SEQUENCES-ASSOC-INDEX", 
        fidscs_globals.TRAIN_FRAME_SEQ_ASSOC_DS_FNAME, 
        fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX
    ) # train_frame_sequences__assoc_index_csv_path


def pl__5__create_val_frame_sequences(val__ctvusts_by_tcp, frame_sequences__by__tcpctvustsfs):
    """
    """

    # join val__ctvusts_by_tcp to tcpctvustsfs
    val__ctvusts_by_tcp__keys = (
        val__ctvusts_by_tcp
        | "Beam PL: extract ((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence), '<val__ctvusts_by_tcp__has_key>') for join to tcpctvustsfs" >> beam.Map(
                lambda val__ctvusts_by_tcp_tpl : (
                    (
                        val__ctvusts_by_tcp_tpl[0], # TokenID
                        val__ctvusts_by_tcp_tpl[1], # CameraPerspective
                        val__ctvusts_by_tcp_tpl[2], # ASLConsultantID
                        val__ctvusts_by_tcp_tpl[3], # TargetVideoFilename
                        val__ctvusts_by_tcp_tpl[4], # UtteranceSequence
                        val__ctvusts_by_tcp_tpl[5]  # TokenSequence
                    ),
                    "<val__ctvusts_by_tcp__has_key>"
                )
            )
    )

    val_tcpctvustsfs = (
        ({
            'has_key': val__ctvusts_by_tcp__keys,
            'frame_sequences': frame_sequences__by__tcpctvustsfs
        })
        | "Beam PL: join val__ctvusts_by_tcp to tcpctvustsfs" >> beam.CoGroupByKey()
        # the above produces tuples of the form:
            # (
            #     (
            #         <TokenID>,
            #         <CameraPerspective>,
            #         <ASLConsultantID>,
            #         <TargetVideoFilename>,
            #         <UtteranceSequence>,
            #         <TokenSequence>
            #     ),
            #     {
            #         'has_key': listof('<val__ctvusts_by_tcp__has_key>'),    # should have only one/single element
            #         'frame_sequences': listof(<FrameSequence>)                      # many
            #     }
            # )
        | "Beam PL: filter out mismatches from joined val__ctvusts_by_tcp to tcpctvustsfs" >> beam.Filter(
                lambda joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl: 
                    len(joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[1]['has_key'])>0 and \
                        len(joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[1]['frame_sequences'])>0
            )
        | "Beam PL: 'explode' listof(<FrameSequence>) from joined val__ctvusts_by_tcp to tcpctvustsfs to list of tuples" >> beam.Map(
                lambda joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl: [
                    (
                        joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][0],   # TokenID
                        joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][1],   # CameraPerspective
                        joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][2],   # ASLConsultantID
                        joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][3],   # TargetVideoFilename
                        joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][4],   # UtteranceSequence
                        joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[0][5],   # TokenSequence
                        frame_seq                                                       # FrameSequence
                    ) for frame_seq in sorted(joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl[1]['frame_sequences'])
                ]
            )
        | "Beam PL: 'explode' listof((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence, FrameSequence)) from joined val__ctvusts_by_tcp to tcpctvustsfs" >> beam.FlatMap(
                lambda list_joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl: list_joined__val__ctvusts_by_tcp__to__tcpctvustsfs__tpl
            )
    )

    return val_tcpctvustsfs


def pl__6__write_val_frame_sequences_index_csv(val_frame_sequences):
    """
    val_frame_sequences:
        (
            <TokenID>,
            <CameraPerspective>,
            <ASLConsultantID>,
            <TargetVideoFilename>,
            <UtteranceSequence>,
            <TokenSequence>,
            <FrameSequence>
        )
    """
    sorted_val_frame_sequences_index_pcoll = beam__common.pl__X__sort_pcoll(val_frame_sequences, pcoll_label="val_frame_sequences_index")
    sorted_val_frame_sequences_index_csv_rows_pcoll = (
        sorted_val_frame_sequences_index_pcoll
        | "Beam PL: apply schema to sorted_val_frame_sequences_index" >> beam.Map(
                lambda sorted_val_frame_sequences_index_pcoll_row_tpl: beam.Row(
                    # SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX = [
                    #     'TokenID',
                    #     'CameraPerspective',
                    #     'ASLConsultantID',
                    #     'TargetVideoFilename',
                    #     'UtteranceSequence',
                    #     'TokenSequence',
                    #     'FrameSequence'
                    # ]
                    TokenID=int(sorted_val_frame_sequences_index_pcoll_row_tpl[0]),
                    CameraPerspective=int(sorted_val_frame_sequences_index_pcoll_row_tpl[1]),
                    ASLConsultantID=int(sorted_val_frame_sequences_index_pcoll_row_tpl[2]),
                    TargetVideoFilename=str(sorted_val_frame_sequences_index_pcoll_row_tpl[3]),
                    UtteranceSequence=int(sorted_val_frame_sequences_index_pcoll_row_tpl[4]),
                    TokenSequence=int(sorted_val_frame_sequences_index_pcoll_row_tpl[5]),
                    FrameSequence=int(sorted_val_frame_sequences_index_pcoll_row_tpl[6])
                )
            )
        | beam.Map(lambda sorted_val_frame_sequences__assoc_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(sorted_val_frame_sequences__assoc_index_schemad_pcoll_row))
    )
    return beam__common.pl__X__write_pcoll_to_csv(
        sorted_val_frame_sequences_index_csv_rows_pcoll, 
        "VAL-FRAME-SEQUENCES-ASSOC-INDEX", 
        fidscs_globals.VAL_FRAME_SEQ_DS_FNAME, 
        fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX
    ) # val_frame_sequences__assoc_index_csv_path


def pl__5__create_train_frame_sequences(ctvusts_by_tcp__lte_1, frame_sequences__by__tcpctvustsfs, train_tcpctvustsfs__gt__1):
    """
    returns:
        train_tcpctvustsfs__all
            (
                <TokenID>,
                <CameraPerspective>,
                <ASLConsultantID>,
                <TargetVideoFilename>,
                <UtteranceSequence>,
                <TokenSequence>,
                <FrameSequence>
            )
    """

    train__ctvusts_by_tcp__lte_1__keys = (
        ctvusts_by_tcp__lte_1
        | "Beam PL: extract ((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence), '<ctvusts_by_tcp__lte_1_tpl__has_key>') for join to tcpctvustsfs" >> beam.Map(
                lambda ctvusts_by_tcp__lte_1_tpl : (
                    (
                        ctvusts_by_tcp__lte_1_tpl[0], # TokenID
                        ctvusts_by_tcp__lte_1_tpl[1], # CameraPerspective
                        ctvusts_by_tcp__lte_1_tpl[2], # ASLConsultantID
                        ctvusts_by_tcp__lte_1_tpl[3], # TargetVideoFilename
                        ctvusts_by_tcp__lte_1_tpl[4], # UtteranceSequence
                        ctvusts_by_tcp__lte_1_tpl[5]  # TokenSequence
                    ),
                    "<ctvusts_by_tcp__lte_1_tpl__has_key>"
                )
            )
    )

    train_tcpctvustsfs__lte_1 = (
        ({
            'has_key': train__ctvusts_by_tcp__lte_1__keys,
            'frame_sequences': frame_sequences__by__tcpctvustsfs
        })
        | "Beam PL: join ctvusts_by_tcp__lte_1 to tcpctvustsfs" >> beam.CoGroupByKey()
        # the above produces tuples of the form:
            # (
            #     (
            #         <TokenID>,
            #         <CameraPerspective>,
            #         <ASLConsultantID>,
            #         <TargetVideoFilename>,
            #         <UtteranceSequence>,
            #         <TokenSequence>
            #     ),
            #     {
            #         'has_key': listof('<ctvusts_by_tcp__lte_1_tpl__has_key>'),    # should have only one/single element
            #         'frame_sequences': listof(<FrameSequence>)                      # many
            #     }
            # )
        | "Beam PL: filter out mismatches from joined train__ctvusts_by_tcp__lte_1 to tcpctvustsfs" >> beam.Filter(
                lambda joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl: 
                    len(joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[1]['has_key'])>0 and \
                        len(joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[1]['frame_sequences'])>0
            )
        | "Beam PL: 'explode' listof(<FrameSequence>) from joined train__ctvusts_by_tcp__lte_1 to tcpctvustsfs to list of tuples" >> beam.Map(
                lambda joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl: [
                    (
                        joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[0][0], # TokenID
                        joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[0][1], # CameraPerspective
                        joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[0][2], # ASLConsultantID
                        joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[0][3], # TargetVideoFilename
                        joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[0][4], # UtteranceSequence
                        joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[0][5], # TokenSequence
                        frame_seq
                    ) for frame_seq in sorted(joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl[1]['frame_sequences'])
                ]
            )
        | "Beam PL: 'explode' listof((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence, FrameSequence)) from joined ttrain__ctvusts_by_tcp__lte_1 to tcpctvustsfs" >> beam.FlatMap(
                lambda list_joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl: list_joined__train__ctvusts_by_tcp__lte_1__to__tcpctvustsfs__tpl
            )
    )

    train_tcpctvustsfs__all = (
        (train_tcpctvustsfs__gt__1, train_tcpctvustsfs__lte_1) 
        | f"Beam PL: merge train_tcpctvustsfs__gt__1 with train_tcpctvustsfs__lte_1" >> beam.Flatten() 
    )

    return train_tcpctvustsfs__all


def pl__6__write_train_frame_sequences_index_csv(train_frame_sequences):
    """
    train_frame_sequences:
        (
            <TokenID>,
            <CameraPerspective>,
            <ASLConsultantID>,
            <TargetVideoFilename>,
            <UtteranceSequence>,
            <TokenSequence>,
            <FrameSequence>
        )
    """
    sorted_train_frame_sequences_index_pcoll = beam__common.pl__X__sort_pcoll(train_frame_sequences, pcoll_label="train_frame_sequences_index")
    sorted_train_frame_sequences_index_csv_rows_pcoll = (
        sorted_train_frame_sequences_index_pcoll
        | "Beam PL: apply schema to sorted_train_frame_sequences_index" >> beam.Map(
                lambda sorted_train_frame_sequences_index_pcoll_row_tpl: beam.Row(
                    # SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX = [
                    #     'TokenID',
                    #     'CameraPerspective',
                    #     'ASLConsultantID',
                    #     'TargetVideoFilename',
                    #     'UtteranceSequence',
                    #     'TokenSequence',
                    #     'FrameSequence'
                    # ]
                    TokenID=int(sorted_train_frame_sequences_index_pcoll_row_tpl[0]),
                    CameraPerspective=int(sorted_train_frame_sequences_index_pcoll_row_tpl[1]),
                    ASLConsultantID=int(sorted_train_frame_sequences_index_pcoll_row_tpl[2]),
                    TargetVideoFilename=str(sorted_train_frame_sequences_index_pcoll_row_tpl[3]),
                    UtteranceSequence=int(sorted_train_frame_sequences_index_pcoll_row_tpl[4]),
                    TokenSequence=int(sorted_train_frame_sequences_index_pcoll_row_tpl[5]),
                    FrameSequence=int(sorted_train_frame_sequences_index_pcoll_row_tpl[6])
                )
            )
        | beam.Map(lambda sorted_train_frame_sequences_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(sorted_train_frame_sequences_index_schemad_pcoll_row))
    )
    return beam__common.pl__X__write_pcoll_to_csv(
        sorted_train_frame_sequences_index_csv_rows_pcoll, 
        "TRAIN-FRAME-SEQUENCES-INDEX", 
        fidscs_globals.TRAIN_FRAME_SEQ_DS_FNAME, 
        fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX
    ) # train_frame_sequences_index_csv_path


def pl__6__create_complete_utterances_from_train_val_tokens(train_val_tcpctvustsfs, tcpdctvustsfs, dcusstettlstrs, token_map_by_id, name_train_val_tcpctvustsfs):
    """
    tcpdctvustsfs:
        (
            <TokenID>,
            <CameraPerspective>,
            <DocumentID>,
            <ASLConsultantID>,
            <TargetVideoFilename>,
            <UtteranceSequence>,
            <TokenSequence>,
            <FrameSequence>
        )

    dcusstettlstrs:
        (
            <DocumentID>,
            <ASLConsultantID>,
            <UtteranceSequence>,
            <StartTime>,
            <EndTime>,
            <Tokens>,
            <Translation>
        )

    token_map_by_id:
        listof(<Token>) # indexed by <TokenID>


    returns:
        complete_utterances__with__train_val_tcp
            tuples of the form:
                (
                    <DocumentID>,
                    <ASLConsultantID>,
                    <TargetVideoFilename>,
                    <UtteranceSequence>,
                    <CameraPerspective>,
                    listof(<TokenID>)
                )
    """

    train_val_tcp = (
        train_val_tcpctvustsfs
        | f"Beam PL: extract (TokenID, CameraPerspective) from {name_train_val_tcpctvustsfs}" >> beam.Map(
                lambda tpl: (
                    tpl[0], # TokenID
                    tpl[1]  # CameraPerspective
                )
            )
        | f"Beam PL: select distinct (TokenID, CameraPerspective) from {name_train_val_tcpctvustsfs}" >> beam.Distinct()
    )

    tvcptidl__by__dcus = (
        tcpdctvustsfs
        | f"Beam PL: extract (DocumentID,ASLConsultantID,TargetVideoFilename,CameraPerspective,UtteranceSequence,TokenSequence,TokenID) from tcpdctvustsfs for {name_train_val_tcpctvustsfs}" >> beam.Map(
                lambda tpl: (
                    tpl[2], # <DocumentID>
                    tpl[3], # <ASLConsultantID>
                    tpl[4], # <TargetVideoFilename>
                    tpl[5], # <UtteranceSequence>
                    tpl[1], # <CameraPerspective>

                    tpl[6], # <TokenSequence>
                    tpl[0]  # <TokenID>
                )
            )
        | f"Beam PL: select distinct (DocumentID,ASLConsultantID,TargetVideoFilename,CameraPerspective,UtteranceSequence,TokenSequence,TokenID) from tcpdctvustsfs for {name_train_val_tcpctvustsfs}" >> beam.Distinct()
        | f"Beam PL: transform distinct dctvcpustst tuples to tst_by_dctvuscp for {name_train_val_tcpctvustsfs}" >> beam.Map(
                lambda tpl: (
                    (
                        tpl[0], # <DocumentID>
                        tpl[1], # <ASLConsultantID>
                        tpl[2], # <TargetVideoFilename>
                        tpl[3], # <UtteranceSequence>
                        tpl[4]  # <CameraPerspective>
                    ),
                    (
                        tpl[5], # <TokenSequence>
                        tpl[6]  # <TokenID>
                    )
                )
            )
        | f"Beam PL: collect list of tokenseq-tokenid for each (<DocumentID>, <ASLConsultantID>, <TargetVideoFilename>, <UtteranceSequence>, <CameraPerspective>) for {name_train_val_tcpctvustsfs}" >> beam.GroupByKey()
        # the above produces tuples of the form:
            # (
            #     (<DocumentID>,<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>), # key
            #     listof((<TokenSequence>,<TokenID>))
            # )
        | f"Beam PL: sort list of tokenseq-tokenid by tokenseq for each (<DocumentID>, <ASLConsultantID>, <TargetVideoFilename>, <UtteranceSequence>, <CameraPerspective>) for {name_train_val_tcpctvustsfs}" >> beam.Map(
                lambda tpl: (
                    (
                        tpl[0][0], # <DocumentID>
                        tpl[0][1], # <ASLConsultantID>
                        tpl[0][2], # <TargetVideoFilename>
                        tpl[0][3], # <UtteranceSequence>
                        tpl[0][4]  # <CameraPerspective>
                    ),
                    [(tst_tpl[1], tpl[0][3]) for tst_tpl in sorted(tpl[1], key=lambda tst_tpl: tst_tpl[0])]
                )
            )
        # the above produces tuples of the form:
            # (
            #     (<DocumentID>,<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>), # key
            #     listof((<TokenID>, <CameraPerspective>)) # sorted by <TokenSequence>
            # )
        # now we need to filter all of the above (<DocumentID>,<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>) where every (<TokenID>, <CameraPerspective>) in the corresponding list exists in train_val_tcp
        | f"Beam PL: filter matching token-cameraperspective tuples for {name_train_val_tcpctvustsfs} from tcpdctvustsfs" >> beam.Filter(
            lambda list_tcp_tpl__by__dctvuscp__tpl, existing_train_val_tcp_tpls: all(tcp_tpl in existing_train_val_tcp_tpls for tcp_tpl in list_tcp_tpl__by__dctvuscp__tpl[1]),
            existing_train_val_tcp_tpls=beam.pvalue.AsIter(train_val_tcp)
        )
        | f"Beam PL: extract (<DocumentID>,<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>,listof(<TokenID>)) for {name_train_val_tcpctvustsfs}" >> beam.Map(
                lambda tpl: (
                    (
                        tpl[0][0],  # <DocumentID>
                        tpl[0][1],  # <ASLConsultantID>
                        tpl[0][3]   # <UtteranceSequence>
                    ),
                    (
                        tpl[0][2],  # <TargetVideoFilename>
                        tpl[0][4],  # <CameraPerspective>
                        [tcp_tpl[0] for tcp_tpl in tpl[1]] # listof(<TokenID>)
                    )
                ) 
            )
        # debug
        # | f"Beam PL: print complete_utterances__with__train_val_tcp for {name_train_val_tcpctvustsfs}" >> beam.ParDo(beam__common.PipelinePcollPrinter(f"complete_utterance (for {name_train_val_tcpctvustsfs}) entry"))
    )

    stettlstrs__by__dcus = (
        dcusstettlstrs
        | f"Beam PL: transform dcusstettlstrs to ((<DocumentID>,<ASLConsultantID>,<UtteranceSequence>),(<StartTime>,<EndTime>,<Tokens>,<Translation>)) for {name_train_val_tcpctvustsfs}" >> beam.Map(
                lambda stettlstrs_tpl: (
                    (
                        stettlstrs_tpl[0],  # <DocumentID>
                        stettlstrs_tpl[1],  # <ASLConsultantID>
                        stettlstrs_tpl[2]   # <UtteranceSequence>
                    ),
                    (
                        stettlstrs_tpl[3],  # <StartTime>
                        stettlstrs_tpl[4],  # <EndTime>
                        stettlstrs_tpl[5],  # <Tokens>          # note that this is a literal string of the concatenated ascii tokens
                        stettlstrs_tpl[6]   # <Translation>     # this is the English translation of the ascii string representation of concatenated (ordered) tokens
                    )
                )
            )
    )

    def resolve_tokens(ctvuscpstetlsttidlsttstrtrstr_tpl, token_map_by_id):
        """
        ctvuscpstetlsttidlsttstrtrstr_tpl:
            (
                <ASLConsultantID>, 
                <TargetVideoFilename>, 
                <UtteranceSequence>, 
                <CameraPerspective>, 
                <StartTime>, 
                <EndTime>, 
                listof(<TokenID>), 
                <Tokens>, 
                <Translation>
            )
        """
        resolved_tokens = ' '.join([token_map_by_id[tid] for tid in ctvuscpstetlsttidlsttstrtrstr_tpl[6]])
        resolved_tokens = resolved_tokens.replace("\\", "")
        return (
            ctvuscpstetlsttidlsttstrtrstr_tpl[0],                                               # <ASLConsultantID>
            ctvuscpstetlsttidlsttstrtrstr_tpl[1],                                               # <TargetVideoFilename>
            ctvuscpstetlsttidlsttstrtrstr_tpl[2],                                               # <UtteranceSequence>
            ctvuscpstetlsttidlsttstrtrstr_tpl[3],                                               # <CameraPerspective>
            ctvuscpstetlsttidlsttstrtrstr_tpl[4],                                               # <StartTime>
            ctvuscpstetlsttidlsttstrtrstr_tpl[5],                                               # <EndTime>
            ' '.join([str(tid) for tid in ctvuscpstetlsttidlsttstrtrstr_tpl[6]]),               # space-delimited string of ordered <TokenID> (avoids the need to encode ',' when we write to CSV)
            resolved_tokens,                                                                    # resolved token string
            ctvuscpstetlsttidlsttstrtrstr_tpl[7],                                               # check token string (retrieved directly from corpus document during data extraction phase)
            ctvuscpstetlsttidlsttstrtrstr_tpl[8]                                                # English translation
        )

    complete_utterances__with__train_val_tcp = (
        ({
            'tvcptidl': tvcptidl__by__dcus,
            'stettlstrs': stettlstrs__by__dcus
        })
        | f"Beam PL: join tvcptidl to stettlstrs for {name_train_val_tcpctvustsfs}" >> beam.CoGroupByKey()
        # the above produces tuples of the form:
            # (
            #     (
            #         <DocumentID>,
            #         <ASLConsultantID>,
            #         <UtteranceSequence>
            #     ),
            #     {
            #         'tvcptidl': listof((<TargetVideoFilename>,<CameraPerspective>,listof(<TokenID>)),     # there will be as many as there are camera perspectives for this utterance-targetvideo combo
            #         'stettlstrs': listof((<StartTime>,<EndTime>,<Tokens>,<Translation>))                  # should only be one
            #     }
            # )
        | f"Beam PL: filter out tuples with empty 'tvcptidl' or 'stettlstrs' lists for {name_train_val_tcpctvustsfs}" >> beam.Filter(
                lambda tvcptidl__joined__stettlstrs__tpl: len(tvcptidl__joined__stettlstrs__tpl[1]['tvcptidl'])>0 and len(tvcptidl__joined__stettlstrs__tpl[1]['stettlstrs'])>0
            )
        | f"Beam PL: explode listof((<TargetVideoFilename>,<CameraPerspective>,listof(<TokenID>)) of tvcptidl__joined__stettlstrs for {name_train_val_tcpctvustsfs}" >> beam.Map(
                lambda tvcptidl__joined__stettlstrs__tpl: [
                    (
                        tvcptidl__joined__stettlstrs__tpl[0][1],                    # <ASLConsultantID>
                        tvcptidl_tpl[0],                                            # <TargetVideoFilename>
                        tvcptidl__joined__stettlstrs__tpl[0][2],                    # <UtteranceSequence>
                        tvcptidl_tpl[1],                                            # <CameraPerspective>
                        tvcptidl__joined__stettlstrs__tpl[1]['stettlstrs'][0][0],   # <StartTime>
                        tvcptidl__joined__stettlstrs__tpl[1]['stettlstrs'][0][1],   # <EndTime>
                        tvcptidl_tpl[2],                                            # listof(<TokenID>)
                        tvcptidl__joined__stettlstrs__tpl[1]['stettlstrs'][0][2],   # <Tokens>              # string of concatenated token ascii literals
                        tvcptidl__joined__stettlstrs__tpl[1]['stettlstrs'][0][3]    # <Translation>
                    ) for tvcptidl_tpl in tvcptidl__joined__stettlstrs__tpl[1]['tvcptidl']
                ]
            )
        | f"Beam PL: 'explode' lst_complete_utterances__with__train_val_tcp__tpl for {name_train_val_tcpctvustsfs}" >> beam.FlatMap(
                lambda lst_complete_utterances__with__train_val_tcp__tpl: lst_complete_utterances__with__train_val_tcp__tpl
            )
        | f"Beam PL: resolve tokens for comparison for {name_train_val_tcpctvustsfs}" >> beam.Map(
                resolve_tokens, 
                token_map_by_id=beam.pvalue.AsSingleton(token_map_by_id)
            )
        # the above produces tuples of the form:
            # (
            #     <ASLConsultantID>,
            #     <TargetVideoFilename>,
            #     <UtteranceSequence>,
            #     <CameraPerspective>,
            #     <StartTime>,
            #     <EndTime>,
            #     str(listof(<TokenID>)),                       # space-delimited
            #     <Tokens (resolved)>                           # resolved token string
            #     <Tokens (master, from corpus document)>,      # string of concatenated token ascii literals
            #     <Translation>
            # )

        # debug
        # | f"Beam PL: print complete_utterances__with__train_val_tcp for {name_train_val_tcpctvustsfs}" >> beam.ParDo(beam__common.PipelinePcollPrinter(f"complete_utterance (for {name_train_val_tcpctvustsfs}) entry"))
    )

    complete_utterances__with__train_val_tcp__failed_token_resolution = (
        complete_utterances__with__train_val_tcp
        | f"Beam PL: filter failed token resolution for {name_train_val_tcpctvustsfs}" >> beam.Filter(
                lambda complete_utterances__with__train_val_tcp__tpl: complete_utterances__with__train_val_tcp__tpl[7] != complete_utterances__with__train_val_tcp__tpl[8]
            )
        # NOT DEBUG! THIS IS A LEGIT NOTIFICATION!
        | f"Beam PL: print failed token resolution for {name_train_val_tcpctvustsfs}" >> beam.ParDo(beam__common.PipelinePcollPrinter(f"{fidscs_globals.VALIDATION_WARNING_TEXT} failed token resolution for {name_train_val_tcpctvustsfs} entry"))
    )

    # complete_utterances__with__train_val_tcp = (
    #     complete_utterances__with__train_val_tcp
    #     | f"Beam PL: filter successful token resolution for {name_train_val_tcpctvustsfs}" >> beam.Filter(
    #             lambda complete_utterances__with__train_val_tcp__tpl: complete_utterances__with__train_val_tcp__tpl[7] == complete_utterances__with__train_val_tcp__tpl[8]
    #         )
    # )

    return complete_utterances__with__train_val_tcp, complete_utterances__with__train_val_tcp__failed_token_resolution


def pl__6__write_complete_utterances_from_train_val_tokens(complete_utterances__with__train_val_tcp, name_complete_utterances_train_val_index, fname_complete_utterances_train_val_index):
    """
    complete_utterances__with__train_val_tcp:
        (
            <ASLConsultantID>,
            <TargetVideoFilename>,
            <UtteranceSequence>,
            <CameraPerspective>,
            <StartTime>,
            <EndTime>,
            str(listof(<TokenID>)),                       # space-delimited
            <Tokens (resolved)>                           # resolved token string
            <Tokens (master, from corpus document)>,      # string of concatenated token ascii literals
            <Translation>
        )
    """
    sorted_complete_utterances__with__train_val_tcp = beam__common.pl__X__sort_pcoll(complete_utterances__with__train_val_tcp, pcoll_label=name_complete_utterances_train_val_index)
    sorted_complete_utterances__with__train_val_tcp_csv_rows_pcoll = (
        sorted_complete_utterances__with__train_val_tcp
        | f"Beam PL: apply schema to sorted_complete_utterances__with__train_val_tcp for {name_complete_utterances_train_val_index}" >> beam.Map(
                lambda sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl: beam.Row(
                    # SCHEMA_COL_NAMES__COMPLETE_UTTERANCES_TRAIN_VAL_TCP_INDEX = [
                    #     'ASLConsultantID',
                    #     'TargetVideoFilename',
                    #     'UtteranceSequence',
                    #     'CameraPerspective',
                    #     'StartTime',
                    #     'EndTime',
                    #     'TokenIDs',
                    #     'Tokens',
                    #     'Translation'
                    # ]
                    ASLConsultantID=int(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[0]),
                    TargetVideoFilename=str(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[1]),
                    UtteranceSequence=int(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[2]),
                    CameraPerspective=int(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[3]),
                    StartTime=int(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[4]),
                    EndTime=int(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[5]),
                    TokenIDs=str(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[6]),
                    Tokens=str(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[8]),
                    Translation=str(sorted_complete_utterances__with__train_val_tcp_pcoll_row_tpl[9])
                )
            )
        | f"Beam PL: beam row to csv string for {name_complete_utterances_train_val_index}" >> beam.Map(
                lambda sorted_complete_utterances__with__train_val_tcp_index_schemad_pcoll_row: 
                    beam__common.beam_row_to_csv_string(sorted_complete_utterances__with__train_val_tcp_index_schemad_pcoll_row)
            )
    )
    return beam__common.pl__X__write_pcoll_to_csv(
        sorted_complete_utterances__with__train_val_tcp_csv_rows_pcoll, 
        name_complete_utterances_train_val_index, 
        fname_complete_utterances_train_val_index, 
        fidscs_globals.SCHEMA_COL_NAMES__COMPLETE_UTTERANCES_TRAIN_VAL_TCP_INDEX
    )




def run(data_dir):
    fidscs_globals.DATA_ROOT_DIR = data_dir
    if not tf.io.gfile.exists(fidscs_globals.DATA_ROOT_DIR) or len(tf.io.gfile.listdir(fidscs_globals.DATA_ROOT_DIR))==0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} data directory does not exist or is empty!")
        return
    fidscs_globals.VIDEO_DIR = os.path.join(fidscs_globals.DATA_ROOT_DIR, 'videos')
    fidscs_globals.STICHED_VIDEO_FRAMES_DIR = os.path.join(fidscs_globals.DATA_ROOT_DIR, 'stitched_video_frames')
    fidscs_globals.CORPUS_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.CORPUS_DS_FNAME)
    fidscs_globals.DOCUMENT_ASL_CONSULTANT_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.DOCUMENT_ASL_CONSULTANT_DS_FNAME)
    fidscs_globals.ASL_CONSULTANT_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.ASL_CONSULTANT_DS_FNAME)
    fidscs_globals.VIDEO_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.VIDEO_DS_FNAME)
    fidscs_globals.VIDEO_SEGMENT_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.VIDEO_SEGMENT_DS_FNAME)
    fidscs_globals.VIDEO_FRAME_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.VIDEO_FRAME_DS_FNAME)
    fidscs_globals.UTTERANCE_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.UTTERANCE_DS_FNAME)
    fidscs_globals.UTTERANCE_VIDEO_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.UTTERANCE_VIDEO_DS_FNAME)
    fidscs_globals.UTTERANCE_TOKEN_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.UTTERANCE_TOKEN_DS_FNAME)
    fidscs_globals.UTTERANCE_TOKEN_FRAME_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.UTTERANCE_TOKEN_FRAME_DS_FNAME)
    fidscs_globals.VOCABULARY_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.VOCABULARY_DS_FNAME)
    fidscs_globals.TRAIN_ASSOC_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.TRAIN_FRAME_SEQ_ASSOC_DS_FNAME)
    fidscs_globals.VAL_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.VAL_FRAME_SEQ_DS_FNAME)
    fidscs_globals.TRAIN_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.TRAIN_FRAME_SEQ_DS_FNAME)
    fidscs_globals.COMPLETE_UTTERANCES_TRAIN_ASSOC_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.COMPLETE_UTTERANCES_TRAIN_ASSOC_DS_FNAME)
    fidscs_globals.COMPLETE_UTTERANCES_VAL_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.COMPLETE_UTTERANCES_VAL_DS_FNAME)
    fidscs_globals.COMPLETE_UTTERANCES_TRAIN_DS_PATH = os.path.join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.COMPLETE_UTTERANCES_TRAIN_DS_FNAME)


    if not beam__common.train_val_csv_files_exist():
        options = {
            'project': 'my-project', # change
            'runner': 'DirectRunner',
            'direct_num_workers': 0, # 0 is use all available cores
            'direct_running_mode': 'multi_threading', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
            'streaming': False # set to True if data source is unbounded (e.g. GCP PubSub)
        }
        pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
        print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
        

        with beam.Pipeline(options=pipeline_options) as pl:
            # full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
            # corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl) # XML is base-64 encode but we no longer need it (to decode it) since it is only used to create the datasets
            # # corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll) # see above
            # asl_consultant_index_schemad_pcoll = beam__common.pl__1__read_asl_consultant_index_csv(pl)

            document_asl_consultant_utterance_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_index_csv(pl)
            dcusstettlstrs = (
                document_asl_consultant_utterance_index_schemad_pcoll
                | "Beam PL: extract (DocumentID,ASLConsultantID,UtteranceSequence,StartTime,EndTime,Tokens,Translation) from dcu schemad pcoll" >> beam.Map(
                        lambda dcuiscp_row: (
                            dcuiscp_row.DocumentID,
                            dcuiscp_row.ASLConsultantID,
                            dcuiscp_row.UtteranceSequence,
                            dcuiscp_row.StartTime,
                            dcuiscp_row.EndTime,
                            dcuiscp_row.Tokens,
                            dcuiscp_row.Translation
                        )
                    )
            )

            # document_asl_consultant_target_video_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_index_csv(pl)
            # document_asl_consultant_utterance_video_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_video_index_csv(pl)
            # document_target_video_segment_index_schemad_pcoll = beam__common.pl__1__read_document_target_video_segment_index_csv(pl)

            vocabulary_index_schemad_pcoll = beam__common.pl__1__read_vocabulary_index_csv(pl)
            tidt = (
                vocabulary_index_schemad_pcoll
                | "Beam PL: transform vocabulary_index_schemad_pcoll row to (TokenID, Token)" >> beam.Map(
                        lambda vocabulary_index_schemad_pcoll_row: (
                            vocabulary_index_schemad_pcoll_row.TokenID,
                            str(vocabulary_index_schemad_pcoll_row.Token[2:-1]) # this implicitly converted to a string from bytes, thus we have to remove to preceding "b" and trailing "'"
                        )
                    )
                # debug
                # | f"Beam PL: print tidt" >> beam.ParDo(beam__common.PipelinePcollPrinter(f"tidt entry"))
            )
            tidt = beam__common.pl__X__sort_pcoll(tidt, "tidt")
            token_map_by_id = (
                tidt
                | "Beam PL: extract (1, Token) from sorted tidt pcoll" >> beam.Map(lambda tidt_tpl: (1, tidt_tpl[1]))
                | "Beam PL: group (1, Token) by key (1)" >> beam.GroupByKey()
                | "Beam PL: extract listof(<Token>) (ordered/indexed by <TokenID>)" >> beam.Map(lambda list_token_tpl: list_token_tpl[1])
            )

            # document_asl_consultant_utterance_token_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_token_index_csv(pl)
            # document_asl_consultant_target_video_frame_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_frame_index_csv(pl)

            document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_utterance_token_frame_index_csv(pl)
            tcpdctvustsfs = (
                document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll
                | "Beam PL: extract (TokenID,CameraPerspective,DocumentID,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence,FrameSequence) from dctvustsfs schemad pcoll" >> beam.Map(
                        lambda dctvustsfs_row: (
                            dctvustsfs_row.TokenID,
                            dctvustsfs_row.CameraPerspective,
                            dctvustsfs_row.DocumentID,
                            dctvustsfs_row.ASLConsultantID,
                            dctvustsfs_row.TargetVideoFilename,
                            dctvustsfs_row.UtteranceSequence,
                            dctvustsfs_row.TokenSequence,
                            dctvustsfs_row.FrameSequence
                        )
                    )
            )


            (
                train_val_split_candidates_keys, 
                train_val_split_NON_candidates_keys,
                tcpctvustsfs
            ) = pl__2__get_keys__train_val_split_candidates(
                document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll
            )

            (
                train__keys, 
                val__keys
            ) = pl__3__do_train_val_split_keys(train_val_split_candidates_keys)

            (
                train_frame_sequences__assoc,   # this is the initial set of training frames with (<TokenID>, <CameraPerspective>) tuples corresponding to at least one observation in the eventual val_frame_sequences set
                frame_sequences__by__tcpctvustsfs
            ) = pl__4__create_train_frame_sequences__assoc(
                train__keys, 
                tcpctvustsfs
            )
            pl__5__write_train_frame_sequences__assoc_index_csv(train_frame_sequences__assoc)

            val_frame_sequences = pl__5__create_val_frame_sequences(
                val__keys, 
                frame_sequences__by__tcpctvustsfs
            )
            pl__6__write_val_frame_sequences_index_csv(val_frame_sequences)

            # this step of the pipeline creates the final train_frame_sequences set
                # which is the union of train_frame_sequences__assoc (from above) and those produced from train_val_split_NON_candidates_keys
                # ultimately we train on some frame sequences that cannot be validated (but we still want to be able to offer some predictive capability based on them)
            train_frame_sequences = pl__5__create_train_frame_sequences(
                train_val_split_NON_candidates_keys, 
                frame_sequences__by__tcpctvustsfs, 
                train_frame_sequences__assoc
            )
            pl__6__write_train_frame_sequences_index_csv(train_frame_sequences)

            complete_utterances__from__train_assoc_tokens, _ = pl__6__create_complete_utterances_from_train_val_tokens(
                train_frame_sequences__assoc, 
                tcpdctvustsfs,
                dcusstettlstrs,
                token_map_by_id,
                name_train_val_tcpctvustsfs="train_frame_sequences__assoc"
            )
            pl__6__write_complete_utterances_from_train_val_tokens(
                complete_utterances__from__train_assoc_tokens, 
                "COMPLETE-UTTERANCES-TRAIN-ASSOC-INDEX", 
                fidscs_globals.COMPLETE_UTTERANCES_TRAIN_ASSOC_DS_FNAME
            )

            complete_utterances__from__val_tokens, _ = pl__6__create_complete_utterances_from_train_val_tokens(
                val_frame_sequences, 
                tcpdctvustsfs,
                dcusstettlstrs,
                token_map_by_id,
                name_train_val_tcpctvustsfs="val_frame_sequences"
            )
            pl__6__write_complete_utterances_from_train_val_tokens(
                complete_utterances__from__val_tokens, 
                "COMPLETE-UTTERANCES-VAL-INDEX", 
                fidscs_globals.COMPLETE_UTTERANCES_VAL_DS_FNAME
            )

            complete_utterances__from__train_tokens, _ = pl__6__create_complete_utterances_from_train_val_tokens(
                train_frame_sequences, 
                tcpdctvustsfs,
                dcusstettlstrs,
                token_map_by_id,
                name_train_val_tcpctvustsfs="train_frame_sequences"
            )
            pl__6__write_complete_utterances_from_train_val_tokens(
                complete_utterances__from__train_tokens, 
                "COMPLETE-UTTERANCES-TRAIN-INDEX", 
                fidscs_globals.COMPLETE_UTTERANCES_TRAIN_DS_FNAME
            )




# **************************************** main: BEGIN ****************************************
if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
    '--work-dir',
    required=True,
    help='Directory for staging and working files. '
          'This can be a Google Cloud Storage path.'
  )

  args = parser.parse_args()
  print(f"args: {args}")
  run(
    os.path.join(args.work_dir, 'data')
  )
  # **************************************** main: END ****************************************
