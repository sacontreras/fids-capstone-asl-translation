import argparse
import os

import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam.runners.interactive.interactive_beam as ib
import apache_beam.transforms.sql

import beam__common
import fidscs_globals
import random




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
                keyed by (TokenID, CameraPerspective) in common with the training set

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
                        frame_seq
                    ) for frame_seq in sorted(joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl[1]['frame_sequences'])
                ]
            )
        | "Beam PL: 'explode' listof((TokenID,CameraPerspective,ASLConsultantID,TargetVideoFilename,UtteranceSequence,TokenSequence, FrameSequence)) from joined train__ctvusts_by_tcp to tcpctvustsfs" >> beam.FlatMap(
                lambda list_joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl: list_joined__train__ctvusts_by_tcp__to__tcpctvustsfs__tpl
            )
    )

    return train_tcpctvustsfs, frame_sequences__by__tcpctvustsfs


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


def pl__6__create_complete_utterances_from_val_tokens(val_tcpctvustsfs, tcpctvustsfs):
    """

    returns:
        complete_utterances__with__val_tcp
    """

    # find all COMPLETE utterances that can be formed with token-cameraperspective pairs from the validation set

    val_tcp = (
        val_tcpctvustsfs
        | "Beam PL: extract (TokenID, CameraPerspective) from val_tcpctvustsfs" >> beam.Map(
                lambda tpl: (
                    tpl[0], # TokenID
                    tpl[1]  # CameraPerspective
                )
            )
        | "Beam PL: select distinct (TokenID, CameraPerspective) from val_tcpctvustsfs" >> beam.Distinct()
    )

    complete_utterances__with__val_tcp = (
        tcpctvustsfs
        | "Beam PL: extract (ASLConsultantID,TargetVideoFilename,CameraPerspective,UtteranceSequence,TokenSequence,TokenID) from tcpctvustsfs" >> beam.Map(
                lambda tpl: (
                    tpl[2], # <ASLConsultantID>
                    tpl[3], # <TargetVideoFilename>
                    tpl[4], # <UtteranceSequence>
                    tpl[1], # <CameraPerspective>

                    tpl[5], # <TokenSequence>
                    tpl[0]  # <TokenID>
                )
            )
        | "Beam PL: select distinct (ASLConsultantID,TargetVideoFilename,CameraPerspective,UtteranceSequence,TokenSequence,TokenID) from tcpctvustsfs" >> beam.Distinct()
        | "Beam PL: transform distinct ctvcpustst tuples to tst_by_ctvuscp" >> beam.Map(
                lambda tpl: (
                    (
                        tpl[0], # <ASLConsultantID>
                        tpl[1], # <TargetVideoFilename>
                        tpl[2], # <UtteranceSequence>
                        tpl[3]  # <CameraPerspective>
                    ),
                    (
                        tpl[4], # <TokenSequence>
                        tpl[5]  # <TokenID>
                    )
                )
            )
        | "Beam PL: collect list of tokenseq-tokenid for each (<ASLConsultantID>, <TargetVideoFilename>, <UtteranceSequence>, <CameraPerspective>)" >> beam.GroupByKey()
        # the above produces tuples of the form:
            # (
            #     (<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>), # key
            #     listof((<TokenSequence>,<TokenID>))
            # )
        | "Beam PL: sort list of tokenseq-tokenid by tokenseq for each (<ASLConsultantID>, <TargetVideoFilename>, <UtteranceSequence>, <CameraPerspective>)" >> beam.Map(
                lambda tpl: (
                    (
                        tpl[0][0], # <ASLConsultantID>
                        tpl[0][1], # <TargetVideoFilename>
                        tpl[0][2], # <UtteranceSequence>
                        tpl[0][3]  # <CameraPerspective>
                    ),
                    [(tst_tpl[1], tpl[0][3]) for tst_tpl in sorted(tpl[1], key=lambda tst_tpl: tst_tpl[0])]
                )
            )
        # the above produces tuples of the form:
            # (
            #     (<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>), # key
            #     listof((<TokenID>, <CameraPerspective>)) # sorted by <TokenSequence>
            # )

        # now we need to filter all of the above (<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>) where every (<TokenID>, <CameraPerspective>) in the corresponding list exists in val_tcp
        | "Beam PL: filter matching rows from vid index" >> beam.Filter(
            lambda list_tcp_tpl__by__ctvuscp__tpl, existing_val_tcp_tpls: all(tcp_tpl in existing_val_tcp_tpls for tcp_tpl in list_tcp_tpl__by__ctvuscp__tpl[1]),
            existing_val_tcp_tpls=beam.pvalue.AsIter(val_tcp)
        )
        | "Beam PL: extract (<ASLConsultantID>,<TargetVideoFilename>,<UtteranceSequence>,<CameraPerspective>,listof(<TokenID>))" >> beam.Map(
                lambda tpl: (
                    tpl[0][0],  # <ASLConsultantID>
                    tpl[0][1],  # <TargetVideoFilename>
                    tpl[0][2],  # <UtteranceSequence>
                    tpl[0][3],  # <CameraPerspective>
                    [tcp_tpl[0] for tcp_tpl in tpl[1]] # listof(<TokenID>)
                ) 
            )
        # debug
        | "Beam PL: print complete_utterances__with__val_tcp" >> beam.ParDo(beam__common.PipelinePcollPrinter("complete_utterances__with__val_tcp entry"))
    )

    return complete_utterances__with__val_tcp




options = {
    'project': 'my-project', # change
    'runner': 'DirectRunner',
    'direct_num_workers': 0, # 0 is use all available cores
    'direct_running_mode': 'multi_threading', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
    'streaming': False # set to True if data source is unbounded (e.g. GCP PubSub)
}
pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")

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


    with beam.Pipeline(options=pipeline_options) as pl:
        # full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
        # corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl) # XML is base-64 encode but we no longer need it (to decode it) since it is only used to create the datasets
        # # corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll) # see above
        # asl_consultant_index_schemad_pcoll = beam__common.pl__1__read_asl_consultant_index_csv(pl)
        # document_asl_consultant_utterance_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_index_csv(pl)
        # document_asl_consultant_target_video_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_index_csv(pl)
        # document_asl_consultant_utterance_video_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_video_index_csv(pl)
        # document_target_video_segment_index_schemad_pcoll = beam__common.pl__1__read_document_target_video_segment_index_csv(pl)
        # vocabulary_index_schemad_pcoll = beam__common.pl__1__read_vocabulary_index_csv(pl)
        # document_asl_consultant_utterance_token_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_token_index_csv(pl)
        # document_asl_consultant_target_video_frame_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_frame_index_csv(pl)
        document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_utterance_token_frame_index_csv(pl)


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

        val_frame_sequences = pl__5__create_val_frame_sequences(
            val__keys, 
            frame_sequences__by__tcpctvustsfs
        )

        # this step of the pipeline creates the final train_frame_sequences set
            # which is the union of train_frame_sequences__assoc (from above) and those produced from train_val_split_NON_candidates_keys
            # ultimately we train on some frame sequences that cannot be validated (but we still want to be able to offer some predictive capability based on them)
        train_frame_sequences = pl__5__create_train_frame_sequences(
            train_val_split_NON_candidates_keys, 
            frame_sequences__by__tcpctvustsfs, 
            train_frame_sequences__assoc
        )

        complete_utterances__from__val_tokens = pl__6__create_complete_utterances_from_val_tokens(
            val_frame_sequences, 
            tcpctvustsfs
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