import argparse
import os

import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam.runners.interactive.interactive_beam as ib

import beam__common
import fidscs_globals

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
        full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
        corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl) # XML is base-64 encode but we no longer need it (to decode it) since it is only used to create the datasets
        # corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll) # see above
        asl_consultant_index_schemad_pcoll = beam__common.pl__1__read_asl_consultant_index_csv(pl)
        document_asl_consultant_utterance_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_index_csv(pl)
        document_asl_consultant_target_video_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_index_csv(pl)
        document_asl_consultant_utterance_video_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_video_index_csv(pl)
        document_target_video_segment_index_schemad_pcoll = beam__common.pl__1__read_document_target_video_segment_index_csv(pl)
        vocabulary_index_schemad_pcoll = beam__common.pl__1__read_vocabulary_index_csv(pl)
        document_asl_consultant_utterance_token_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_token_index_csv(pl)
        document_asl_consultant_target_video_frame_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_frame_index_csv(pl)
        document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_utterance_token_frame_index_csv(pl)

        """
        document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll is the main table we use for training.
          This will ultimately provide which frame sequences correspond to individual tokens.

        But our first measure is to build to train and validation sets (for tokens).
        """




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
