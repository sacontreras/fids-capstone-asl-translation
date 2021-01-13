import argparse
import os

import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

import beam__common
import globals

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
    globals.DATA_ROOT_DIR = data_dir
    if not tf.io.gfile.exists(globals.DATA_ROOT_DIR) or len(tf.io.gfile.listdir(globals.DATA_ROOT_DIR))==0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} data directory does not exist or is empty!")
        return
    globals.VIDEO_DIR = os.path.join(globals.DATA_ROOT_DIR, 'videos')
    globals.STICHED_VIDEO_FRAMES_DIR = os.path.join(globals.DATA_ROOT_DIR, 'stitched_video_frames')
    globals.CORPUS_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.CORPUS_DS_FNAME)
    globals.DOCUMENT_ASL_CONSULTANT_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.DOCUMENT_ASL_CONSULTANT_DS_FNAME)
    globals.ASL_CONSULTANT_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.ASL_CONSULTANT_DS_FNAME)
    globals.VIDEO_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_DS_FNAME)
    globals.VIDEO_SEGMENT_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_SEGMENT_DS_FNAME)
    globals.VIDEO_FRAME_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_FRAME_DS_FNAME)
    globals.UTTERANCE_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_DS_FNAME)
    globals.UTTERANCE_VIDEO_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_VIDEO_DS_FNAME)
    globals.UTTERANCE_TOKEN_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_TOKEN_DS_FNAME)
    globals.UTTERANCE_TOKEN_FRAME_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_TOKEN_FRAME_DS_FNAME)
    globals.VOCABULARY_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.VOCABULARY_DS_FNAME)


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
