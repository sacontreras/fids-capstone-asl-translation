import argparse
import os

import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow_transform as tft

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




def _make_train_or_eval_input_fn(
  feature_spec, 
  labels, 
  file_pattern, 
  batch_size, 
  mode, 
  shuffle=True
):
  def input_fn():
    def decode(elem):
      model_features = tf.parse_single_example(elem, features=feature_spec)
      model_labels = tf.stack([model_features.pop(label) for label in labels])
      return model_features, model_labels

    # For more information, check:
    # https://www.tensorflow.org/performance/datasets_performance
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.apply(
      tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, 
        cycle_length=mp.cpu_count()
      )
    )
    dataset = dataset.map(decode, num_parallel_calls=mp.cpu_count())
    dataset = dataset.take(-1)
    if mode == tf.estimator.ModeKeys.TRAIN:
      if shuffle:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(batch_size * 8))
      else:
        dataset = dataset.cache()
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

  return input_fn

def make_train_input_fn(
  feature_spec, 
  labels, 
  file_pattern, 
  batch_size, 
  shuffle=True
):
  """Makes an input_fn for training."""
  return _make_train_or_eval_input_fn(
    feature_spec,
    labels,
    file_pattern,
    batch_size,
    tf.estimator.ModeKeys.TRAIN,
    shuffle
  )


def make_serving_input_fn(
    tft_output, input_feature_spec, labels):
  """Makes an input_fn for serving prediction.

  This will use the inputs format produced by the preprocessing PTransform. This
  applies the transformations from the tf.Transform preprocessing_fn before
  serving it to the TensorFlow model.
  """
  def serving_input_fn():
    input_features = {}
    for feature_name in input_feature_spec:
      if feature_name in labels:
        continue
      dtype = input_feature_spec[feature_name].dtype
      input_features[feature_name] = tf.placeholder(
        dtype, 
        shape=[None], 
        name=feature_name
      )

    inputs = tft_output.transform_raw_features(input_features)

    return tf.estimator.export.ServingInputReceiver(inputs, input_features)

  return serving_input_fn


def make_eval_input_fn(feature_spec, labels, file_pattern, batch_size):
  """Makes an input_fn for evaluation."""
  return _make_train_or_eval_input_fn(
    feature_spec,
    labels,
    file_pattern,
    batch_size,
    tf.estimator.ModeKeys.EVAL
  )


def train_and_evaluate(
    work_dir,
    input_feature_spec,
    labels,
    train_files_pattern,
    eval_files_pattern,
    batch_size=64,
    train_max_steps=1000):
  """
  Trains and evaluates the estimator given.

  The input functions are generated by the preprocessing function.
  """

  model_dir = os.path.join(work_dir, 'model')
  if tf.gfile.Exists(model_dir):
    tf.gfile.DeleteRecursively(model_dir)

  # Specify where to store our model
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=model_dir)

  # This will give us a more granular visualization of the training
  run_config = run_config.replace(save_summary_steps=10)



  # ORIGINAL (from example), but it looks like I need: tf.estimator.DNNEstimator
  # Create a Deep Neural Network Regressor estimator
  estimator = tf.estimator.DNNRegressor(
    feature_columns=[
      tf.feature_column.numeric_column('NormalizedC', dtype=tf.float32),
      tf.feature_column.numeric_column('NormalizedH', dtype=tf.float32),
      tf.feature_column.numeric_column('NormalizedO', dtype=tf.float32),
      tf.feature_column.numeric_column('NormalizedN', dtype=tf.float32),
    ],
    hidden_units=[128, 64],
    dropout=0.5,
    config=run_config
  )
  # estimator = tf.estimator.DNNEstimator() # my contribution but likely is not correct



  # Get the transformed feature_spec
  tft_output = tft.TFTransformOutput(work_dir)
  feature_spec = tft_output.transformed_feature_spec()

  # Create the training and evaluation specifications
  train_spec = tf.estimator.TrainSpec(
    input_fn=make_train_input_fn(
      feature_spec, 
      labels, 
      train_files_pattern, 
      batch_size
    ),
    max_steps=train_max_steps
  )

  exporter = tf.estimator.FinalExporter(
    'final', 
    make_serving_input_fn(
      tft_output, 
      input_feature_spec, 
      labels
    )
  )

  eval_spec = tf.estimator.EvalSpec(
    input_fn=make_eval_input_fn(
      feature_spec, 
      labels, 
      eval_files_pattern, 
      batch_size
    ),
    exporters=[exporter]
  )

  # Train and evaluate the model
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)




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
