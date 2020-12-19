#!/usr/bin/env python

# Copyright 2019 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import
import os
import argparse
import numpy as np
import tensorflow as tf
from importlib import import_module
# import signstreamxmlparser.analysis as sxa
# import signstreamxmlparser.analysis.signstream as ss
sxa = import_module('.analysis', 'signstreamxmlparser-refactored')
ss = import_module('.signstream', 'signstreamxmlparser-refactored.analysis')
import cv2
import globals




def run(max_data_files, data_dir, use_beam=False):
  print(f"use_beam: {use_beam}")

  # see https://www.tensorflow.org/tutorials/distribute/keras, https://www.tensorflow.org/guide/distributed_training
  #   tf.distribute.MirroredStrategy 
  #     ... supports synchronous distributed training on multiple GPUs on one machine.
  #     It creates one replica per GPU device. Each variable in the model is mirrored across all the replicas.
  #     These variables are kept in sync with each other by applying identical updates. 
  #     Efficient all-reduce algorithms are used to communicate the variable updates across the devices. 
  #     All-reduce aggregates tensors across all the devices by adding them up, and makes them available on each device. 
  #     It’s a fused algorithm that is very efficient and can reduce the overhead of synchronization significantly. 
  #     There are many all-reduce algorithms and implementations available, depending on the type of communication available between devices. 
  #     By default, it uses NVIDIA NCCL as the all-reduce implementation. You can choose from a few other options, or write your own.
  # strategy = tf.distribute.MirroredStrategy()

  #   tf.distribute.experimental.MultiWorkerMirroredStrategy
  #     ... is very similar to MirroredStrategy. It implements synchronous distributed training across multiple workers, each with potentially multiple GPUs. 
  #     Similar to MirroredStrategy, it creates copies of all variables in the model on each device across all workers.
  #     It uses CollectiveOps as the multi-worker all-reduce communication method used to keep variables in sync. 
  #     A collective op is a single op in the TensorFlow graph which can automatically choose an all-reduce algorithm in the TensorFlow runtime according to hardware, network topology and tensor sizes.
  #     It also implements additional performance optimizations. 
  #     For example, it includes a static optimization that converts multiple all-reductions on small tensors into fewer all-reductions on larger tensors. 
  #     In addition, it is designed to have a plugin architecture - so that in the future, you will be able to plugin algorithms that are better tuned for your hardware. 
  #     Note that collective ops also implement other collective operations such as broadcast and all-gather.
  #     MultiWorkerMirroredStrategy currently allows you to choose between two different implementations of collective ops. 
  #     CollectiveCommunication.RING implements ring-based collectives using gRPC as the communication layer. 
  #     CollectiveCommunication.NCCL uses Nvidia's NCCL to implement collectives. CollectiveCommunication.AUTO defers the choice to the runtime.
  #     The best choice of collective implementation depends upon the number and kind of GPUs, and the network interconnect in the cluster.
  # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  #   tf.distribute.experimental.CentralStorageStrategy 
  #     ... does synchronous training as well. Variables are not mirrored, instead they are placed on the CPU and operations are replicated across all local GPUs. 
  #     If there is only one GPU, all variables and operations will be placed on that GPU.
  # strategy = tf.distribute.experimental.CentralStorageStrategy()

  #   tf.distribute.OneDeviceStrategy
  #     ... is a strategy to place all variables and computation on a single specified device. This strategy is distinct from the default strategy in a number of ways. 
  #     In default strategy, the variable placement logic remains unchanged when compared to running TensorFlow without any distribution strategy. 
  #     But when using OneDeviceStrategy, all variables created in its scope are explicitly placed on the specified device. 
  #     Moreover, any functions called via OneDeviceStrategy.run will also be placed on the specified device. 
  #     Input distributed through this strategy will be prefetched to the specified device. In default strategy, there is no input distribution.
  #     Similar to the default strategy, this strategy could also be used to test your code before switching to other strategies which actually distribute to multiple devices/machines. 
  #     This will exercise the distribution strategy machinery somewhat more than default strategy, but not to the full extent as using MirroredStrategy or TPUStrategy etc. 
  #     If you want code that behaves as if no strategy, then use default strategy. 

  # if tf.config.list_physical_devices('gpu'):
  #   # strategy = tf.distribute.MirroredStrategy()
  #   strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  #   # tf.distribute.experimental.CentralStorageStrategy()
  # else:  # use default strategy
  #   strategy = tf.distribute.get_strategy()
  # strategy = tf.distribute.experimental.CentralStorageStrategy() 
  # strategy = tf.distribute.MirroredStrategy()
  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  print(f'Number of devices available for parallel processing: {strategy.num_replicas_in_sync}')

  # ******************** global variables set at runtime: BEGIN ********************
  globals.MAX_DATA_FILES = max_data_files

  globals.DATA_ROOT_DIR = data_dir
  if not tf.io.gfile.exists(globals.DATA_ROOT_DIR):
    tf.io.gfile.makedirs(globals.DATA_ROOT_DIR)
  if not tf.io.gfile.exists(globals.TMP_DIR):
    tf.io.gfile.makedirs(globals.TMP_DIR)

  globals.VIDEO_DIR = os.path.join(globals.DATA_ROOT_DIR, 'videos')
  if not tf.io.gfile.exists(globals.VIDEO_DIR):
    tf.io.gfile.makedirs(globals.VIDEO_DIR)

  globals.STICHED_VIDEO_FRAMES_DIR = os.path.join(globals.DATA_ROOT_DIR, 'stitched_video_frames')
  if not tf.io.gfile.exists(globals.STICHED_VIDEO_FRAMES_DIR):
    tf.io.gfile.makedirs(globals.STICHED_VIDEO_FRAMES_DIR)

  globals.CORPUS_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.CORPUS_DS_FNAME)

  globals.DOCUMENT_ASL_CONSULTANT_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.DOCUMENT_ASL_CONSULTANT_DS_FNAME)

  globals.ASL_CONSULTANT_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.ASL_CONSULTANT_DS_FNAME)

  globals.VIDEO_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_DS_FNAME)

  globals.UTTERANCE_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_DS_FNAME)

  globals.UTTERANCE_VIDEO_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_VIDEO_DS_FNAME)

  globals.UTTERANCE_TOKEN_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_TOKEN_DS_FNAME)

  globals.VOCABULARY_DS_PATH = os.path.join(globals.DATA_ROOT_DIR, globals.VOCABULARY_DS_FNAME)
  # ******************** global variables set at runtime: END ********************

  if use_beam:
    import preprocessor__beam
    preprocessor__beam.run()
  else:
    import preprocessor__pandas
    preprocessor__pandas.run()





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

  parser.add_argument(
    '--max-data-files',
    type=int,
    default=-1,
    help='Maximum number of data files for every file pattern expansion. '
          'Set to -1 to use all files.'
  )

  # courtesy of https://stackoverflow.com/a/43357954
  #   script --use_beam
  #   script --use_beam <bool>
  def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
  parser.add_argument(
    "--use-beam", 
    type=str2bool, 
    nargs='?',
    const=True, 
    default=True,
    help=""
  )

  args = parser.parse_args()
  print(f"args: {args}")
  run(
    args.max_data_files if args.max_data_files!=-1 else None, 
    os.path.join(args.work_dir, 'data'), 
    use_beam=args.use_beam
  )
  # **************************************** main: END ****************************************