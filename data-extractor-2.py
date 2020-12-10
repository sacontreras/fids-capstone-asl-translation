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

import argparse
import ftplib
import multiprocessing as mp
# from multiprocessing.sharedctypes import Value, Array
import os
import re
import signal
import tempfile
import tensorflow as tf
import zlib
import zipfile
# from io import BytesIO
import pandas as pd
import urllib
import utils
# from . import utils

import subprocess
import sys
def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import imp

try:
  imp.find_module('apache_beam')
except ImportError:
  pip_install("apache-beam[gcp]") # !pip install apache-beam[gcp]
import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.options.pipeline_options import PipelineOptions
import csv
import io
import typing
from apache_beam.transforms.sql import SqlTransform

try:
  imp.find_module('cv2')
except ImportError:
  pip_install("opencv-python") # !pip install opencv-python
import cv2


def _function_wrapper(args_tuple):
  """Function wrapper to call from multiprocessing."""
  function, args = args_tuple
  return function(*args)


def parallel_map(function, iterable):
  """Calls a function for every element in an iterable using multiple cores."""
  if FORCE_DISABLE_MULTIPROCESSING:
    return [function(*args) for args in iterable]

  original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
  num_threads = mp.cpu_count() * 2
  pool = mp.Pool(processes=num_threads)
  signal.signal(signal.SIGINT, original_sigint_handler)

  p = pool.map_async(_function_wrapper, ((function, args) for args in iterable))
  try:
    results = p.get(0xFFFFFFFF)
  except KeyboardInterrupt:
    pool.terminate()
    raise
  pool.close()
  return results


def download_video_segment(segment_url, data_dir):
  log_results = []
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.makedirs(data_dir)
  local_segment_path = os.path.join(data_dir, segment_url.split('/')[-1])
  if not tf.io.gfile.exists(local_segment_path):
    # memfile, _ = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False)
    memfile = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False) # returns with memfile.seek(0)
    memfile.seek(0)
    with tf.io.gfile.GFile(name=local_segment_path, mode='w') as f:
      f.write(memfile.getvalue())
    # print('\tDownloaded {} to {}'.format(segment_url, local_segment_path))
    log_results.append('\tDownloaded {} to {}'.format(segment_url, local_segment_path))
  else:
    # print('\tFound target segment {} (from {})'.format(local_segment_path, segment_url))
    log_results.append('\tFound target segment {} (from {})'.format(local_segment_path, segment_url))
  print("\n".join(log_results))

def extract_frames(segment_urls, video_fname, frames_dir, videos_dir, df_decomposition):
  log_results = []

  target_stitched_vid_frames_dir = frames_dir
  if not tf.io.gfile.exists(target_stitched_vid_frames_dir):
    tf.io.gfile.makedirs(target_stitched_vid_frames_dir)

  local_vid_segment_paths = [os.path.join(videos_dir, segment_url.split('/')[-1]) for segment_url in segment_urls]

  vid_caps = [cv2.VideoCapture(local_vid_segment_path) for local_vid_segment_path in local_vid_segment_paths]
  for seg_vid_cap in vid_caps:
    seg_vid_cap.set(cv2.CAP_PROP_FPS, FPS)
  frame_counts = list(map(lambda vc: int(vc.get(cv2.CAP_PROP_FRAME_COUNT)), vid_caps))
  n_frames_expected = sum(frame_counts)

  failed_target_videos = []

  if n_frames_expected > 0:
    # get count of existing stitched frames in target_stitched_vid_frames_dir
    n_stitched_frames = len(tf.io.gfile.listdir(target_stitched_vid_frames_dir))

    b_restitch = n_stitched_frames < n_frames_expected
    n_stitched_frames = 0 if b_restitch else n_stitched_frames

    for i, seg_vid_cap in enumerate(vid_caps):
      _n_frames_expected = frame_counts[i]
      fblocks = range(0, n_frames_expected, 1)
      # nested_tqdm_pb__stitch.set_description(desc=s_decompose.format(i+1,n_segs))
      # nested_tqdm_pb__stitch.leave = True
      # nested_tqdm_pb__stitch.reset(total=_n_frames_expected)
      # nested_tqdm_pb__stitch.refresh(nolock=False)

      if b_restitch:
        success, frame = seg_vid_cap.read()
        n_frames = 0
        while success:
          cv2.imwrite(os.path.join(target_stitched_vid_frames_dir, f"{n_stitched_frames}.jpg"), frame)
          n_frames += 1
          n_stitched_frames += 1
          # nested_tqdm_pb__stitch.update(1)
          success, frame = seg_vid_cap.read()

        if n_frames != _n_frames_expected:
          # print(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since {_n_frames_expected} frames were expected from segment {local_vid_segment_paths[i]} but only {n_frames} were successfully extracted")
          log_results.append(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since {_n_frames_expected} frames were expected from segment {local_vid_segment_paths[i]} but only {n_frames} were successfully extracted")
          failed_target_videos.append(video_fname)
          fail = True
          break
        else:
          # print(f"\tAdded {n_stitched_frames} frames from segment {local_vid_segment_paths[i]} for target video {video_fname} (stitched-frames dir {target_stitched_vid_frames_dir})")
          log_results.append(f"\tAdded {n_stitched_frames} frames from segment {local_vid_segment_paths[i]} for target video {video_fname} (stitched-frames dir {target_stitched_vid_frames_dir})")

      else:
        n_frames = _n_frames_expected
        # nested_tqdm_pb__stitch.update(_n_frames_expected)
        # print('\tFound existing stiched-frames for {} ({} frames in {})'.format(target_stitched_vid_frames_dir, n_stitched_frames, target_stitched_vid_frames_dir))
        log_results.append('\tFound existing stiched-frames for {} ({} frames in {})'.format(target_stitched_vid_frames_dir, n_stitched_frames, target_stitched_vid_frames_dir))

      # df_decomposition.loc[len(df_decomposition)] = [local_vid_segment_paths[i], target_stitched_vid_frames_dir, n_frames]

  else:
    # print(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segments have zero frames")
    log_results.append(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segments have zero frames")
    failed_target_videos.append(video_fname)
    fail = True  

  print("\n".join(log_results))

  return df_decomposition




# Good for debugging.
FORCE_DISABLE_MULTIPROCESSING = False

TMP_DIR = '/tmp'
DATA_ROOT_DIR = None
VIDEO_INDEX_BASE = 'video_index-20120129'
VIDEO_INDEXES_ARCHIVE = VIDEO_INDEX_BASE+'.zip'
VIDEO_INDEXES_DIR = os.path.join(TMP_DIR, VIDEO_INDEX_BASE)
SELECTED_VIDEO_INDEX_PATH = os.path.join(VIDEO_INDEXES_DIR, 'files_by_video_name.csv')
VIDEO_DIR = None
STICHED_VIDEO_FRAMES_DIR = None
_1KB = 1024
_1MB = _1KB**2
FPS = 30

# BEAM_BOOTSTRAP_VIDEO_INDEX = False
df_decomposition = pd.DataFrame(columns=['src_video', 'dest_dir', 'n_frames'])

# note that this "schema" assumes intimate knowledge of 'files_by_video_name.csv' layout (i.e. the column-name/order mappings in it)
SCHEMA_COL_NAMES = [
  'filename', 
  'video_seq_id', 
  'perspective_cam_id', 
  'compressed_mov_url', 
  'uncompressed_avi_url', 
  'uncompressed_avi_mirror_1_url', 
  'uncompressed_avi_mirror_2_url'
]

def run(max_data_files, data_dir, beam_bootstrap_video_index=False):
  """Extracts the specified number of data files in parallel."""
  # mpmanager = mp.Manager()

  print(f"beam_bootstrap_video_index: {beam_bootstrap_video_index}")

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

  global DATA_ROOT_DIR
  DATA_ROOT_DIR = data_dir
  if not tf.io.gfile.exists(DATA_ROOT_DIR):
    tf.io.gfile.makedirs(DATA_ROOT_DIR)
  if not tf.io.gfile.exists(TMP_DIR):
    tf.io.gfile.makedirs(TMP_DIR)

  global VIDEO_DIR
  VIDEO_DIR = os.path.join(TMP_DIR, 'videos')
  if not tf.io.gfile.exists(VIDEO_DIR):
    tf.io.gfile.makedirs(VIDEO_DIR)

  global STICHED_VIDEO_FRAMES_DIR
  STICHED_VIDEO_FRAMES_DIR = os.path.join(DATA_ROOT_DIR, 'stitched_video_frames')
  if not tf.io.gfile.exists(STICHED_VIDEO_FRAMES_DIR):
    tf.io.gfile.makedirs(STICHED_VIDEO_FRAMES_DIR)

  def boostrap_video_index(d_vid_indexes_info):
    """
    d_vid_indexes_info MUST be a dict as follows:
      {
        'vid_indexes_dir': VIDEO_INDEXES_DIR, 
        'sel_vid_index_path': SELECTED_VIDEO_INDEX_PATH, 
        'video_indexes_archive': VIDEO_INDEXES_ARCHIVE, 
        'tmp_dir': TMP_DIR
      }

    this function downloads d_vid_indexes_info['video_indexes_archive'] from http://www.bu.edu/asllrp/ncslgr-for-download
      and extracts it to os.path.join(d_vid_indexes_info['tmp_dir'], d_vid_indexes_info['video_indexes_archive'])
      (assuming that has not already been done - i.e. if not os.path.isdir(d_vid_indexes_info['vid_indexes_dir']) or not os.path.isfile(d_vid_indexes_info['sel_vid_index_path']))

    this function returns d_vid_indexes_info['sel_vid_index_path'] only after the above has been done
    """

    print(f"video index boostrap info: {d_vid_indexes_info}")
    if not os.path.isdir(d_vid_indexes_info['vid_indexes_dir']) or not os.path.isfile(d_vid_indexes_info['sel_vid_index_path']):
      remote_archive_path = os.path.join('http://www.bu.edu/asllrp/ncslgr-for-download', d_vid_indexes_info['video_indexes_archive'])
      local_archive_path = os.path.join(d_vid_indexes_info['tmp_dir'], d_vid_indexes_info['video_indexes_archive'])
      utils.download(
          remote_archive_path, 
          local_archive_path, 
          block_sz=_1MB
      )
      zip_ref = zipfile.ZipFile(local_archive_path, 'r')
      print(f"unzipping {local_archive_path} to {d_vid_indexes_info['vid_indexes_dir']}...")
      zip_ref.extractall(d_vid_indexes_info['tmp_dir'])
      zip_ref.close()
      print(f"\tDONE")
      print(f"deleting {local_archive_path}...")
      os.remove(local_archive_path)
      print(f"\tDONE")
    else:
      print(f"Found video index {d_vid_indexes_info['sel_vid_index_path']}")
    return d_vid_indexes_info['sel_vid_index_path']



  if beam_bootstrap_video_index:
    # ************* Test Apache Beam: BEGIN *************
    def vid_index_csv_rows(sel_vid_index_csv_path, rows_to_dicts=False, dict_field_names=None):
      """
      this function opens the sel_vid_index_csv_path file (as a CSV),
        reads its contents and returns a list of its rows
      
      by default, each row is a list of elements (separated initially by comma (',') of course)

      if rows_to_dicts is True, each row is converted to a dict keyed by field names
        if dict_field_names is None
          csv.DictReader uses the first row in the csv file as field names
        otherwise
          dict_field_names provides field names (keys of each dict)
      """
      f = beam.io.filesystems.FileSystems.open(sel_vid_index_csv_path)
      if sys.version_info >= (3,0):
        f = io.TextIOWrapper(f)
      if rows_to_dicts:
        csv_reader = csv.DictReader(f,fieldnames=dict_field_names) if dict_field_names is not None else csv.DictReader(f)
      else:
        csv_reader = csv.reader(f)
      if dict_field_names is not None:
          next(csv_reader) # skip past first row (contains column names that we do not want to use)
      return csv_reader
    
    class VideoIndexEntry(typing.NamedTuple):
      """
      fields should be identical to SCHEMA_COL_NAMES
      """
      filename: str                       # 'Video file name in XML file'
      video_seq_id: int                   # 'Video sequence id'
      perspective_cam_id: int             # 'Perspective/Camera id'
      compressed_mov_url: str             # 'Compressed MOV file'
      uncompressed_avi_url: str           # 'Uncompressed AVI'
      uncompressed_avi_mirror_1_url: str  # 'Uncompressed AVI mirror 1'
      uncompressed_avi_mirror_2_url: str  # 'Uncompressed AVI mirror 2'

    # now register this schema with beam as a RowCoder
    beam.coders.registry.register_coder(VideoIndexEntry, beam.coders.RowCoder)

    def vid_index_csv_rows_to_dicts(sel_vid_index_csv_path): # 
      """
      this function simply wraps the call to vid_index_csv_rows() but shares the same goal of the VideoIndexEntry class: to produce a "schema'd" pcoll
      so we fix the definition of dict_field_names to:
        dict_field_names=['filename', 'video_seq_id', 'perspective_cam_id', 'compressed_mov_url', 'uncompressed_avi_url', 'uncompressed_avi_mirror_1_url', 'uncompressed_avi_mirror_2_url']
      """
      return vid_index_csv_rows(sel_vid_index_csv_path, rows_to_dicts=True, dict_field_names=SCHEMA_COL_NAMES)


    class PipelinePcollPrinter(beam.DoFn):
      """
      prints each element of the pcoll
      should only be used for debugging - i.e. NOT in production
      """
      def process(self, pcoll_element):
          print(pcoll_element)
          # print(dir(pcoll_element))
          # print()
          return [pcoll_element] # passthrough

    class VideoIndexPandasDataframeFromSchemadPcoll(beam.DoFn):
      """
      creates an underlying pandas DataFrame
      appends pcoll dict element to this dataframe
      """
      def __init__(self):
        self.df_video_index = pd.DataFrame(columns=SCHEMA_COL_NAMES)
        # debug
        self.rows = 0

      def process(self, pcoll_dict_element):
        self.df_video_index = self.df_video_index.append([pcoll_dict_element])
        return [pcoll_dict_element] # passthrough


    class VideoSegmentDownloadInfoGatherer(beam.DoFn):
      """
      assumes pcoll is already schemad
      """
      def process(self, schemad_pcoll_element):
        video_fname = schemad_pcoll_element.filename
        frames_dir = os.path.join(STICHED_VIDEO_FRAMES_DIR, video_fname.split('.')[0])
        urls = schemad_pcoll_element.compressed_mov_url.split(';') # this can be a list, separated by ';'
        return [{'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': url} for url in urls]


    def download_video_segment_2(d_vid_seg_download_info):
      """
      expects d_vid_seg_download_info: {'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': url}
      """
      segment_url = d_vid_seg_download_info['segment_url']
      log_results = []
      if not tf.io.gfile.exists(VIDEO_DIR):
        tf.io.gfile.makedirs(VIDEO_DIR)
      local_segment_path = os.path.join(VIDEO_DIR, segment_url.split('/')[-1])
      if not tf.io.gfile.exists(local_segment_path):
        memfile = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False) # returns with memfile.seek(0)
        memfile.seek(0)
        with tf.io.gfile.GFile(name=local_segment_path, mode='w') as f:
          f.write(memfile.getvalue())
        # print('\tDownloaded {} to {}'.format(segment_url, local_segment_path))
        log_results.append('Downloaded {} to {}'.format(segment_url, local_segment_path))
      else:
        # print('\tFound target segment {} (from {})'.format(local_segment_path, segment_url))
        log_results.append('Found target segment {} (from {})'.format(local_segment_path, segment_url))
      print("\n".join(log_results))
      return [d_vid_seg_download_info] # passthrough

    class VideoSegmentExtractor(beam.DoFn):
      def process(self, d_vid_seg_download_info):
        return download_video_segment_2(d_vid_seg_download_info)




    vid_index_df_converter = VideoIndexPandasDataframeFromSchemadPcoll()
    
    options = {
      'project': 'my-project', # change
      'runner:': 'DirectRunner',
      'direct_num_workers': 0, # 0 is use all available cores
      'direct_running_mode': 'multi_threading', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
      'streaming': False
    }
    pipeline_options = PipelineOptions(flags=[], **options)
    with beam.Pipeline(options=pipeline_options) as pl:
      vid_index_schemad_pcoll = (
        pl

        | beam.Create(  # pcoll containing values required to bootstrap from vid index
            [ # one row containing dict of:
                # 1. url of video indexes archive
                # 2. local destination (path) for the downloaded archive
                # 3. local destination (path) which will receive the extracted archive csv files (there are more than one)
                # 4. final path to the selected videx index csv
                #   (note that the dict is not laid out in the above order)
              {
                'vid_indexes_dir': VIDEO_INDEXES_DIR, 
                'sel_vid_index_path': SELECTED_VIDEO_INDEX_PATH, 
                'video_indexes_archive': VIDEO_INDEXES_ARCHIVE, 
                'tmp_dir': TMP_DIR
              }
            ]
          )
        | "Beam PL: bootstrap video index" >> beam.Map(boostrap_video_index) # boostrap_video_index outputs SELECTED_VIDEO_INDEX_PATH but beam.Map() wraps this in a pcoll and is fed to...

        # | "Beam PL: read csv rows" >> beam.FlatMap(vid_index_csv_rows) # but rows of this PColl are lists and the first one is the header row (column names), which we do not want...
        | "Beam PL: read video index into pcoll" >> beam.FlatMap(vid_index_csv_rows_to_dicts) # outputs another pcoll but with each row as dict

        # note that we want rows as dicts since dicts help us apply a schema to the pcoll, which is what we want in the end

        # NOTE! This is scheduled for deletion since it violates thread-safety requirement - DO NOT UNCOMMENT; it is only here for reference for the time being
        # | "Beam PL: create pandas df video index from schemad PCollection" >> beam.ParDo(vid_index_df_converter)

        # now we want to apply the schema so that we can ultimately use beam's SqlTransform (very similar to pandas sqldf)
          # Haven't yet worked the following out
          # | "Beam PL: map csv rows to schema" >> beam.Map(
          #     lambda x: VideoIndexEntry(
          #       str(urllib.parse.quote(x[SCHEMA_COL_NAMES[0]])),  # originally 'Video file name in XML file': str
          #       int(x[SCHEMA_COL_NAMES[1]]),                      # originally 'Video sequence id': int
          #       int(x[SCHEMA_COL_NAMES[2]]),                      # originally 'Perspective/Camera id': int
          #       str(x[SCHEMA_COL_NAMES[3]]),                      # originally 'Compressed MOV file': str (note that this is actually a list with ';' as delimiter)
          #       str(x[SCHEMA_COL_NAMES[4]]),                      # originally 'Uncompressed AVI': str
          #       str(x[SCHEMA_COL_NAMES[5]]),                      # originally 'Uncompressed AVI mirror 1': str
          #       str(x[SCHEMA_COL_NAMES[6]])                       # originally 'Uncompressed AVI mirror 2': str
          #     )
          #   ).with_output_types(VideoIndexEntry)

        # So for now, we settle for beam.Row implementation, which is almost as good (although doesn't respect field order)...
        | "Beam PL: apply schema to video index pcoll" >> beam.Map(lambda x: beam.Row(
              filename=str(urllib.parse.quote(x[SCHEMA_COL_NAMES[0]])),  # We MUST URL encode filenames since some of them sloppily contain spaces!
              video_seq_id=int(x[SCHEMA_COL_NAMES[1]]),                            
              perspective_cam_id=int(x[SCHEMA_COL_NAMES[2]]),                  
              compressed_mov_url=str(x[SCHEMA_COL_NAMES[3]]),            # this is actually a list with ';' as delimiter)
              uncompressed_avi_url=str(x[SCHEMA_COL_NAMES[4]]),                     
              uncompressed_avi_mirror_1_url=str(x[SCHEMA_COL_NAMES[5]]),   
              uncompressed_avi_mirror_2_url=str(x[SCHEMA_COL_NAMES[6]])
            )
          )
        # | "Beam PL: print schemad video index pcoll" >> beam.ParDo(PipelinePcollPrinter())  # comment out for production

        # filter schemad pcoll as desired (if necessary) using SqlTransform(), for example
        # | SqlTransform("SELECT * FROM PCOLLECTION")
      )

      (
        vid_index_schemad_pcoll
        | "Beam PL: gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: print download info for video segments" >> beam.ParDo(PipelinePcollPrinter())  # comment out for production
        | "Beam PL: download video segments" >> beam.ParDo(VideoSegmentExtractor()) # if not cpu_par_vid_seg_dl else 
        # | beam.FlatMap(download_video_segment) # now flatten so we can parallelize segment downloads
      )

    print(f"Beam PL: ALL DONE!")
    # ************* Test Apache Beam: END *************
    df_video_index = vid_index_df_converter.df_video_index

  else:
    # first make sure SELECTED_VIDEO_INDEX exists locally
    boostrap_video_index(d_vid_indexes_info={
      'vid_indexes_dir': VIDEO_INDEXES_DIR, 
      'sel_vid_index_path': SELECTED_VIDEO_INDEX_PATH, 
      'video_indexes_archive': VIDEO_INDEXES_ARCHIVE, 
      'tmp_dir': TMP_DIR
    })

    # df_video_index drives the (parallel) download of video segments
    df_video_index = pd.read_csv(SELECTED_VIDEO_INDEX_PATH)
    df_video_index.rename(
        columns={
            'Video file name in XML file': SCHEMA_COL_NAMES[0],
            'Video sequence id': SCHEMA_COL_NAMES[1],
            'Perspective/Camera id': SCHEMA_COL_NAMES[2],
            'Compressed MOV file': SCHEMA_COL_NAMES[3],
            'Uncompressed AVI': SCHEMA_COL_NAMES[4],
            'Uncompressed AVI mirror 1': SCHEMA_COL_NAMES[5],
            'Uncompressed AVI mirror 2': SCHEMA_COL_NAMES[6]
        }, 
        inplace=True
    )
    # NOTE!
    #   This is a CRUCIAL step! We MUST URL encode filenames since some of them sloppily contain spaces!
    df_video_index['filename'] = df_video_index['filename'].map(lambda filename: urllib.parse.quote(filename))
    df_video_index_csv_path = os.path.join(DATA_ROOT_DIR, 'df_video_index.csv')
    df_video_index.to_csv(path_or_buf=df_video_index_csv_path)
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(df_video_index_csv_path) else 'FAILED to save'} {df_video_index_csv_path}")

    target_videos = []
    for idx, media_record in df_video_index.iterrows():
      video_fname = media_record['filename']
      frames_dir = os.path.join(STICHED_VIDEO_FRAMES_DIR, video_fname.split('.')[0])
      urls = media_record['compressed_mov_url'].split(';') # this can be a list, separated by ';'
      # local_paths = [os.path.join(VIDEO_DIR, url.split('/')[-1]) for url in urls]
      d = {
        'video_fname': video_fname,
        'frames_dir': frames_dir,
        'segment_urls': urls
      }
      target_videos.append(d)
    
    if not max_data_files:
      max_data_files = len(target_videos)
    assert max_data_files >= 1
    print('Found {} target video records, using {}'.format(len(target_videos), max_data_files))
    target_videos = target_videos[:max_data_files]

    # download data (video segment) files in parallel (on the CPU of the machine - either local or VM in GCP DataFlow)
    #   note that in order to accomplish parallelism, since this uses the file system of the machine, this must be done
    #   using the CPU of the machine
    #   please see the definition of the parallel_map() function for details
    print('Downloading segments for target videos...')
    parallel_map(
      download_video_segment,
      ((seg_url, VIDEO_DIR) for tvd in target_videos for seg_url in tvd['segment_urls'])
    )
      
    # extract frames from video segments in parallel (on the CPU of the machine - either local or VM in GCP DataFlow)
    #   note that in order to accomplish parallelism, since this uses the file system of the machine, this must be done
    #   using the CPU of the machine
    #   please see the definition of the parallel_map() function for details
    print('\nExtracting and aggregating frames from video-segments into target video-frames directories ...')
    parallel_map(
      extract_frames,
      (
        (
          tvd['segment_urls'],      # segment_urls
          tvd['video_fname'],       # video_fname
          tvd['frames_dir'],
          VIDEO_DIR,
          df_decomposition
        ) for tvd in target_videos
      ) 
    )
    df_decomposition.to_csv(path_or_buf=os.path.join(DATA_ROOT_DIR, 'df_decomposition.csv'))
    df_decomposition_csv_path = os.path.join(DATA_ROOT_DIR, 'df_decomposition.csv')
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(df_decomposition_csv_path) else 'FAILED to save'} {df_decomposition_csv_path}")

  return df_video_index, df_decomposition




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
  #   script --beam-bootstrap-video-index
  #   script --beam-bootstrap-video-index <bool>
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
    "--beam-bootstrap-video-index", 
    type=str2bool, 
    nargs='?',
    const=True, 
    default=False,
    help=""
  )

  args = parser.parse_args()
  run(
    args.max_data_files if args.max_data_files!=-1 else None, 
    os.path.join(args.work_dir, 'data'), 
    beam_bootstrap_video_index=args.beam_bootstrap_video_index
  )