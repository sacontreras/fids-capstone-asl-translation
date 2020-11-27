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

# This tool downloads SDF files from an FTP source.

from __future__ import absolute_import

import argparse
import ftplib
import multiprocessing as mp
import os
import re
import signal
import tempfile
import tensorflow as tf
import zlib
import zipfile
from io import BytesIO
import pandas as pd
import urllib
import utils

# # Regular expressions to parse an FTP URI.
# _USER_RE = r'''(?P<user>[^:@]+|'[^']+'|"[^"]+")'''
# _PASSWORD_RE = r'''(?P<password>[^@]+|'[^']+'|"[^"]+")'''
# _CREDS_RE = r'{}(?::{})?'.format(_USER_RE, _PASSWORD_RE)
# FTP_RE = re.compile(r'^ftp://(?:{}@)?(?P<abs_path>.*)$'.format(_CREDS_RE))

# Good for debugging.
FORCE_DISABLE_MULTIPROCESSING = False

ROOT_DIR = '/content'
DATA_DIR_NAME = 'fids-capstone-data' 
DATA_ROOT_DIR = os.path.join(ROOT_DIR, DATA_DIR_NAME)
CORPUS_DIR_NAME = 'ncslgr-xml'
CORPUS_DIR = os.path.join(DATA_ROOT_DIR, CORPUS_DIR_NAME)
VIDEO_INDEX_BASE = 'video_index-20120129'
VIDEO_INDEX_ARCHIVE = VIDEO_INDEX_BASE+'.zip'
VIDEO_INDEX_PATH = os.path.join(DATA_ROOT_DIR, VIDEO_INDEX_ARCHIVE)
VIDEO_INDEX_DIR = os.path.join(DATA_ROOT_DIR, VIDEO_INDEX_BASE)
VIDEO_DIR = os.path.join(DATA_ROOT_DIR, 'videos')
VIDEO_FRAMES_DIR = os.path.join(DATA_ROOT_DIR, 'frames')
STICHED_VIDEO_FRAMES_DIR = os.path.join(DATA_ROOT_DIR, 'stitched_video_frames')
TMP_DIR = os.path.join(DATA_ROOT_DIR, 'tmp')
_1KB = 1024
_1MB = _1KB**2


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


# def extract_data_file(ftp_file, data_dir):
#   """Function to extract a single PubChem data file."""
#   user = ftp_file['user']
#   password = ftp_file['password']
#   server = ftp_file['server']
#   path = ftp_file['path']
#   basename = os.path.basename(path)
#   sdf_file = os.path.join(data_dir, os.path.splitext(basename)[0])

#   if not tf.io.gfile.exists(sdf_file):
#     # The `ftp` object cannot be pickled for multithreading, so we open a
#     # new connection here
#     memfile = BytesIO()
#     ftp = ftplib.FTP(server, user, password)
#     ftp.retrbinary('RETR ' + path, memfile.write)
#     ftp.quit()

#     memfile.seek(0)
#     with tf.io.gfile.GFile(name=sdf_file, mode='w') as f:
#       gzip_wbits_format = zlib.MAX_WBITS | 16
#       contents = zlib.decompress(memfile.getvalue(), gzip_wbits_format)
#       f.write(contents)
#     print('Extracted {}'.format(sdf_file))

#   else:
#     print('Found {}'.format(sdf_file))

def download_video_segment(segment_url, data_dir):
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.makedirs(data_dir)
  local_segment_path = os.path.join(data_dir, segment_url.split('/')[-1])
  if not tf.io.gfile.exists(local_segment_path):
    # memfile, _ = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False)
    memfile = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False)
    # memfile.seek(0)
    with tf.io.gfile.GFile(name=local_segment_path, mode='w') as f:
      # gzip_wbits_format = zlib.MAX_WBITS | 16
      # contents = zlib.decompress(memfile.getvalue(), gzip_wbits_format)
      f.write(memfile.getvalue())
    print('Downloaded {}'.format(local_segment_path))
  else:
    print('Found {}'.format(local_segment_path))


# def download_process_media(df_video_index=df_video_index, dest_dir=STICHED_VIDEO_FRAMES_DIR, fps=30):
#   n_missing = 0
#   n_downloaded = 0
#   failed_downloads = []

#   try:
#     os.mkdir(VIDEO_DIR)
#   except:
#     pass

#   try:
#     os.mkdir(dest_dir)
#   except:
#     pass

#   print(f"Downloading/processing media-segments for {len(df_video_index)} videos...")
#   tqdm_pb = tqdm(range(0, len(df_video_index), 1))
#   tqdm_pb.set_description(desc='Video')
#   media_record_iterator = df_video_index.iterrows()
#   nested_tqdm_pb__byte = trange(1, leave=True)
#   s_download = "DOWNLOAD ({} of {})"
#   nested_tqdm_pb__byte.set_description(desc=s_download)
#   nested_tqdm_pb__stitch = trange(1, leave=True)
#   s_decompose = "DECOMPOSE ({} of {})"
#   nested_tqdm_pb__stitch.set_description(desc=s_decompose)

#   df_decomposition = pd.DataFrame(columns=['src_video', 'dest_dir', 'n_frames'])

#   failed_target_videos = []

#   for tfblock in tqdm_pb:
#     idx, media_record = next(media_record_iterator)

#     remote_vid_paths = media_record['compressed_mov_url'].split(';') # this can be a list, separated by ';'
#     local_vid_segment_paths = []
#     n_segs = len(remote_vid_paths)

#     # ************************************ DOWNLOAD corresponding video segments: BEGIN ************************************
#     for i, remote_vid_path in enumerate(remote_vid_paths):
#       nested_tqdm_pb__byte.set_description(desc=s_download.format(i+1,n_segs))
#       nested_tqdm_pb__byte.refresh()
#       fname = remote_vid_path.split('/')[-1]
#       local_vid_path = os.path.join(VIDEO_DIR, fname)
#       local_vid_segment_paths.append(local_vid_path)
#       if not os.path.isfile(local_vid_path):
#         n_missing += 1
#         try:
#           nested_tqdm_pb__byte = utils.download(
#               remote_vid_path, 
#               local_vid_path, 
#               block_sz=_1MB,
#               display=False, 
#               nested_tqdm_pb=nested_tqdm_pb__byte
#           )
#           n_downloaded += 1
#         except Exception as e:
#           print(f"Failed downloading {remote_vid_path} to {local_vid_path}: {e}")
#           # to do: remove associated rows??  (NOT YET!)
#           failed_downloads.append(remote_vid_path)
#       else:
#         nested_tqdm_pb__byte.leave = True
#         nested_tqdm_pb__byte.reset(total=1)
#         nested_tqdm_pb__byte.refresh(nolock=False)
#         nested_tqdm_pb__byte.update(1)
#     # ************************************ DOWNLOAD corresponding video segments: END ************************************


#     # ************************************ DECOMPOSE segments into frames and then COMBINE frames into destination/target video: BEGIN ************************************
#     target_video_fname = media_record['filename']
#     target_stitched_vid_frames_dir = os.path.join(STICHED_VIDEO_FRAMES_DIR, target_video_fname.split('.')[0])
#     # TO DO: check if this even needs to be done... if it has already been done, don't do it again!
#     # if os.path.isdir(target_stitched_vid_frames_dir) or len(os.listdir(target_stitched_vid_frames_dir))>0:
#     try:
#       os.mkdir(target_stitched_vid_frames_dir)
#     except:
#       pass

#     fail = False
#     n_stitched_frames = 0

#     try:
#       vid_caps = [cv2.VideoCapture(local_vid_segment_path) for local_vid_segment_path in local_vid_segment_paths]
#       for seg_vid_cap in vid_caps:
#         seg_vid_cap.set(cv2.CAP_PROP_FPS, fps)
#       frame_counts = list(map(lambda vc: int(vc.get(cv2.CAP_PROP_FRAME_COUNT)), vid_caps))
#       n_frames_expected = sum(frame_counts)
#     except:
#       vid_caps = None
#       frame_counts = None
#       n_frames_expected = 0

#     if n_frames_expected > 0:
#       n_stitched_frames = len(os.listdir(target_stitched_vid_frames_dir))
#       b_restitch = n_stitched_frames < n_frames_expected
#       # if not b_restitch:
#       #   print(f"\tDecomposition directory {target_stitched_vid_frames_dir} (for {target_video_fname}) contains {n_stitched_frames} image files (frames)")  
#       n_stitched_frames = 0 if b_restitch else n_stitched_frames

#       for i, seg_vid_cap in enumerate(vid_caps):
#         _n_frames_expected = frame_counts[i]
#         fblocks = range(0, n_frames_expected, 1)
#         nested_tqdm_pb__stitch.set_description(desc=s_decompose.format(i+1,n_segs))
#         nested_tqdm_pb__stitch.leave = True
#         nested_tqdm_pb__stitch.reset(total=_n_frames_expected)
#         nested_tqdm_pb__stitch.refresh(nolock=False)

#         if b_restitch:
#           success, frame = seg_vid_cap.read()
#           n_frames = 0
#           while success:
#             cv2.imwrite(os.path.join(target_stitched_vid_frames_dir, f"{n_stitched_frames}.jpg"), frame)
#             n_frames += 1
#             n_stitched_frames += 1
#             nested_tqdm_pb__stitch.update(1)
#             success, frame = seg_vid_cap.read()

#           if n_frames != _n_frames_expected:
#             print(f"\t***WARNING!!!*** Cannot stitch together target video {target_video_fname} since {_n_frames_expected} frames were expected from segment {local_vid_segment_paths[i]} but only {n_frames} were successfully extracted")
#             failed_target_videos.append(target_video_fname)
#             fail = True
#             break

#         else:
#           n_frames = _n_frames_expected
#           nested_tqdm_pb__stitch.update(_n_frames_expected)

#         df_decomposition.loc[len(df_decomposition)] = [local_vid_segment_paths[i], target_stitched_vid_frames_dir, n_frames]

#     else:
#       # print(f"\t***WARNING!!!*** Cannot stitch together target video {target_video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segment {local_vid_segment_path} has zero frames")
#       failed_target_videos.append(target_video_fname)
#       fail = True
#       break    

#     if fail:
#       try:
#         shutil.rmtree(target_stitched_vid_frames_dir)
#       except:
#         pass
#     # ************************************ DECOMPOSE segments into frames and then COMBINE frames into destination/target video: END ************************************
     
#   print("\tDONE: " + f"Successfully downloaded {n_downloaded}{' (but failed to download '+str(len(failed_downloads))+')' if len(failed_downloads)>0 else ''} video-segment files (out of {n_missing} missing media files)" if n_missing>0 else "\tDONE: there were no missing media files")

#   return df_decomposition


def run(max_data_files, data_dir):
  """Extracts the specified number of data files in parallel."""
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.makedirs(data_dir)

  # Get available data files
  video_index_path = os.path.join(VIDEO_INDEX_DIR, 'files_by_video_name.csv')
  if not os.path.isdir(VIDEO_INDEX_DIR) or not os.path.isfile(video_index_path):
    remote_archive_path = os.path.join('http://www.bu.edu/asllrp/ncslgr-for-download', VIDEO_INDEX_ARCHIVE)
    local_archive_path = os.path.join('./', VIDEO_INDEX_ARCHIVE)
    utils.download(
        remote_archive_path, 
        local_archive_path, 
        block_sz=_1MB
    )
    zip_ref = zipfile.ZipFile(local_archive_path, 'r')
    print(f"unzipping {local_archive_path} to {VIDEO_INDEX_DIR}...")
    zip_ref.extractall(DATA_ROOT_DIR) # this will video_index-20120129.zip to content dir first
    zip_ref.close()
    print(f"\tDONE")
    print(f"deleting {local_archive_path}...")
    os.remove(local_archive_path)
    print(f"\tDONE")
  else:
    print(f'Found video index {video_index_path}')

  df_video_index = pd.read_csv(video_index_path)
  df_video_index.rename(
      columns={
          'Video file name in XML file':'filename',
          'Video sequence id': 'video_seq_id',
          'Perspective/Camera id': 'perspective_cam_id',
          'Compressed MOV file': 'compressed_mov_url',
          'Uncompressed AVI': 'uncompressed_avi_url',
          'Uncompressed AVI mirror 1': 'uncompressed_avi_mirror_1_url',
          'Uncompressed AVI mirror 2': 'uncompressed_avi_mirror_2_url'
      }, 
      inplace=True
  )
  # NOTE!
  #   This is a CRUCIAL step! We MUST URL encode filenames since some of them sloppily contain spaces!
  df_video_index['filename'] = df_video_index['filename'].map(lambda filename: urllib.parse.quote(filename))

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

  # download segments in parallel
  target_videos = target_videos[:max_data_files]
  print('Downloading segments for target videos...')
  parallel_map(
    download_video_segment,
    ((seg_url, VIDEO_DIR) for tvd in target_videos for seg_url in tvd['segment_urls'])
  )
    
  # extract frames from video segments in parallel
  # ftp_files = ftp_files.loc[:max_data_files]
  # print('Extracting data files...')
  # parallel_map(
  #   extract_data_file, 
  #   ((ftp_file, data_dir) for ftp_file in ftp_files)
  # )


if __name__ == '__main__':
  """Main function"""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--work-dir',
      required=True,
      help='Directory for staging and working files. '
           'This can be a Google Cloud Storage path.')

  # parser.add_argument(
  #     '--data-sources',
  #     nargs='+',
  #     default=[CORPUS_DIR],
  #     help='Data source location where SDF file(s) are stored. '
  #          'Paths can be local, ftp://<path>, or gcs://<path>. '
  #          'Examples: '
  #          'ftp://hostname/path '
  #          'ftp://username:password@hostname/path')

  # parser.add_argument(
  #     '--filter-regex',
  #     default=r'\.sdf',
  #     help='Regular expression to filter which files to use. '
  #          'The regular expression will be searched on the full absolute path. '
  #          'Every match will be kept.')

  parser.add_argument(
      '--max-data-files',
      type=int,
      required=True,
      help='Maximum number of data files for every file pattern expansion. '
           'Set to -1 to use all files.')

  args = parser.parse_args()

  max_data_files = args.max_data_files
  if args.max_data_files == -1:
    max_data_files = None

  data_dir = os.path.join(args.work_dir, 'data')
  run(max_data_files, data_dir)