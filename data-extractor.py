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
from importlib import import_module

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
# import apache_beam.runners.interactive.interactive_beam as ib

# import signstreamxmlparser.analysis as sxa
# import signstreamxmlparser.analysis.signstream as ss
sxa = import_module('.analysis', 'signstreamxmlparser-refactored')
ss = import_module('.signstream', 'signstreamxmlparser-refactored.analysis')

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


# **************************************** global variables: BEGIN ****************************************
_1KB = 1024
_1MB = _1KB**2
FPS = 30


# Good for debugging beam pipelines
FORCE_DISABLE_MULTIPROCESSING = False


TMP_DIR = '/tmp'
VIDEO_INDEX_BASE = 'video_index-20120129'
VIDEO_INDEXES_ARCHIVE = VIDEO_INDEX_BASE+'.zip'
VIDEO_INDEXES_DIR = os.path.join(TMP_DIR, VIDEO_INDEX_BASE)
SELECTED_VIDEO_INDEX_PATH = os.path.join(VIDEO_INDEXES_DIR, 'files_by_video_name.csv')
CORPUS_ARCHIVE = 'ncslgr-xml.zip'
CORPUS_DS_FNAME = 'corpus.csv'
DOCUMENT_ASL_CONSULTANT_DS_FNAME = 'document-asl-consultant.csv'
ASL_CONSULTANT_DS_FNAME = 'asl-consultant.csv'
VIDEO_DS_FNAME = 'video.csv'
UTTERANCE_DS_FNAME = 'utterance.csv'
UTTERANCE_VIDEO_DS_FNAME = 'utterance-video.csv'
UTTERANCE_TOKEN_DS_FNAME = 'utterance-token.csv'
VOCABULARY_DS_FNAME = 'vocabulary.csv'
CORPUS_BASE = 'ncslgr-xml'
CORPUS_ARCHIVE = CORPUS_BASE+'.zip'
CORPUS_DIR = os.path.join(TMP_DIR, CORPUS_BASE)
CORPUS_DS_FNAME = 'ncslgr-corpus.csv'
DOCUMENT_ASL_CONSULTANT_DS_FNAME = 'document-asl-consultant.csv'
ASL_CONSULTANT_DS_FNAME = 'asl-consultant.csv'
VIDEO_DS_FNAME = 'video.csv'
VIDEO_SEGMENT_DS_FNAME = 'video-segment.csv'
UTTERANCE_DS_FNAME = 'utterance.csv'
UTTERANCE_VIDEO_DS_FNAME = 'utterance-video.csv'
UTTERANCE_TOKEN_DS_FNAME = 'utterance-token.csv'
UTTERANCE_TOKEN_FRAME_DS_FNAME = 'utterance-token-frame.csv'
VOCABULARY_DS_FNAME = 'vocabulary.csv'


# ********** SCHEMA-related (FIXED) globals: BEGIN **********
SCHEMA_COL_NAMES__CORPUS_DS = [
  'DocumentID',
  'Filename',
  'XML'
]
SCHEMA_PK__CORPUS_DS = [SCHEMA_COL_NAMES__CORPUS_DS[0]]

SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS = [
  'DocumentID',
  'ASLConsultantID'
]
SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS = [
  SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0], 
  SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[1]
]

SCHEMA_COL_NAMES__ASL_CONSULTANT_DS = [
  'ASLConsultantID',
  'Name',
  'Age',
  'Gender'
]
SCHEMA_PK__ASL_CONSULTANT_DS = [SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[0]]

SCHEMA_COL_NAMES__VIDEO_DS = [
  'DocumentID',
  'ASLConsultantID',
  'CameraPerspective',
  'Filename'
]
SCHEMA_PK__VIDEO_DS = [
  SCHEMA_COL_NAMES__VIDEO_DS[0],
  SCHEMA_COL_NAMES__VIDEO_DS[1],
  SCHEMA_COL_NAMES__VIDEO_DS[2]
]

SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS = [
  'DocumentID',
  'ASLConsultantID',
  'CameraPerspective',
  'SegmentSequence',
  'Filename',
  'URL'
]
SCHEMA_PK__VIDEO_SEGMENT_DS = [
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[0],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[1],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[2],
  SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[3]
]

SCHEMA_COL_NAMES__UTTERANCE_DS = [
  'DocumentID',
  'ASLConsultantID',
  'UtteranceSequence',
  'StartTime',
  'EndTime',
  'Tokens',
  'Translation'
]
SCHEMA_PK__UTTERANCE_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_DS[2]
]

SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS = [
  'DocumentID',
  'ASLConsultantID',
  'UtteranceSequence',
  'CameraPerspective'
]
SCHEMA_PK__UTTERANCE_VIDEO_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[3]
]

SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS = [
  'DocumentID',
  'ASLConsultantID',
  'UtteranceSequence',
  'TokenSequence',
  'StartTime',
  'EndTime',
  'TokenID',
  'Field',
  'FieldValue'
]
SCHEMA_PK__UTTERANCE_TOKEN_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[3]
]

SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS = [
  'UtteranceID',
  'TokenID',
  'TokenSequence',
  'FrameSequence',
  'ImageTensor'
]
SCHEMA_PK__UTTERANCE_TOKEN_FRAME_DS = [
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[0],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[1],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[2],
  SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[3]
]

SCHEMA_COL_NAMES__VOCABULARY_DS = [
  'TokenID',
  'Token'
]
SCHEMA_PK__VOCABULARY_DS = [SCHEMA_COL_NAMES__VOCABULARY_DS[0]]

# note that this "schema" assumes intimate knowledge of 'files_by_video_name.csv' layout (i.e. the column-name/order mappings in it)
global SCHEMA_COL_NAMES__VIDEO_INDEX
SCHEMA_COL_NAMES__VIDEO_INDEX = [
  'filename', 
  'video_seq_id', 
  'perspective_cam_id', 
  'compressed_mov_url', 
  'uncompressed_avi_url', 
  'uncompressed_avi_mirror_1_url', 
  'uncompressed_avi_mirror_2_url'
]
SCHEMA_PK__VIDEO_INDEX = [SCHEMA_COL_NAMES__VIDEO_INDEX[0]]
# ********** SCHEMA-related (FIXED) globals: END **********


# the following globals are set at runtime
DATA_ROOT_DIR = None
VIDEO_DIR = None
STICHED_VIDEO_FRAMES_DIR = None
CORPUS_DS_PATH = None
ASL_CONSULTANT_DS_PATH = None
VIDEO_DS_PATH = None
UTTERANCE_DS_PATH = None
UTTERANCE_VIDEO_DS_PATH = None
UTTERANCE_TOKEN_DS_PATH = None
VOCABULARY_DS_PATH = None
CORPUS_DS_PATH = None
DOCUMENT_ASL_CONSULTANT_DS_PATH = None
ASL_CONSULTANT_DS_PATH = None
VIDEO_DS_PATH = None
VIDEO_SEGMENT_DS_PATH = None
UTTERANCE_DS_PATH = None
UTTERANCE_VIDEO_DS_PATH = None
UTTERANCE_TOKEN_DS_PATH = None
UTTERANCE_TOKEN_FRAME_DS_PATH = None
VOCABULARY_DS_PATH = None
# **************************************** global variables: END ****************************************




df_decomposition = pd.DataFrame(columns=['src_video', 'dest_dir', 'n_frames'])




# **************************************** global functions: BEGIN ****************************************
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

  if not os.path.isdir(d_vid_indexes_info['vid_indexes_dir']) or not os.path.isfile(d_vid_indexes_info['sel_vid_index_path']):
    print(f"video index boostrap info: {d_vid_indexes_info}")
    remote_archive_path = os.path.join('http://www.bu.edu/asllrp/ncslgr-for-download', d_vid_indexes_info['video_indexes_archive'])
    local_archive_parent_dir = d_vid_indexes_info['tmp_dir']
    local_archive_path = os.path.join(local_archive_parent_dir, d_vid_indexes_info['video_indexes_archive'])
    utils.download(
        remote_archive_path, 
        local_archive_path, 
        block_sz=_1MB
    )
    zip_ref = zipfile.ZipFile(local_archive_path, 'r')
    print(f"unzipping {local_archive_path} to {d_vid_indexes_info['vid_indexes_dir']}...")
    zip_ref.extractall(local_archive_parent_dir)
    zip_ref.close()
    print(f"\tDONE")
    print(f"deleting {local_archive_path}...")
    os.remove(local_archive_path)
    print(f"\tDONE")
  else:
    print(f"Found video index {d_vid_indexes_info['sel_vid_index_path']}")
  return d_vid_indexes_info['sel_vid_index_path']


def load_video_index_dataset(debug=False):
  # df_video_index drives the (parallel) download of video segments but it is also used in bootstrapping the corpus as it corresponds to videos
  df_video_index_csv_path = os.path.join(DATA_ROOT_DIR, 'df_video_index.csv')
  if not os.path.isfile(df_video_index_csv_path):
    df_video_index = pd.read_csv(SELECTED_VIDEO_INDEX_PATH)
    df_video_index.rename(
      columns={
        'Video file name in XML file': SCHEMA_COL_NAMES__VIDEO_INDEX[0],
        'Video sequence id': SCHEMA_COL_NAMES__VIDEO_INDEX[1],
        'Perspective/Camera id': SCHEMA_COL_NAMES__VIDEO_INDEX[2],
        'Compressed MOV file': SCHEMA_COL_NAMES__VIDEO_INDEX[3],
        'Uncompressed AVI': SCHEMA_COL_NAMES__VIDEO_INDEX[4],
        'Uncompressed AVI mirror 1': SCHEMA_COL_NAMES__VIDEO_INDEX[5],
        'Uncompressed AVI mirror 2': SCHEMA_COL_NAMES__VIDEO_INDEX[6]
      }, 
      inplace=True
    )
    # NOTE!
    #   This is a CRUCIAL step! We MUST URL encode filenames since some of them sloppily contain spaces!
    df_video_index[SCHEMA_COL_NAMES__VIDEO_INDEX[0]] = df_video_index[SCHEMA_COL_NAMES__VIDEO_INDEX[0]].map(lambda filename: urllib.parse.quote(filename))
    df_video_index.set_index(SCHEMA_COL_NAMES__VIDEO_INDEX[0], inplace=True)
    df_video_index.to_csv(path_or_buf=df_video_index_csv_path)
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(df_video_index_csv_path) else 'FAILED to save'} {df_video_index_csv_path}")

  df_video_index = pd.read_csv(df_video_index_csv_path)
  df_video_index.set_index(SCHEMA_COL_NAMES__VIDEO_INDEX[0], inplace=True)
  return df_video_index


def append_corpus__document(xml_db_path, df_corpus, debug=False):
  xml_db_fname = xml_db_path.split(os.path.sep)[-1]
  if debug:
    print(f"\tfilename: {xml_db_fname}")
  f = beam.io.filesystems.FileSystems.open(xml_db_path)
  if sys.version_info >= (3,0):
    f = io.TextIOWrapper(f)
  xml_lines_with_cr = f.readlines()
  f.close()
  raw_xml = "".join([xml_line.replace('\n', '') for xml_line in xml_lines_with_cr])
  # if debug: # this produces way too much output to stdout; uncomment with caution!
  #   print(f"\tXML (RAW):\n\t\t{raw_xml}")
  doc_id = None
  try:
    df_document_lookup = df_corpus.query(f"{SCHEMA_COL_NAMES__CORPUS_DS[1]}=='{xml_db_fname}'")
    if df_document_lookup.empty:
      data = {
        SCHEMA_COL_NAMES__CORPUS_DS[1]: xml_db_fname,
        SCHEMA_COL_NAMES__CORPUS_DS[2]: raw_xml,
      }
      _doc_id = len(df_corpus)
      df_corpus.loc[_doc_id] = data
      doc_id = _doc_id
    else:
      doc_id = df_document_lookup.index.values[0]
  except Exception as e:
    print(e)
  if debug:
    print(f"\tdoc_id: {doc_id}")
  return doc_id


def append_corpus__video(media, doc_id, df_video_index, df_video, debug=False):
  fname = str(urllib.parse.quote(media.get_filename().split(':')[-1])) # there may be spaces in the fname
  df_video_index_lookup = df_video_index.query(f"{SCHEMA_COL_NAMES__VIDEO_INDEX[0]}=='{fname}'")
  camera_perspective = None if df_video_index_lookup.empty else df_video_index_lookup[SCHEMA_COL_NAMES__VIDEO_INDEX[2]].values[0]
  video_id = None
  try:
    if camera_perspective is None:
      if debug:
        print(f"\t\t{fname}\t\t*** ValueError: video '{fname}' is not in the video index, has no valid camera perspective ***")
    else:
      df_video_lookup = df_video.query(f"{SCHEMA_COL_NAMES__VIDEO_DS[3]}=='{fname}'")
      if df_video_lookup.empty:
        df_video.reset_index(inplace=True)
        data = {
          SCHEMA_COL_NAMES__VIDEO_DS[0]: doc_id,
          SCHEMA_COL_NAMES__VIDEO_DS[2]: camera_perspective,
          SCHEMA_COL_NAMES__VIDEO_DS[3]: fname
        }
        video_id = len(df_video)
        df_video.loc[video_id] = data
        df_video.columns = SCHEMA_COL_NAMES__VIDEO_DS
        df_video.set_index(SCHEMA_PK__VIDEO_DS, inplace=True)
        df_video.sort_index(ascending=[True for c in SCHEMA_PK__VIDEO_DS], inplace=True)
      else:
        # if debug:
        #   print(f"KeyError: video '{fname}' has already been inserted")
        video_id = df_video_lookup.index.values[0]
      if debug:
        print(f"\t\t{fname} (camera perspective {camera_perspective})")
  except Exception as e:
    print(e)
  return video_id


def append_corpus__participant(participant, df_asl_consultant, debug=False):
  reconciled_participant_id = None
  try:
    df_asl_consultant_lookup = df_asl_consultant.query(f"{SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[1]}=='{participant.get_name()}'")
    if not df_asl_consultant_lookup.empty:
      # if debug:
      #   print(f"KeyError: participant '{participant.get_name()}' has already been inserted")
      reconciled_participant_id = df_asl_consultant_lookup.index.values[0]
    else:
      data = {
        SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[1]: participant.get_name(),
        SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[2]: participant.get_age(),
        SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[3]: participant.get_gender()
        # , 'language': participant.get_language()
      }
      reconciled_participant_id = len(df_asl_consultant)
      df_asl_consultant.loc[reconciled_participant_id] = data
    if debug:
      print(f"\tParticipant: id: [{participant.get_id()} (in-file), {reconciled_participant_id} (in-dataset)], name: {participant.get_name()}, age: {participant.get_age()}, gender: {participant.get_gender()}, language: {participant.get_language()}")
  except Exception as e:
    print(e)
  return reconciled_participant_id


def append_corpus__document_participant_mapping(doc_id, reconciled_participant_id, df_document_asl_consultant):
  insert_document_asl_consultant = False
  try:
    df_document_asl_consultant_lookup = df_document_asl_consultant.loc[([doc_id], [reconciled_participant_id]), :] # this will raise KeyError if (doc_id, reconciled_participant_id) is not found
    insert_document_asl_consultant = df_document_asl_consultant_lookup.empty
  except KeyError as ke:
    insert_document_asl_consultant = True
  except Exception as e:
    print(e)
  if insert_document_asl_consultant:
    df_document_asl_consultant.reset_index(inplace=True)
    data = {
      SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0]: doc_id,
      SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[1]: reconciled_participant_id
    }
    df_document_asl_consultant.loc[len(df_document_asl_consultant)] = data
    df_document_asl_consultant.columns = SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS
    df_document_asl_consultant.set_index(SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
    df_document_asl_consultant.sort_index(ascending=[True for c in SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS], inplace=True)


def append_corpus__utterance(
  doc_id, 
  reconciled_participant_id, 
  ui, 
  utterance_time_codes, 
  utterance_main_gloss, 
  utterance_translation,
  df_utterance
):
  df_utterance.reset_index(inplace=True)
  try:
    data = {
      SCHEMA_COL_NAMES__UTTERANCE_DS[0]: doc_id,
      SCHEMA_COL_NAMES__UTTERANCE_DS[1]: reconciled_participant_id,
      SCHEMA_COL_NAMES__UTTERANCE_DS[2]: ui,
      SCHEMA_COL_NAMES__UTTERANCE_DS[3]: utterance_time_codes[0],
      SCHEMA_COL_NAMES__UTTERANCE_DS[4]: utterance_time_codes[1],
      SCHEMA_COL_NAMES__UTTERANCE_DS[5]: utterance_main_gloss,
      SCHEMA_COL_NAMES__UTTERANCE_DS[6]: utterance_translation
    }
    df_utterance.loc[len(df_utterance)] = data
  except Exception as e:
    print(e)
  df_utterance.columns = SCHEMA_COL_NAMES__UTTERANCE_DS
  df_utterance.set_index(SCHEMA_PK__UTTERANCE_DS, inplace=True)
  df_utterance.sort_index(ascending=[True for c in SCHEMA_PK__UTTERANCE_DS], inplace=True)


def append_corpus__vocabulary(token, df_vocabulary):
  tkn = token.get_text().encode('utf-8') # must be encoded as binary since token can have punctuation and possibly other non-alphabetic characters
  token_id = None
  try:
    df_token_lookup = df_vocabulary.query(f"{SCHEMA_COL_NAMES__VOCABULARY_DS[1]}=={tkn}")
    if df_token_lookup.empty:
      data = {
        SCHEMA_COL_NAMES__VOCABULARY_DS[1]: tkn
      }
      _token_id = len(df_vocabulary)
      df_vocabulary.loc[_token_id] = data
      token_id = _token_id
    else:
      token_id = df_token_lookup.index.values[0]
  except Exception as e:
    print(e)
  return token_id


def append_corpus__utterance_token(
  token, 
  doc_id,
  reconciled_participant_id,
  ui,
  ti,
  token_id,
  df_utterance_token
):
  token_time_codes = token.get_timecodes()
  df_utterance_token.reset_index(inplace=True)
  try:
    field = token.get_field().get_name()
    field_value = None
    try:
      field_value = token.get_field_value().get_name()
    except:
      pass
    data = {
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[0]: doc_id,
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[1]: reconciled_participant_id,
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[2]: ui,
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[3]: ti,
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[4]: token_time_codes[0],
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[5]: token_time_codes[1],
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[6]: token_id,
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[7]: field,
      SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[8]: field_value
    }
    df_utterance_token.loc[len(df_utterance_token)] = data
  except Exception as e:
    print(e)
  df_utterance_token.columns = SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS
  df_utterance_token.set_index(SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
  df_utterance_token.sort_index(ascending=[True for c in SCHEMA_PK__UTTERANCE_DS], inplace=True)


def append_corpus__utterance_video(doc_id, reconciled_participant_id, ui, df_video_lookup, df_utterance_video):
  df_utterance_video.reset_index(inplace=True)
  try:
    data = {
      SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[0]: doc_id,
      SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[1]: reconciled_participant_id,
      SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[2]: ui,
      SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[3]: df_video_lookup[SCHEMA_COL_NAMES__VIDEO_DS[2]].values[0]
    }
    df_utterance_video.loc[len(df_utterance_video)] = data
  except Exception as e:
    print(e)
  df_utterance_video.columns = SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS
  df_utterance_video.set_index(SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
  df_utterance_video.sort_index(ascending=[True for c in SCHEMA_PK__UTTERANCE_VIDEO_DS], inplace=True)


def update_corpus__video____append_corpus__utterance_video(doc_id, fname, reconciled_participant_id, ui, df_video, df_utterance_video, debug=False):
  df_video.reset_index(inplace=True)
  df_video_lookup = df_video.query(f"{SCHEMA_COL_NAMES__VIDEO_DS[3]}=='{fname}'") # there must be exactly one
  try:
    if len(df_video_lookup) == 1:
      existing_participant_id = df_video_lookup[SCHEMA_COL_NAMES__VIDEO_DS[1]].values[0]
      if not pd.isna(existing_participant_id) and existing_participant_id != reconciled_participant_id:
        if debug:
          print(f"\t\t\t\t\tValueError: existing participant_id ({existing_participant_id}) for video entry corresponding to '{fname}' conflicts with this participant_id ({reconciled_participant_id})")
      else:
        df_video.loc[df_video_lookup.index, SCHEMA_COL_NAMES__VIDEO_DS[1]] = reconciled_participant_id
        if debug:
          print(f"\t\t\t\t\t{fname}")
    else:
      if debug:
        print(f"\t\t\t\t\tValueError: cannot update df_video since video '{fname}' does not have exactly one entry")
  except Exception as e:
    print(e)
  append_corpus__utterance_video(doc_id, reconciled_participant_id, ui, df_video_lookup, df_utterance_video)
  # don't forget to re-apply original index
  df_video.columns = SCHEMA_COL_NAMES__VIDEO_DS
  df_video.set_index(SCHEMA_PK__VIDEO_DS, inplace=True)
  df_video.sort_index(ascending=[True for c in SCHEMA_PK__VIDEO_DS], inplace=True)


def boostrap_signstream_corpus(d_corpus_info, df_video_index):
  """
  d_corpus_info MUST be a dict as follows:
    {
      'tmp_dir': TMP_DIR,
      'data_dir': DATA_ROOT_DIR,
      'corpus_archive': CORPUS_ARCHIVE, 
      'corpus_ds_path': CORPUS_DS_PATH,
      'document_asl_consultant_ds_path': DOCUMENT_ASL_CONSULTANT_DS_PATH,
      'asl_consultant_ds_path': ASL_CONSULTANT_DS_PATH,
      'video_ds_path': VIDEO_DS_PATH,
      # 'video_segment_ds_path':VIDEO_SEGMENT_DS_PATH, # this is handled in boostrap_video_index()
      'utterance_ds_path': UTTERANCE_DS_PATH,
      'utterance_video_ds_path': UTTERANCE_VIDEO_DS_PATH,
      'utterance_token_ds_path': UTTERANCE_TOKEN_DS_PATH,
      # 'utterance_token_frame_ds_path': UTTERANCE_TOKEN_FRAME_DS_PATH, # this is handled in boostrap_video_index()
      'vocabulary_ds_path': VOCABULARY_DS_PATH
    }

  this function downloads d_corpus_info['corpus_archive'] from http://secrets.rutgers.edu/dai/xml
    and extracts it to os.path.join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])
    (assuming that has not already been done - i.e. if not os.path.isdir(os.path.join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])) 
      or len(os.listdir(os.path.join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])))==0
    )
  """

  corpus_parent_dir = d_corpus_info['tmp_dir']
  corpus_dir = os.path.join(corpus_parent_dir, d_corpus_info['corpus_archive'].split('.')[0])

  # if not os.path.isdir(corpus_dir) or len(os.listdir(corpus_dir))==0:
  if not os.path.isfile(d_corpus_info['corpus_ds_path']) \
    or not os.path.isfile(d_corpus_info['document_asl_consultant_ds_path']) \
    or not os.path.isfile(d_corpus_info['asl_consultant_ds_path']) \
    or not os.path.isfile(d_corpus_info['video_ds_path']) \
    or not os.path.isfile(d_corpus_info['utterance_ds_path']) \
    or not os.path.isfile(d_corpus_info['utterance_video_ds_path']) \
    or not os.path.isfile(d_corpus_info['utterance_token_ds_path']) \
    or not os.path.isfile(d_corpus_info['vocabulary_ds_path']):

    print(f"corpus boostrap info: {d_corpus_info}")

    # download archive
    """
    requires:
      d_corpus_info['corpus_archive']
      d_corpus_info['tmp_dir']
    """
    remote_archive_path = os.path.join('http://secrets.rutgers.edu/dai/xml', d_corpus_info['corpus_archive'])
    local_archive_parent_dir = d_corpus_info['tmp_dir']
    local_archive_path = os.path.join(local_archive_parent_dir, d_corpus_info['corpus_archive'])
    utils.download(
        remote_archive_path, 
        local_archive_path, 
        block_sz=_1MB
    )
    zip_ref = zipfile.ZipFile(local_archive_path, 'r')
    print(f"unzipping {local_archive_path} to {corpus_dir}...")
    zip_ref.extractall(corpus_parent_dir)
    zip_ref.close()
    print(f"\tDONE")
    print(f"deleting {local_archive_path}...")
    os.remove(local_archive_path)
    print(f"\tDONE")

    # create/save datasets from corpus docs using SignStream parser
    df_corpus = pd.DataFrame(columns=SCHEMA_COL_NAMES__CORPUS_DS)
    df_corpus.set_index(SCHEMA_PK__CORPUS_DS, inplace=True)
    df_document_asl_consultant = pd.DataFrame(columns=SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS)
    df_document_asl_consultant.set_index(SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
    df_asl_consultant = pd.DataFrame(columns=SCHEMA_COL_NAMES__ASL_CONSULTANT_DS)
    df_asl_consultant.set_index(SCHEMA_PK__ASL_CONSULTANT_DS, inplace=True)
    df_video = pd.DataFrame(columns=SCHEMA_COL_NAMES__VIDEO_DS)
    df_video.set_index(SCHEMA_PK__VIDEO_DS, inplace=True)
    # df_video_segment = pd.DataFrame(columns=SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS) # created in boostrap_video_index() ??
    # df_video_segment.set_index(SCHEMA_PK__VIDEO_SEGMENT_DS, inplace=True)
    df_utterance = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_DS)
    df_utterance.set_index(SCHEMA_PK__UTTERANCE_DS, inplace=True)
    df_utterance_video = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS)
    df_utterance_video.set_index(SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
    df_utterance_token = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS)
    df_utterance_token.set_index(SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
    # df_utterance_token_frame = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS) # created in boostrap_video_index() ??
    # df_utterance_token_frame.set_index(SCHEMA_PK__UTTERANCE_TOKEN_FRAME_DS, inplace=True)
    df_vocabulary = pd.DataFrame(columns=SCHEMA_COL_NAMES__VOCABULARY_DS)
    df_vocabulary.set_index(SCHEMA_PK__VOCABULARY_DS, inplace=True)

    def format_headshake(head_movements):
      temp = []
      for hm in head_movements:
          (hs, he) = hm.get_timecodes()
          hstext = hm.get_text()
          temp.append("%s (%d-%d)" % (hstext, hs, he))
      return "headshake: " + ", ".join(temp)

    def enum_db(
      xml_db_path,
      df_corpus=None, 
      df_document_asl_consultant=None,
      df_asl_consultant=None,
      df_video=None,
      df_utterance=None,
      df_utterance_video=None,
      df_utterance_token=None,
      df_vocabulary=None,
      debug=False
    ):
      if df_corpus is None:
        df_corpus = pd.DataFrame(columns=SCHEMA_COL_NAMES__CORPUS_DS)
        df_corpus.set_index(SCHEMA_PK__CORPUS_DS, inplace=True)
      if df_document_asl_consultant is None:
        df_document_asl_consultant = pd.DataFrame(columns=SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS)
        df_document_asl_consultant.set_index(SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
      if df_asl_consultant is None:
        df_asl_consultant = pd.DataFrame(columns=SCHEMA_COL_NAMES__ASL_CONSULTANT_DS)
        df_asl_consultant.set_index(SCHEMA_PK__ASL_CONSULTANT_DS, inplace=True)
      if df_video is None:
        df_video = pd.DataFrame(columns=SCHEMA_COL_NAMES__VIDEO_DS)
        df_video.set_index(SCHEMA_PK__VIDEO_DS, inplace=True)
      # if df_video_segment is None:  # created in boostrap_video_index()
      #   df_video_segment = pd.DataFrame(columns=SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS)
      #   df_video_segment.set_index(SCHEMA_PK__VIDEO_SEGMENT_DS, inplace=True)
      if df_utterance is None:
        df_utterance = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_DS)
        df_utterance.set_index(SCHEMA_PK__UTTERANCE_DS, inplace=True)
      if df_utterance_video is None:
        df_utterance_video = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS)
        df_utterance_video.set_index(SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
      if df_utterance_token is None:
        df_utterance_token = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS)
        df_utterance_token.set_index(SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
      # if df_utterance_token_frame is None: # created in boostrap_video_index()
      #   df_utterance_token_frame = pd.DataFrame(columns=SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS) 
      #   df_utterance_token_frame.set_index(SCHEMA_PK__UTTERANCE_TOKEN_FRAME_DS, inplace=True)
      if df_vocabulary is None:
        df_vocabulary = pd.DataFrame(columns=SCHEMA_COL_NAMES__VOCABULARY_DS)
        df_vocabulary.set_index(SCHEMA_PK__VOCABULARY_DS, inplace=True)

      
      if debug:
        print(f"XML-DB: {xml_db_path}")

      # ********** populate df_corpus: BEGIN **********
      doc_id = append_corpus__document(xml_db_path, df_corpus, debug=debug)
      # ********** populate df_corpus: END **********

      # ********** parse (XML) document with SignStream: BEGIN **********
      ss_xml_db = ss.SignStreamDatabase.read_xml(xml_db_path)
      if debug:
        print(f"\tfields:")
        fields = [field for field in ss_xml_db.get_fields()]
        for fi, field in enumerate(fields):
          field_values = [fv.get_name() for fv in field.get_values()]
          print(f"\t\t#{fi}:")
          print(f"\t\t\tname: {field.get_name()}")
          print(f"\t\t\tlabel: {field.get_label()}")
          print(f"\t\t\tvalues:")
          for field_value in field_values:
            print(f"\t\t\t\t{field_value}")
      # ********** parse (XML) document with SignStream: END **********

      # ********** populate df_video: BEGIN **********
      """
      Note that we don't have all information at this point to populate every column
        of the videos dataset.  For now, we only populate the DocumentID, CameraPerspective,
        and Filename columns.
      """
      if debug:
        print(f"\tmedia:")
      for media in ss_xml_db.get_media():
        # ********** populate df_video: BEGIN **********
        video_id = append_corpus__video(media, doc_id, df_video_index, df_video, debug=debug)
        # ********** populate df_video: BEGIN **********

      for participant in ss_xml_db.get_participants():
        # ********** populate df_asl_consultant: BEGIN **********
        reconciled_participant_id = append_corpus__participant(participant, df_asl_consultant, debug=debug)
        # ********** populate df_asl_consultant: END **********

        # ********** populate df_document_asl_consultant: BEGIN **********
        append_corpus__document_participant_mapping(doc_id, reconciled_participant_id, df_document_asl_consultant)
        # ********** populate df_document_asl_consultant: END **********

        # Utterances
        if debug:
          print(f"\t\tutterances:")
        utterances = [utterance for utterance in participant.get_utterances()]
        for ui, utterance in enumerate(utterances):
          token_sequences = [token_sequence for token_sequence in utterance.get_tokens()]
          main_gloss_token_sequence = [token for token in utterance.get_tokens_for_field("main gloss")]
          utterance_main_gloss = ' '.join([token.get_text() for token in main_gloss_token_sequence])
          utterance_translation = ' '.join([token.get_text() for token in token_sequences[-1]])
          utterance_time_codes = utterance.get_timecodes()
          if debug:
            print(f"\t\t\t#{utterance.get_id()} (time-codes: start={utterance_time_codes[0]}, end={utterance_time_codes[1]}):")
            print(f"\t\t\t\tEnglish Translation: {utterance_translation}")
            print(f"\t\t\t\tMain Gloss (Linguistic Tokens): {utterance_main_gloss}")

          # ********** populate df_utterance: BEGIN **********
          append_corpus__utterance(
            doc_id, 
            reconciled_participant_id, 
            ui, 
            utterance_time_codes, 
            utterance_main_gloss, 
            utterance_translation,
            df_utterance
          )
          # ********** populate df_utterance: END **********

          # utterance-token sequences
          if debug:
            print(f"\t\t\t\ttoken sequences:")
            for ti, token_sequence in enumerate(token_sequences):
              token_sequence = [token.get_text() for token in token_sequence]
              print(f"\t\t\t\t\t#{ti}: {' '.join(token_sequence)}")

          # "main gloss" token sequence - these are the linguistic tokens (not English translation tokens... since it's not 1-to-1)
          for ti, token in enumerate(main_gloss_token_sequence):
            # ********** populate df_vocabulary: BEGIN **********
            token_id = append_corpus__vocabulary(token, df_vocabulary)
            # ********** populate df_vocabulary: END **********

            # ********** populate df_utterance_token: BEGIN **********\
            append_corpus__utterance_token(
              token, 
              doc_id,
              reconciled_participant_id,
              ui,
              ti,
              token_id,
              df_utterance_token
            )
            # ********** populate df_utterance_token: END **********

          # media (videos)
          if debug:
            print(f"\t\t\t\tmedia:")
          for media in utterance.get_media():
            fname = str(urllib.parse.quote(media.get_filename().split(':')[-1])) # there may be spaces in the fname
            # ********** update df_video: BEGIN **********
            # ********** populate df_utterance_video: BEGIN **********
            update_corpus__video____append_corpus__utterance_video(
              doc_id, 
              fname, 
              reconciled_participant_id, 
              ui, 
              df_video, 
              df_utterance_video, 
              debug=debug
            )
            # ********** populate df_utterance_video: END **********
            # ********** update df_video: END **********

    xml_db_paths = [os.path.join(corpus_dir, fname) for fname in os.listdir(corpus_dir)]
    dbg = True
    for xml_db_path in xml_db_paths:
      enum_db(
        xml_db_path, 
        df_corpus, 
        df_document_asl_consultant,
        df_asl_consultant,
        df_video,
        df_utterance,
        df_utterance_video,
        df_utterance_token,
        df_vocabulary,
        debug=dbg
      )
      if dbg:
          print()

    df_corpus.to_csv(path_or_buf=d_corpus_info['corpus_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['corpus_ds_path']) else 'FAILED to save'} {d_corpus_info['corpus_ds_path']}")
    df_asl_consultant.to_csv(path_or_buf=d_corpus_info['asl_consultant_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['asl_consultant_ds_path']) else 'FAILED to save'} {d_corpus_info['asl_consultant_ds_path']}")
    df_document_asl_consultant.to_csv(path_or_buf=d_corpus_info['document_asl_consultant_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['document_asl_consultant_ds_path']) else 'FAILED to save'} {d_corpus_info['document_asl_consultant_ds_path']}")
    df_video.to_csv(path_or_buf=d_corpus_info['video_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['video_ds_path']) else 'FAILED to save'} {d_corpus_info['video_ds_path']}")
    df_utterance.to_csv(path_or_buf=d_corpus_info['utterance_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['utterance_ds_path']) else 'FAILED to save'} {d_corpus_info['utterance_ds_path']}")
    df_utterance_video.to_csv(path_or_buf=d_corpus_info['utterance_video_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['utterance_video_ds_path']) else 'FAILED to save'} {d_corpus_info['utterance_video_ds_path']}")
    df_vocabulary.to_csv(path_or_buf=d_corpus_info['vocabulary_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['vocabulary_ds_path']) else 'FAILED to save'} {d_corpus_info['vocabulary_ds_path']}")
    df_utterance_token.to_csv(path_or_buf=d_corpus_info['utterance_token_ds_path'])
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(d_corpus_info['utterance_token_ds_path']) else 'FAILED to save'} {d_corpus_info['utterance_token_ds_path']}")

  else:

    print(f"Found dataset {d_corpus_info['corpus_ds_path']}")
    print(f"Found dataset {d_corpus_info['document_asl_consultant_ds_path']}")
    print(f"Found dataset {d_corpus_info['asl_consultant_ds_path']}")
    print(f"Found dataset {d_corpus_info['video_ds_path']}")
    print(f"Found dataset {d_corpus_info['utterance_ds_path']}")
    print(f"Found dataset {d_corpus_info['utterance_video_ds_path']}")
    print(f"Found dataset {d_corpus_info['utterance_token_ds_path']}")
    print(f"Found dataset {d_corpus_info['vocabulary_ds_path']}")


def load_corpus_datasets(d_corpus_dataset_info, debug=False):
  """
  d_corpus_dataset_info MUST be a dict as follows:
    {
      'corpus_ds_path': CORPUS_DS_PATH,
      'document_asl_consultant_ds_path': DOCUMENT_ASL_CONSULTANT_DS_PATH,
      'asl_consultant_ds_path': ASL_CONSULTANT_DS_PATH,
      'video_ds_path': VIDEO_DS_PATH,
      # 'video_segment_ds_path':VIDEO_SEGMENT_DS_PATH, # this is handled in boostrap_video_index() ??
      'utterance_ds_path': UTTERANCE_DS_PATH,
      'utterance_video_ds_path': UTTERANCE_VIDEO_DS_PATH,
      'utterance_token_ds_path': UTTERANCE_TOKEN_DS_PATH,
      # 'utterance_token_frame_ds_path': UTTERANCE_TOKEN_FRAME_DS_PATH, # this is handled in boostrap_video_index() ??
      'vocabulary_ds_path': VOCABULARY_DS_PATH
    }
  """
  df_corpus = pd.read_csv(d_corpus_dataset_info['corpus_ds_path'])
  df_corpus.set_index(SCHEMA_PK__CORPUS_DS, inplace=True)
  if debug:
    print(f"CORPUS dataset:\n{df_corpus}\n\n")

  df_asl_consultant = pd.read_csv(d_corpus_dataset_info['asl_consultant_ds_path'])
  df_asl_consultant.set_index(SCHEMA_PK__ASL_CONSULTANT_DS, inplace=True)
  if debug:
    print(f"ASL CONSULTANT dataset:\n{df_asl_consultant}\n\n")

  df_document_asl_consultant = pd.read_csv(d_corpus_dataset_info['document_asl_consultant_ds_path'])
  df_document_asl_consultant.set_index(SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
  if debug:
    print(f"DOCUMENT-CONSULTANT (mapping) dataset:\n{df_document_asl_consultant.reset_index()}\n\n") # reset index since it only has keys

  df_video = pd.read_csv(d_corpus_dataset_info['video_ds_path'])
  df_video.set_index(SCHEMA_PK__VIDEO_DS, inplace=True)
  if debug:
    print(f"VIDEO dataset:\n{df_video}\n\n")

  df_utterance = pd.read_csv(d_corpus_dataset_info['utterance_ds_path'])
  df_utterance.set_index(SCHEMA_PK__UTTERANCE_DS, inplace=True)
  if debug:
    print(f"UTTERANCE dataset:\n{df_utterance}\n\n")

  df_utterance_video = pd.read_csv(d_corpus_dataset_info['utterance_video_ds_path'])
  df_utterance_video.set_index(SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
  if debug:
    print(f"UTTERANCE-VIDEO (mapping) dataset:\n{df_utterance_video.reset_index()}\n\n")  # reset index since it only has keys

  df_utterance_token = pd.read_csv(d_corpus_dataset_info['utterance_token_ds_path'])
  df_utterance_token.set_index(SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
  if debug:
    print(f"UTTERANCE-TOKEN (mapping) dataset:\n{df_utterance_token}\n\n")

  df_vocabulary = pd.read_csv(d_corpus_dataset_info['vocabulary_ds_path'])
  df_vocabulary.set_index(SCHEMA_PK__VOCABULARY_DS, inplace=True)
  if debug:
    print(f"VOCBULARY (linguistic tokens) dataset:\n{df_vocabulary}\n\n")

  return (
    df_corpus,
    df_asl_consultant,
    df_document_asl_consultant,
    df_video,
    df_utterance,
    df_utterance_video,
    df_utterance_token,
    df_vocabulary
  )


def download_video_segment(segment_url, data_dir):
  # log_results = []
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.makedirs(data_dir)
  local_segment_path = os.path.join(data_dir, segment_url.split('/')[-1])
  if not tf.io.gfile.exists(local_segment_path):
    # memfile, _ = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False)
    memfile = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False) # returns with memfile.seek(0)
    memfile.seek(0)
    with tf.io.gfile.GFile(name=local_segment_path, mode='w') as f:
      f.write(memfile.getvalue())
    print(f'\tDownloaded {segment_url} to {local_segment_path}')
  else:
    print(f'\tFound target segment {local_segment_path} (from {segment_url})'.format(local_segment_path, segment_url))


def extract_frames(segment_urls, video_fname, frames_dir, videos_dir, df_decomposition):
  # log_results = []

  target_stitched_vid_frames_dir = frames_dir
  target_stitched_vid_name = target_stitched_vid_frames_dir.split(os.path.sep)[-1]
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

        seg_path = local_vid_segment_paths[i]
        seg_fname = seg_path.split(os.path.sep)[-1]
        if n_frames != _n_frames_expected:
          print(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since {_n_frames_expected} frames were expected from segment {seg_fname} ({seg_path}) but only {n_frames} were successfully extracted")
          failed_target_videos.append(video_fname)
          fail = True
          break
        else:
          print(f"\tAdded {n_stitched_frames} frames from segment {seg_fname} for target video {video_fname} (stitched-frames dir {target_stitched_vid_frames_dir})")

      else:
        n_frames = _n_frames_expected
        # nested_tqdm_pb__stitch.update(_n_frames_expected)
        print(f'\tFound existing stiched-frames for {target_stitched_vid_name} ({n_stitched_frames} frames in {target_stitched_vid_frames_dir})')

      # df_decomposition.loc[len(df_decomposition)] = [local_vid_segment_paths[i], target_stitched_vid_frames_dir, n_frames]

  else:
    print(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segments have zero frames")
    failed_target_videos.append(video_fname)
    fail = True  

  # print("\n".join(log_results))

  return df_decomposition


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
  VIDEO_DIR = os.path.join(DATA_ROOT_DIR, 'videos')
  if not tf.io.gfile.exists(VIDEO_DIR):
    tf.io.gfile.makedirs(VIDEO_DIR)

  global STICHED_VIDEO_FRAMES_DIR
  STICHED_VIDEO_FRAMES_DIR = os.path.join(DATA_ROOT_DIR, 'stitched_video_frames')
  if not tf.io.gfile.exists(STICHED_VIDEO_FRAMES_DIR):
    tf.io.gfile.makedirs(STICHED_VIDEO_FRAMES_DIR)

  global CORPUS_DS_PATH
  CORPUS_DS_PATH = os.path.join(DATA_ROOT_DIR, CORPUS_DS_FNAME)

  global DOCUMENT_ASL_CONSULTANT_DS_PATH
  DOCUMENT_ASL_CONSULTANT_DS_PATH = os.path.join(DATA_ROOT_DIR, DOCUMENT_ASL_CONSULTANT_DS_FNAME)

  global ASL_CONSULTANT_DS_PATH
  ASL_CONSULTANT_DS_PATH = os.path.join(DATA_ROOT_DIR, ASL_CONSULTANT_DS_FNAME)

  global VIDEO_DS_PATH
  VIDEO_DS_PATH = os.path.join(DATA_ROOT_DIR, VIDEO_DS_FNAME)

  global UTTERANCE_DS_PATH
  UTTERANCE_DS_PATH = os.path.join(DATA_ROOT_DIR, UTTERANCE_DS_FNAME)

  global UTTERANCE_VIDEO_DS_PATH
  UTTERANCE_VIDEO_DS_PATH = os.path.join(DATA_ROOT_DIR, UTTERANCE_VIDEO_DS_FNAME)

  global UTTERANCE_TOKEN_DS_PATH
  UTTERANCE_TOKEN_DS_PATH = os.path.join(DATA_ROOT_DIR, UTTERANCE_TOKEN_DS_FNAME)

  global VOCABULARY_DS_PATH
  VOCABULARY_DS_PATH = os.path.join(DATA_ROOT_DIR, VOCABULARY_DS_FNAME)

  if beam_bootstrap_video_index:
    import random

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
      fields should be identical to SCHEMA_COL_NAMES__VIDEO_INDEX
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
      return vid_index_csv_rows(sel_vid_index_csv_path, rows_to_dicts=True, dict_field_names=SCHEMA_COL_NAMES__VIDEO_INDEX)


    class PipelinePcollPrinter(beam.DoFn):
      """
      prints each element of the pcoll
      should only be used for debugging - i.e. NOT in production
      """
      def __init__(self, label=""):
        self.label = label

      def process(self, pcoll_element):
          print(f"{self.label+': ' if len(self.label)>0 else ''}{pcoll_element}")
          return [pcoll_element] # passthrough

    class VideoIndexPandasDataframeFromSchemadPcoll(beam.DoFn):
      """
      creates an underlying pandas DataFrame
      appends pcoll dict element to this dataframe
      """
      def __init__(self):
        self.df_video_index = pd.DataFrame(columns=SCHEMA_COL_NAMES__VIDEO_INDEX)
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


    def beam_download_video_segment(d_vid_seg_download_info, label=""):
      """
      expects d_vid_seg_download_info: {'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': url}
      """
      segment_url = d_vid_seg_download_info['segment_url']
      # log_results = []
      if not tf.io.gfile.exists(VIDEO_DIR):
        tf.io.gfile.makedirs(VIDEO_DIR)
      local_segment_path = os.path.join(VIDEO_DIR, segment_url.split('/')[-1])
      if not tf.io.gfile.exists(local_segment_path):
        memfile = utils.download_to_memfile(segment_url, block_sz=_1MB, display=False) # returns with memfile.seek(0)
        memfile.seek(0)
        with tf.io.gfile.GFile(name=local_segment_path, mode='w') as f:
          f.write(memfile.getvalue())
        print(f"\t{label+': ' if len(label)>0 else ''}Downloaded {segment_url} to {local_segment_path}")
      else:
        print(f"\t{label+': ' if len(label)>0 else ''}Found target segment {local_segment_path} (from {segment_url})".format(local_segment_path, segment_url))
      return [d_vid_seg_download_info] # passthrough

    class VideoSegmentExtractor(beam.DoFn):
      def __init__(self, label=""):
        self.label = label

      def process(self, d_vid_seg_download_info):
        return beam_download_video_segment(d_vid_seg_download_info, self.label)

    def beam_extract_frames(segment_urls, video_fname, frames_dir, videos_dir, df_decomposition):
      # log_results = []

      target_stitched_vid_frames_dir = frames_dir
      target_stitched_vid_name = target_stitched_vid_frames_dir.split(os.path.sep)[-1]
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

            seg_path = local_vid_segment_paths[i]
            seg_fname = seg_path.split(os.path.sep)[-1]
            if n_frames != _n_frames_expected:
              print(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since {_n_frames_expected} frames were expected from segment {seg_fname} ({seg_path}) but only {n_frames} were successfully extracted")
              failed_target_videos.append(video_fname)
              fail = True
              break
            else:
              print(f"\tAdded {n_stitched_frames} frames from segment {seg_fname} for target video {video_fname} (stitched-frames dir {target_stitched_vid_frames_dir})")

          else:
            n_frames = _n_frames_expected
            # nested_tqdm_pb__stitch.update(_n_frames_expected)
            print(f'\tFound existing stiched-frames for {target_stitched_vid_name} ({n_stitched_frames} frames in {target_stitched_vid_frames_dir})')

          # df_decomposition.loc[len(df_decomposition)] = [local_vid_segment_paths[i], target_stitched_vid_frames_dir, n_frames]

      else:
        print(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segments have zero frames")
        failed_target_videos.append(video_fname)
        fail = True  

      # print("\n".join(log_results))

      return df_decomposition

    




    vid_index_df_converter = VideoIndexPandasDataframeFromSchemadPcoll()
    
    options = {
      'project': 'my-project', # change
      'runner': 'DirectRunner',
      'direct_num_workers': 0, # 0 is use all available cores
      'direct_running_mode': 'multi_threading', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
      'streaming': False # set to True if data source is unbounded (e.g. GCP PubSub)
    }
    pipeline_options = PipelineOptions(flags=[], **options)
    # pipeline_options = PipelineOptions(
    #   save_main_session=True,
    #   runner='DirectRunner',
    #   direct_num_workers=0,
    #   direct_running_mode='multi_threading', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
    #   streaming=False,
    # )
    # print(f"PipelineOptions:\n{pipeline_options.get_all_options()}")

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
          #       str(urllib.parse.quote(x[SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),  # originally 'Video file name in XML file': str
          #       int(x[SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                      # originally 'Video sequence id': int
          #       int(x[SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                      # originally 'Perspective/Camera id': int
          #       str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),                      # originally 'Compressed MOV file': str (note that this is actually a list with ';' as delimiter)
          #       str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                      # originally 'Uncompressed AVI': str
          #       str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),                      # originally 'Uncompressed AVI mirror 1': str
          #       str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[6]])                       # originally 'Uncompressed AVI mirror 2': str
          #     )
          #   ).with_output_types(VideoIndexEntry)

        # So for now, we settle for beam.Row implementation, which is almost as good (although doesn't respect field order)...
        | "Beam PL: apply schema to video index pcoll" >> beam.Map(lambda x: beam.Row(
              filename=str(urllib.parse.quote(x[SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),  # We MUST URL encode filenames since some of them sloppily contain spaces!
              video_seq_id=int(x[SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                            
              perspective_cam_id=int(x[SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                  
              compressed_mov_url=str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),            # this is actually a list with ';' as delimiter)
              uncompressed_avi_url=str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                     
              uncompressed_avi_mirror_1_url=str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),   
              uncompressed_avi_mirror_2_url=str(x[SCHEMA_COL_NAMES__VIDEO_INDEX[6]])
            )
          )
        # | "Beam PL: print schemad video index pcoll" >> beam.ParDo(PipelinePcollPrinter())  # comment out for production

        # filter schemad pcoll as desired (if necessary) using SqlTransform(), for example limiting size of pcoll data items to max_data_files
        | SqlTransform(f"SELECT * FROM PCOLLECTION LIMIT {max_data_files}")
      )

      # (
      #   vid_index_schemad_pcoll
      #   | 'Count videos queued for download' >> beam.combiners.Count.Globally()
      #   | 'Print result' >> beam.Map(lambda count_pcol_element: print(f"Videos queued for download: {count_pcol_element}"))
      # )

      # this does the job but is much much slower than parallel downloads since each item is processed sequentially
      # (
      #   vid_index_schemad_pcoll
      #   | "Beam PL: gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
      #   # | "Beam PL: print download info for video segments" >> beam.ParDo(PipelinePcollPrinter())  # comment out for production
      #   | "Beam PL: download video segments" >> beam.ParDo(VideoSegmentExtractor())
      # )

      # create as many partitions as we have workers (cores for DirectRunner) available
      #   this is done so that downloads can occur in parallel
      #   we randomly assign each data item to one of the num_partitions partitions 
      p1, p2, p3, p4, p5, p6, p7, p8 = (
      # p1, p2, p3, p4 = (
        vid_index_schemad_pcoll

        # Partition accepts a function that receives the number of partitions, and returns the index of the desired partition for the element. 
        # The number of partitions passed must be a positive integer, and it must return an integer in the range 0 to num_partitions-1.
        | 'Partition' >> beam.Partition(
          lambda vid_index_row, num_partitions: random.randint(0,num_partitions-1), 
          8 # number of partitions we use to download in parallel
        )
      )

      # here, we download in parallel by partition
      (
        p1
        | "Beam PL: p1 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p1 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p1"))  # comment out for production
        | "Beam PL: p1 download video segments" >> beam.ParDo(VideoSegmentExtractor("p1")) # outputs a pcoll with each row as {'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': url}
        | "Beam PL: p1 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p1 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp1: videos downloaded: {count_pcol_element}")) # comment out for production
      )
      (
        p2
        | "Beam PL: p2 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p2 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p2"))  # comment out for production
        | "Beam PL: p2 download video segments" >> beam.ParDo(VideoSegmentExtractor("p2"))
        | "Beam PL: p2 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p2 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp2: videos downloaded: {count_pcol_element}")) # comment out for production
      )
      (
        p3
        | "Beam PL: p3 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p3 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p3"))  # comment out for production
        | "Beam PL: p3 download video segments" >> beam.ParDo(VideoSegmentExtractor("p3"))
        | "Beam PL: p3 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p3 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp3: videos downloaded: {count_pcol_element}")) # comment out for production
      )
      (
        p4
        | "Beam PL: p4 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p4 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p4"))  # comment out for production
        | "Beam PL: p4 download video segments" >> beam.ParDo(VideoSegmentExtractor("p4"))
        | "Beam PL: p4 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p4 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp4: videos downloaded: {count_pcol_element}")) # comment out for production
      )
      (
        p5
        | "Beam PL: p5 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p5 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p5"))  # comment out for production
        | "Beam PL: p5 download video segments" >> beam.ParDo(VideoSegmentExtractor("p5"))
        | "Beam PL: p5 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p5 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp5: videos downloaded: {count_pcol_element}")) # comment out for production
      )
      (
        p6
        | "Beam PL: p6 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p6 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p6"))  # comment out for production
        | "Beam PL: p6 download video segments" >> beam.ParDo(VideoSegmentExtractor("p6"))
        | "Beam PL: p6 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p6 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp6: videos downloaded: {count_pcol_element}")) # comment out for production
      )
      (
        p7
        | "Beam PL: p7 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p7 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p7"))  # comment out for production
        | "Beam PL: p7 download video segments" >> beam.ParDo(VideoSegmentExtractor("p7"))
        | "Beam PL: p7 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p7 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp7: videos downloaded: {count_pcol_element}")) # comment out for production
      )
      (
        p8
        | "Beam PL: p8 gather download info for video segments" >> beam.ParDo(VideoSegmentDownloadInfoGatherer())
        # | "Beam PL: p8 print download info for video segments" >> beam.ParDo(PipelinePcollPrinter("p8"))  # comment out for production
        | "Beam PL: p8 download video segments" >> beam.ParDo(VideoSegmentExtractor("p8"))
        | "Beam PL: p8 count videos downloaded" >> beam.combiners.Count.Globally() | "Beam PL: p8 print result" >> beam.Map(lambda count_pcol_element: print(f"\tp8: videos downloaded: {count_pcol_element}")) # comment out for production
      )



    print(f"Beam PL: ALL DONE!")
    # ************* Test Apache Beam: END *************
    df_video_index = vid_index_df_converter.df_video_index

  else:

    boostrap_video_index(d_vid_indexes_info={
      'vid_indexes_dir': VIDEO_INDEXES_DIR, 
      'sel_vid_index_path': SELECTED_VIDEO_INDEX_PATH, 
      'video_indexes_archive': VIDEO_INDEXES_ARCHIVE, 
      'tmp_dir': TMP_DIR
    })
    df_video_index = load_video_index_dataset(debug=True)
    
    d_corpus_info={
      'tmp_dir': TMP_DIR,
      'data_dir': DATA_ROOT_DIR,
      'corpus_archive': CORPUS_ARCHIVE, 
      'corpus_ds_path': CORPUS_DS_PATH,
      'document_asl_consultant_ds_path': DOCUMENT_ASL_CONSULTANT_DS_PATH,
      'asl_consultant_ds_path': ASL_CONSULTANT_DS_PATH,
      'video_ds_path': VIDEO_DS_PATH,
      'utterance_ds_path': UTTERANCE_DS_PATH,
      'utterance_video_ds_path': UTTERANCE_VIDEO_DS_PATH,
      'utterance_token_ds_path': UTTERANCE_TOKEN_DS_PATH,
      'vocabulary_ds_path': VOCABULARY_DS_PATH
    }
    boostrap_signstream_corpus(d_corpus_info, df_video_index=df_video_index)
    (
      df_corpus,
      df_asl_consultant,
      df_document_asl_consultant,
      df_video,
      df_utterance,
      df_utterance_video,
      df_utterance_token,
      df_vocabulary
    ) = load_corpus_datasets(d_corpus_info, debug=True)

    target_videos = []
    for idx, media_record in df_video_index.iterrows(): 
      # video_fname = media_record['filename'] # idx holds the filename now
      video_fname = idx
      frames_dir = os.path.join(STICHED_VIDEO_FRAMES_DIR, video_fname.split('.')[0])
      urls = media_record['compressed_mov_url'].split(';') # this can be a list, separated by ';'
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
# **************************************** global functions: END ****************************************




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
  print(f"args: {args}")
  run(
    args.max_data_files if args.max_data_files!=-1 else None, 
    os.path.join(args.work_dir, 'data'), 
    beam_bootstrap_video_index=args.beam_bootstrap_video_index
  )
  # **************************************** main: END ****************************************