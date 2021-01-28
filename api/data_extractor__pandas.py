import io
import multiprocessing as mp
import os
import signal
import sys
import urllib
import zipfile
from importlib import import_module

import apache_beam as beam
import pandas as pd
import tensorflow as tf

from . import utils

sxa = import_module('signstreamxmlparser-refactored.analysis', '.')
ss = import_module('signstreamxmlparser-refactored.analysis.signstream', '.')
import cv2
from . import data_extractor__common
from . import fidscs_globals


def _function_wrapper(args_tuple):
  """Function wrapper to call from multiprocessing."""
  function, args = args_tuple
  return function(*args)


def parallel_map(function, iterable):
  """Calls a function for every element in an iterable using multiple cores."""
  if fidscs_globals.FORCE_DISABLE_MULTIPROCESSING:
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

  
def load_video_index_dataset(debug=False):
  # df_video_index drives the (parallel) download of video segments but it is also used in bootstrapping the corpus as it corresponds to videos
  df_video_index_csv_path = os.path.join(fidscs_globals.DATA_ROOT_DIR, 'df_video_index.csv')
  if not os.path.isfile(df_video_index_csv_path):
    df_video_index = pd.read_csv(fidscs_globals.SELECTED_VIDEO_INDEX_PATH)
    df_video_index.rename(
      columns={
        'Video file name in XML file': fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0],
        'Video sequence id': fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[1],
        'Perspective/Camera id': fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2],
        'Compressed MOV file': fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[3],
        'Uncompressed AVI': fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[4],
        'Uncompressed AVI mirror 1': fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[5],
        'Uncompressed AVI mirror 2': fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[6]
      }, 
      inplace=True
    )
    # NOTE!
    #   This is a CRUCIAL step! We MUST URL encode filenames since some of them sloppily contain spaces!
    df_video_index[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]] = df_video_index[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]].map(lambda filename: urllib.parse.quote(filename))
    df_video_index.set_index(fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0], inplace=True)
    df_video_index.to_csv(path_or_buf=df_video_index_csv_path)
    print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(df_video_index_csv_path) else 'FAILED to save'} {df_video_index_csv_path}")

  df_video_index = pd.read_csv(df_video_index_csv_path)
  df_video_index.set_index(fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0], inplace=True)
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
    df_document_lookup = df_corpus.query(f"{fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS[1]}=='{xml_db_fname}'")
    if df_document_lookup.empty:
      data = {
        fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS[1]: xml_db_fname,
        fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS[2]: raw_xml,
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
  df_video_index_lookup = df_video_index.query(f"{fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]}=='{fname}'")
  camera_perspective = None if df_video_index_lookup.empty else df_video_index_lookup[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]].values[0]
  video_id = None
  try:
    if camera_perspective is None:
      if debug:
        print(f"\t\t{fname}\t\t*** ValueError: video '{fname}' is not in the video index, has no valid camera perspective ***")
    else:
      df_video_lookup = df_video.query(f"{fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[3]}=='{fname}'")
      if df_video_lookup.empty:
        df_video.reset_index(inplace=True)
        data = {
          fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[0]: doc_id,
          fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[2]: camera_perspective,
          fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[3]: fname
        }
        video_id = len(df_video)
        df_video.loc[video_id] = data
        df_video.columns = fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS
        df_video.set_index(fidscs_globals.SCHEMA_PK__VIDEO_DS, inplace=True)
        df_video.sort_index(ascending=[True for c in fidscs_globals.SCHEMA_PK__VIDEO_DS], inplace=True)
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
    df_asl_consultant_lookup = df_asl_consultant.query(f"{fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[1]}=='{participant.get_name()}'")
    if not df_asl_consultant_lookup.empty:
      # if debug:
      #   print(f"KeyError: participant '{participant.get_name()}' has already been inserted")
      reconciled_participant_id = df_asl_consultant_lookup.index.values[0]
    else:
      data = {
        fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[1]: participant.get_name(),
        fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[2]: participant.get_age(),
        fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[3]: participant.get_gender()
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
      fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0]: doc_id,
      fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[1]: reconciled_participant_id
    }
    df_document_asl_consultant.loc[len(df_document_asl_consultant)] = data
    df_document_asl_consultant.columns = fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS
    df_document_asl_consultant.set_index(fidscs_globals.SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
    df_document_asl_consultant.sort_index(ascending=[True for c in fidscs_globals.SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS], inplace=True)


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
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[0]: doc_id,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[1]: reconciled_participant_id,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[2]: ui,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[3]: utterance_time_codes[0],
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[4]: utterance_time_codes[1],
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[5]: utterance_main_gloss,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[6]: utterance_translation
    }
    df_utterance.loc[len(df_utterance)] = data
  except Exception as e:
    print(e)
  df_utterance.columns = fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS
  df_utterance.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_DS, inplace=True)
  df_utterance.sort_index(ascending=[True for c in fidscs_globals.SCHEMA_PK__UTTERANCE_DS], inplace=True)


def append_corpus__vocabulary(token, df_vocabulary):
  tkn = token.get_text().encode('utf-8') # must be encoded as binary since token can have punctuation and possibly other non-alphabetic characters
  token_id = None
  try:
    df_token_lookup = df_vocabulary.query(f"{fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS[1]}=={tkn}")
    if df_token_lookup.empty:
      data = {
        fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS[1]: tkn
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
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[0]: doc_id,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[1]: reconciled_participant_id,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[2]: ui,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[3]: ti,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[4]: token_time_codes[0],
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[5]: token_time_codes[1],
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[6]: token_id,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[7]: field,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[8]: field_value
    }
    df_utterance_token.loc[len(df_utterance_token)] = data
  except Exception as e:
    print(e)
  df_utterance_token.columns = fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS
  df_utterance_token.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
  df_utterance_token.sort_index(ascending=[True for c in fidscs_globals.SCHEMA_PK__UTTERANCE_DS], inplace=True)


def append_corpus__utterance_video(doc_id, reconciled_participant_id, ui, df_video_lookup, df_utterance_video):
  df_utterance_video.reset_index(inplace=True)
  try:
    data = {
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[0]: doc_id,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[1]: reconciled_participant_id,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[2]: ui,
      fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[3]: df_video_lookup[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[2]].values[0]
    }
    df_utterance_video.loc[len(df_utterance_video)] = data
  except Exception as e:
    print(e)
  df_utterance_video.columns = fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS
  df_utterance_video.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
  df_utterance_video.sort_index(ascending=[True for c in fidscs_globals.SCHEMA_PK__UTTERANCE_VIDEO_DS], inplace=True)


def update_corpus__video____append_corpus__utterance_video(doc_id, fname, reconciled_participant_id, ui, df_video, df_utterance_video, debug=False):
  df_video.reset_index(inplace=True)
  df_video_lookup = df_video.query(f"{fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[3]}=='{fname}'") # there must be exactly one
  try:
    if len(df_video_lookup) == 1:
      existing_participant_id = df_video_lookup[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[1]].values[0]
      if not pd.isna(existing_participant_id) and existing_participant_id != reconciled_participant_id:
        if debug:
          print(f"\t\t\t\t\tValueError: existing participant_id ({existing_participant_id}) for video entry corresponding to '{fname}' conflicts with this participant_id ({reconciled_participant_id})")
      else:
        df_video.loc[df_video_lookup.index, fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[1]] = reconciled_participant_id
        if debug:
          print(f"\t\t\t\t\t{fname}")
    else:
      if debug:
        print(f"\t\t\t\t\tValueError: cannot update df_video since video '{fname}' does not have exactly one entry")
  except Exception as e:
    print(e)
  append_corpus__utterance_video(doc_id, reconciled_participant_id, ui, df_video_lookup, df_utterance_video)
  # don't forget to re-apply original index
  df_video.columns = fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS
  df_video.set_index(fidscs_globals.SCHEMA_PK__VIDEO_DS, inplace=True)
  df_video.sort_index(ascending=[True for c in fidscs_globals.SCHEMA_PK__VIDEO_DS], inplace=True)


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
        block_sz=fidscs_globals._1MB
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
    df_corpus = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS)
    df_corpus.set_index(fidscs_globals.SCHEMA_PK__CORPUS_DS, inplace=True)
    df_document_asl_consultant = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS)
    df_document_asl_consultant.set_index(fidscs_globals.SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
    df_asl_consultant = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS)
    df_asl_consultant.set_index(fidscs_globals.SCHEMA_PK__ASL_CONSULTANT_DS, inplace=True)
    df_video = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS)
    df_video.set_index(fidscs_globals.SCHEMA_PK__VIDEO_DS, inplace=True)
    # df_video_segment = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS) # created in boostrap_video_index() ??
    # df_video_segment.set_index(fidscs_globals.SCHEMA_PK__VIDEO_SEGMENT_DS, inplace=True)
    df_utterance = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS)
    df_utterance.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_DS, inplace=True)
    df_utterance_video = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS)
    df_utterance_video.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
    df_utterance_token = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS)
    df_utterance_token.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
    # df_utterance_token_frame = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS) # created in boostrap_video_index() ??
    # df_utterance_token_frame.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_TOKEN_FRAME_DS, inplace=True)
    df_vocabulary = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS)
    df_vocabulary.set_index(fidscs_globals.SCHEMA_PK__VOCABULARY_DS, inplace=True)

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
        df_corpus = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS)
        df_corpus.set_index(fidscs_globals.SCHEMA_PK__CORPUS_DS, inplace=True)
      if df_document_asl_consultant is None:
        df_document_asl_consultant = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS)
        df_document_asl_consultant.set_index(fidscs_globals.SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
      if df_asl_consultant is None:
        df_asl_consultant = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS)
        df_asl_consultant.set_index(fidscs_globals.SCHEMA_PK__ASL_CONSULTANT_DS, inplace=True)
      if df_video is None:
        df_video = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS)
        df_video.set_index(fidscs_globals.SCHEMA_PK__VIDEO_DS, inplace=True)
      # if df_video_segment is None:  # created in boostrap_video_index()
      #   df_video_segment = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS)
      #   df_video_segment.set_index(fidscs_globals.SCHEMA_PK__VIDEO_SEGMENT_DS, inplace=True)
      if df_utterance is None:
        df_utterance = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS)
        df_utterance.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_DS, inplace=True)
      if df_utterance_video is None:
        df_utterance_video = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS)
        df_utterance_video.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
      if df_utterance_token is None:
        df_utterance_token = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS)
        df_utterance_token.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
      # if df_utterance_token_frame is None: # created in boostrap_video_index()
      #   df_utterance_token_frame = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS) 
      #   df_utterance_token_frame.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_TOKEN_FRAME_DS, inplace=True)
      if df_vocabulary is None:
        df_vocabulary = pd.DataFrame(columns=fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS)
        df_vocabulary.set_index(fidscs_globals.SCHEMA_PK__VOCABULARY_DS, inplace=True)

      
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
  df_corpus.set_index(fidscs_globals.SCHEMA_PK__CORPUS_DS, inplace=True)
  if debug:
    print(f"CORPUS dataset:\n{df_corpus}\n\n")

  df_asl_consultant = pd.read_csv(d_corpus_dataset_info['asl_consultant_ds_path'])
  df_asl_consultant.set_index(fidscs_globals.SCHEMA_PK__ASL_CONSULTANT_DS, inplace=True)
  if debug:
    print(f"ASL CONSULTANT dataset:\n{df_asl_consultant}\n\n")

  df_document_asl_consultant = pd.read_csv(d_corpus_dataset_info['document_asl_consultant_ds_path'])
  df_document_asl_consultant.set_index(fidscs_globals.SCHEMA_PK__DOCUMENT_ASL_CONSULTANT_DS, inplace=True)
  if debug:
    print(f"DOCUMENT-CONSULTANT (mapping) dataset:\n{df_document_asl_consultant.reset_index()}\n\n") # reset index since it only has keys

  df_video = pd.read_csv(d_corpus_dataset_info['video_ds_path'])
  df_video.set_index(fidscs_globals.SCHEMA_PK__VIDEO_DS, inplace=True)
  if debug:
    print(f"VIDEO dataset:\n{df_video}\n\n")

  df_utterance = pd.read_csv(d_corpus_dataset_info['utterance_ds_path'])
  df_utterance.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_DS, inplace=True)
  if debug:
    print(f"UTTERANCE dataset:\n{df_utterance}\n\n")

  df_utterance_video = pd.read_csv(d_corpus_dataset_info['utterance_video_ds_path'])
  df_utterance_video.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_VIDEO_DS, inplace=True)
  if debug:
    print(f"UTTERANCE-VIDEO (mapping) dataset:\n{df_utterance_video.reset_index()}\n\n")  # reset index since it only has keys

  df_utterance_token = pd.read_csv(d_corpus_dataset_info['utterance_token_ds_path'])
  df_utterance_token.set_index(fidscs_globals.SCHEMA_PK__UTTERANCE_TOKEN_DS, inplace=True)
  if debug:
    print(f"UTTERANCE-TOKEN (mapping) dataset:\n{df_utterance_token}\n\n")

  df_vocabulary = pd.read_csv(d_corpus_dataset_info['vocabulary_ds_path'])
  df_vocabulary.set_index(fidscs_globals.SCHEMA_PK__VOCABULARY_DS, inplace=True)
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
    # memfile, _ = utils.download_to_memfile(segment_url, block_sz=fidscs_globals._1MB, display=False)
    memfile = utils.download_to_memfile(segment_url, block_sz=fidscs_globals._1MB, display=False) # returns with memfile.seek(0)
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
    seg_vid_cap.set(cv2.CAP_PROP_FPS, fidscs_globals.FPS)
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

  return df_decomposition

def run():
  data_extractor__common.boostrap_target_video_index(d_vid_indexes_info={
      'vid_indexes_dir': fidscs_globals.VIDEO_INDEXES_DIR, 
      'sel_vid_index_path': fidscs_globals.SELECTED_VIDEO_INDEX_PATH, 
      'video_indexes_archive': fidscs_globals.VIDEO_INDEXES_ARCHIVE, 
      'tmp_dir': fidscs_globals.TMP_DIR
  })
  df_video_index = load_video_index_dataset(debug=True)

  d_corpus_info={
      'tmp_dir': fidscs_globals.TMP_DIR,
      'data_dir': fidscs_globals.DATA_ROOT_DIR,
      'corpus_archive': fidscs_globals.CORPUS_ARCHIVE, 
      'corpus_ds_path': fidscs_globals.CORPUS_DS_PATH,
      'document_asl_consultant_ds_path': fidscs_globals.DOCUMENT_ASL_CONSULTANT_DS_PATH,
      'asl_consultant_ds_path': fidscs_globals.ASL_CONSULTANT_DS_PATH,
      'video_ds_path': fidscs_globals.VIDEO_DS_PATH,
      'utterance_ds_path': fidscs_globals.UTTERANCE_DS_PATH,
      'utterance_video_ds_path': fidscs_globals.UTTERANCE_VIDEO_DS_PATH,
      'utterance_token_ds_path': fidscs_globals.UTTERANCE_TOKEN_DS_PATH,
      'vocabulary_ds_path': fidscs_globals.VOCABULARY_DS_PATH
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
      frames_dir = os.path.join(fidscs_globals.STICHED_VIDEO_FRAMES_DIR, video_fname.split('.')[0])
      urls = media_record['compressed_mov_url'].split(';') # this can be a list, separated by ';'
      d = {
        'video_fname': video_fname,
        'frames_dir': frames_dir,
        'segment_urls': urls
      }
      target_videos.append(d)

  if not fidscs_globals.MAX_TARGET_VIDEOS:
      fidscs_globals.MAX_TARGET_VIDEOS = len(target_videos)
  assert fidscs_globals.MAX_TARGET_VIDEOS >= 1
  print('Found {} target video records, using {}'.format(len(target_videos), fidscs_globals.MAX_TARGET_VIDEOS))
  target_videos = target_videos[:fidscs_globals.MAX_TARGET_VIDEOS]

  # download data (video segment) files in parallel (on the CPU of the machine - either local or VM in GCP DataFlow)
  #   note that in order to accomplish parallelism, since this uses the file system of the machine, this must be done
  #   using the CPU of the machine
  #   please see the definition of the parallel_map() function for details
  print('Downloading segments for target videos...')
  parallel_map(
      download_video_segment,
      ((seg_url, fidscs_globals.VIDEO_DIR) for tvd in target_videos for seg_url in tvd['segment_urls'])
  )
      
  # extract frames from video segments in parallel (on the CPU of the machine - either local or VM in GCP DataFlow)
  #   note that in order to accomplish parallelism, since this uses the file system of the machine, this must be done
  #   using the CPU of the machine
  #   please see the definition of the parallel_map() function for details
  print('\nExtracting and aggregating frames from video-segments into target video-frames directories ...')
  df_decomposition = pd.DataFrame(columns=['src_video', 'dest_dir', 'n_frames'])
  parallel_map(
      extract_frames,
      ((
          tvd['segment_urls'],      # segment_urls
          tvd['video_fname'],       # video_fname
          tvd['frames_dir'],
          fidscs_globals.VIDEO_DIR,
          df_decomposition
      ) for tvd in target_videos) 
  )
  df_decomposition.to_csv(path_or_buf=os.path.join(fidscs_globals.DATA_ROOT_DIR, 'df_decomposition.csv'))
  df_decomposition_csv_path = os.path.join(fidscs_globals.DATA_ROOT_DIR, 'df_decomposition.csv')
  print(f"{'SUCCESSFULLY saved' if tf.io.gfile.exists(df_decomposition_csv_path) else 'FAILED to save'} {df_decomposition_csv_path}")

  return df_video_index, df_decomposition
