import csv
import io
import os
import random
import sys
import typing
import urllib
import zipfile

import apache_beam as beam
# import apache_beam.runners.interactive.interactive_beam as ib
import tensorflow as tf
from apache_beam.io import fileio
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.sql import SqlTransform

import globals
import utils
import preprocessor__common
import copy

from importlib import import_module
sxa = import_module('.analysis', 'signstreamxmlparser-refactored')
ss = import_module('.signstream', 'signstreamxmlparser-refactored.analysis')
import cv2
import time
import re


def prepare_output_str(str, label=""):
  return f"{label+': ' if len(label)>0 else ''}{str}"


def boostrap_signstream_corpus(d_corpus_info, label=""):
  """
  d_corpus_info MUST be a dict as follows:
    {
      'tmp_dir': TMP_DIR,
      'data_dir': DATA_ROOT_DIR,
      'corpus_archive': CORPUS_ARCHIVE, 
      'corpus_ds_path': CORPUS_DS_PATH,
    }

  this function downloads d_corpus_info['corpus_archive'] from http://secrets.rutgers.edu/dai/xml
    and extracts it to os.path.join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])
    (assuming that has not already been done - i.e. if not os.path.isdir(os.path.join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])) 
      or len(os.listdir(os.path.join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])))==0
    )
  """

  corpus_parent_dir = d_corpus_info['tmp_dir']
  corpus_dir = os.path.join(corpus_parent_dir, d_corpus_info['corpus_archive'].split('.')[0])

  if not tf.io.gfile.exists(d_corpus_info['corpus_ds_path']) \
    or not tf.io.gfile.exists(d_corpus_info['document_asl_consultant_ds_path']) \
    or not tf.io.gfile.exists(d_corpus_info['asl_consultant_ds_path']) \
    or not tf.io.gfile.exists(d_corpus_info['video_ds_path']) \
    or not tf.io.gfile.exists(d_corpus_info['utterance_ds_path']) \
    or not tf.io.gfile.exists(d_corpus_info['utterance_video_ds_path']) \
    or not tf.io.gfile.exists(d_corpus_info['utterance_token_ds_path']) \
    or not tf.io.gfile.exists(d_corpus_info['vocabulary_ds_path']):

    print(prepare_output_str(f"CORPUS INDEX BOOTSTRAP INFO: {d_corpus_info}", label=label))

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
        block_sz=globals._1MB
    )
    zip_ref = zipfile.ZipFile(local_archive_path, 'r')
    print(f"unzipping {local_archive_path} to {corpus_dir}...")
    zip_ref.extractall(corpus_parent_dir)
    zip_ref.close()
    print(f"\tDONE")
    print(f"deleting {local_archive_path}...")
    os.remove(local_archive_path)
    print(f"\tDONE")

    return [os.path.join(corpus_dir,"*")]

  else:

    print(prepare_output_str(f"Found dataset {d_corpus_info['corpus_ds_path']}", label=label))
    print(prepare_output_str(f"Found dataset {d_corpus_info['document_asl_consultant_ds_path']}", label=label))
    print(prepare_output_str(f"Found dataset {d_corpus_info['asl_consultant_ds_path']}", label=label))
    print(prepare_output_str(f"Found dataset {d_corpus_info['video_ds_path']}", label=label))
    print(prepare_output_str(f"Found dataset {d_corpus_info['utterance_ds_path']}", label=label))
    print(prepare_output_str(f"Found dataset {d_corpus_info['utterance_video_ds_path']}", label=label))
    print(prepare_output_str(f"Found dataset {d_corpus_info['utterance_token_ds_path']}", label=label))
    print(prepare_output_str(f"Found dataset {d_corpus_info['vocabulary_ds_path']}", label=label))

    return []


class SignstreamCorpusBootsrapper(beam.DoFn):
  def __init__(self, label=""):
    self.label = label

  def process(self, d_corpus_info):
    return boostrap_signstream_corpus(d_corpus_info, self.label)


def _load_csv(sel_csv_readable_file, rows_to_dicts=False, dict_field_names=None, max_len=None):
  """
  this function opens the "readable file" (as a CSV),
    reads its contents and returns a list of its rows
  
  by default, each row is a list of elements (separated initially by comma (',') of course)

  if rows_to_dicts is True, each row is converted to a dict keyed by field names
    if dict_field_names is None
      csv.DictReader uses the first row in the csv file as field names
    otherwise
      dict_field_names provides field names (keys of each dict)
  """
  if max_len is not None and max_len>0:
    csv.field_size_limit(max_len)
  if sys.version_info >= (3,0):
    sel_csv_readable_file = io.TextIOWrapper(sel_csv_readable_file)
  if rows_to_dicts:
    csv_reader = csv.DictReader(sel_csv_readable_file,fieldnames=dict_field_names,skipinitialspace=True) if dict_field_names is not None else csv.DictReader(sel_csv_readable_file,skipinitialspace=True)
  else:
    csv_reader = csv.reader(sel_csv_readable_file,skipinitialspace=True)
  if dict_field_names is not None:
      next(csv_reader) # skip past first row (contains column names that we do not want to use)
  return csv_reader


def load_csv(sel_csv_path, rows_to_dicts=False, dict_field_names=None, max_len=None):
  return _load_csv(
    beam.io.filesystems.FileSystems.open(sel_csv_path), 
    rows_to_dicts, 
    dict_field_names,
    max_len
  )


# class VideoIndexEntry(typing.NamedTuple):
#   """
#   fields should be identical to SCHEMA_COL_NAMES__VIDEO_INDEX
#   """
#   filename: str                       # 'Video file name in XML file'
#   video_seq_id: int                   # 'Video sequence id'
#   perspective_cam_id: int             # 'Perspective/Camera id'
#   compressed_mov_url: str             # 'Compressed MOV file'
#   uncompressed_avi_url: str           # 'Uncompressed AVI'
#   uncompressed_avi_mirror_1_url: str  # 'Uncompressed AVI mirror 1'
#   uncompressed_avi_mirror_2_url: str  # 'Uncompressed AVI mirror 2'

# # now register this schema with beam as a RowCoder
# beam.coders.registry.register_coder(VideoIndexEntry, beam.coders.RowCoder)

def load_vid_index_csv(sel_csv_path):
  """
  this function simply wraps the call to load_csv() to produce a "schema'd" pcoll
  so we fix the definition of dict_field_names to:
    dict_field_names=['filename', 'video_seq_id', 'perspective_cam_id', 'compressed_mov_url', 'uncompressed_avi_url', 'uncompressed_avi_mirror_1_url', 'uncompressed_avi_mirror_2_url']
  """
  return load_csv(sel_csv_path, rows_to_dicts=True, dict_field_names=globals.SCHEMA_COL_NAMES__VIDEO_INDEX)


class PipelinePcollElementProcessor(beam.DoFn):
  def __init__(self, fn_pcoll_element_processor, kargs=None, return_result=False):
    self.fn_pcoll_element_processor = fn_pcoll_element_processor
    self.kargs = kargs
    self.return_result = return_result

  def process(self, pcoll_element):
    result = self.fn_pcoll_element_processor(pcoll_element, **self.kargs) if self.kargs is not None else self.fn_pcoll_element_processor(pcoll_element)
    return result if self.return_result else [pcoll_element]


def print_element(pcoll_element, label="", msg=""):
    print(f"{label+': ' if len(label)>0 else ''}{msg+': ' if len(msg)>0 else ''}{pcoll_element}")
    return [pcoll_element] # passthrough
class PipelinePcollPrinter(PipelinePcollElementProcessor):
  """
  prints each element of the pcoll
  should generally only be used for debugging
  """
  def __init__(self, label="", msg=""):
    super(PipelinePcollPrinter, self).__init__(
      fn_pcoll_element_processor=print_element,
      kargs={'label':label,'msg':msg},
      return_result=True
    )
    self.label = label
    self.msg = msg


class VideoSegmentInfoGatherer(beam.DoFn):
  """
  assumes pcoll is already schemad
  """
  def process(self, schemad_pcoll_element):
    video_fname = schemad_pcoll_element.filename
    frames_dir = os.path.join(globals.STICHED_VIDEO_FRAMES_DIR, video_fname.split('.')[0])
    urls = schemad_pcoll_element.compressed_mov_url.split(';') # this can be a list, separated by ';'
    return [{'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]} for url in urls]

import time


def beam_download_video_segment(d_vid_seg_download_info, max_fail=3, label=""):
  """
  expects d_vid_seg_download_info: {'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': url, 'segment_fname': url.split('/')[-1]}
  """
  segment_url = d_vid_seg_download_info['segment_url']
  segment_fname = d_vid_seg_download_info['segment_fname']
  if not tf.io.gfile.exists(globals.VIDEO_DIR):
    tf.io.gfile.makedirs(globals.VIDEO_DIR)
  local_segment_path = os.path.join(globals.VIDEO_DIR, segment_fname)
  n_fail = 0
  if not tf.io.gfile.exists(local_segment_path):
    while n_fail < max_fail:
      try:
        memfile = utils.download_to_memfile(segment_url, block_sz=globals._1MB, display=False) # returns with memfile.seek(0)
        memfile.seek(0)
        with tf.io.gfile.GFile(name=local_segment_path, mode='w') as f:
          f.write(memfile.getvalue())
        print(f"{label+': ' if len(label)>0 else ''}Downloaded {segment_url} to {local_segment_path}")
        break
      except Exception as e:
        n_fail += 1
        if n_fail < max_fail:
          print(f"{label+': ' if len(label)>0 else ''}*** {e} ***: fail count: {n_fail}, max fail: {max_fail} --> sleeping 1 second, then trying again...")
          time.sleep(1)
        else:
          print(f"{label+': ' if len(label)>0 else ''}*** {e} ***: fail count: {n_fail}, max fail: {max_fail} --> giving up!")
  else:
    print(f"{label+': ' if len(label)>0 else ''}Found target segment {local_segment_path} (from {segment_url})".format(local_segment_path, segment_url))
  return [d_vid_seg_download_info] # passthrough


class VideoSegmentDownloader(beam.DoFn):
  def __init__(self, label=""):
    self.label = label

  def process(self, d_vid_seg_download_info):
    return beam_download_video_segment(d_vid_seg_download_info, label=self.label)


def beam_extract_frames(tpl_target_video_extraction_info, label=""):
  """
  expects tpl_target_video_extraction_info: (video_fname, list({'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}))
  """

  # # log_results = []
  video_fname = tpl_target_video_extraction_info[0]
  segment_dicts = sorted(tpl_target_video_extraction_info[1], key=lambda segment_dict: segment_dict['segment_fname'])
  frames_dir = segment_dicts[0]['frames_dir']

  target_stitched_vid_frames_dir = frames_dir
  target_stitched_vid_name = target_stitched_vid_frames_dir.split(os.path.sep)[-1]
  if not tf.io.gfile.exists(target_stitched_vid_frames_dir):
    tf.io.gfile.makedirs(target_stitched_vid_frames_dir)

  local_vid_segment_paths = [os.path.join(globals.VIDEO_DIR, segment_dict['segment_fname']) for segment_dict in segment_dicts]
  for segment_dict in segment_dicts:
    segment_dict['n_frames_extracted'] = 0

  vid_caps = [cv2.VideoCapture(local_vid_segment_path) for local_vid_segment_path in local_vid_segment_paths]
  for seg_vid_cap in vid_caps:
    seg_vid_cap.set(cv2.CAP_PROP_FPS, globals.FPS)
  frame_counts = list(map(lambda vc: int(vc.get(cv2.CAP_PROP_FRAME_COUNT)), vid_caps))
  n_frames_expected = sum(frame_counts)

  failed_target_videos = []

  n_stitched_frames = 0
  if n_frames_expected > 0:
    # get count of existing stitched frames in target_stitched_vid_frames_dir
    n_stitched_frames = len(tf.io.gfile.listdir(target_stitched_vid_frames_dir))

    b_restitch = n_stitched_frames < n_frames_expected
    n_stitched_frames = 0 if b_restitch else n_stitched_frames

    for i, seg_vid_cap in enumerate(vid_caps):
      segment_dict = segment_dicts[i]
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
          print(f"{label+': ' if len(label)>0 else ''}***WARNING!!!*** Cannot stitch together target video {video_fname} since {_n_frames_expected} frames were expected from segment {seg_fname} ({seg_path}) but only {n_frames} were successfully extracted")
          failed_target_videos.append(video_fname)
          fail = True
          break
        else:
          print(f"{label+': ' if len(label)>0 else ''}Added {n_stitched_frames} frames from segment {seg_fname} for target video {video_fname} (stitched-frames dir {target_stitched_vid_frames_dir})")

      else:
        n_frames = _n_frames_expected
        # nested_tqdm_pb__stitch.update(_n_frames_expected)
        print(f"{label+': ' if len(label)>0 else ''}Found existing stiched-frames for {target_stitched_vid_name} ({n_stitched_frames} frames in {target_stitched_vid_frames_dir})")

      segment_dict['n_frames_extracted'] = n_frames

  else:
    print(f"\t***WARNING!!!*** Cannot stitch together target video {video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segments have zero frames")
    failed_target_videos.append(video_fname)
    fail = True  

  return [(tpl_target_video_extraction_info[0], n_stitched_frames, segment_dicts)]


class SegmentFrameExtractor(beam.DoFn):
  def __init__(self, label=""):
    self.label = label

  def process(self, tpl_target_video_extraction_info):
    return beam_extract_frames(tpl_target_video_extraction_info, self.label)


def row_to_string(row):
  d_row = row.as_dict()
  return ", ". join([str(d_row[k]) for k in d_row.keys()])


import base64
class CorpusDocumentFileProcessor(beam.DoFn):
  def __init__(self, label=""):
    self.label = label
    self.next_doc_id = 0

  def process(self, corpus_readable_file):
    xml_db_path = corpus_readable_file.metadata.path
    xml_db_fname = xml_db_path.split(os.path.sep)[-1].strip()
    f = beam.io.filesystems.FileSystems.open(xml_db_path)
    if sys.version_info >= (3,0):
      f = io.TextIOWrapper(f)
    xml_lines_with_cr = f.readlines()
    f.close()
    # encode each row to bytes
    raw_xml_b64 = base64.b64encode("".join([xml_line.replace('\n','').strip() for xml_line in xml_lines_with_cr]).encode('ascii')) # we end up with a string containing the base-64 encoded "characters"
    # debug
    # print(f"length of (base-64 encoded) XML document {xml_db_fname}: {len(raw_xml_b64)}")
    row = beam.Row(
      # SCHEMA_COL_NAMES__CORPUS_DS = [
      #   'DocumentID',
      #   'Filename',
      #   'XML_B64',
      #   'LEN'
      # ]
      DocumentID=int(self.next_doc_id),
      Filename=xml_db_fname,
      XML_B64=raw_xml_b64,
      LEN=len(raw_xml_b64)
    )
    self.next_doc_id += 1
    return [row]


def load_corpus_index_csv(d_corpus_info):
  """
  this function simply wraps the call to load_csv() to produce a "schema'd" pcoll
  so we fix the definition of dict_field_names to globals.SCHEMA_COL_NAMES__CORPUS_DS

  d_corpus_info: {
    'corpus_index_csv_path': globals.CORPUS_DS_PATH,
    'max_len': globals.MAX_RAW_XML_B64_LEN
  }
  """
  return load_csv(
    d_corpus_info['corpus_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=globals.SCHEMA_COL_NAMES__CORPUS_DS,
    max_len=d_corpus_info['max_len']+4 # note that we need 4 more bytes since due to base-64 encoding
  )
  

class RowIndexer(beam.DoFn):
  def __init__(self, var_name_prefix):
    self.var_name = var_name_prefix+"_next_id"

  def process(self, element):
    tpl = (globals.D_IN_MEMORY_VARS.get(self.var_name, 0), element)
    globals.D_IN_MEMORY_VARS[self.var_name] = globals.D_IN_MEMORY_VARS.get(self.var_name, 0)+1
    return [tpl]


def decode_XML(d_corpus_index_schemad_pcoll_row):
  """
  d_corpus_index_schemad_pcoll_row: {'DocumentID': <corpus doc id (as string!)>, 'Filename': <corpus doc filename>, 'XML_B64': <raw xml (base64-encoded)>, 'LEN': <length of raw xml (base64-encoded)>}
  """
  raw_XML_b64_as_str = d_corpus_index_schemad_pcoll_row['XML_B64']
  raw_XML_b64_as_str = str(raw_XML_b64_as_str[2:-1]) # strip
  raw_XML_b64_to_ascii = raw_XML_b64_as_str.encode('ascii')
  raw_XML_b64 = base64.b64decode(raw_XML_b64_to_ascii)
  raw_xml = raw_XML_b64.decode('ascii').strip()
  # print(raw_xml)
  return [
    {
      'DocumentID': d_corpus_index_schemad_pcoll_row['DocumentID'], 
      'Filename': d_corpus_index_schemad_pcoll_row['Filename'],
      'XML': raw_xml,
      'LEN': len(raw_xml)
    }
  ]


class GlobalVarValueAssigner(PipelinePcollElementProcessor):
  def __init__(self, fn_assign_to_global, kargs=None):
    super(GlobalVarValueAssigner, self).__init__(
      fn_pcoll_element_processor=fn_assign_to_global,
      kargs=kargs,
      return_result=True
    )


def assign_to_global__raw_xml_b64_max_len(max_xml_b64_len):
    globals.MAX_RAW_XML_B64_LEN = max_xml_b64_len+4
    # debug
    # print(f"ASSIGNED globals.MAX_RAW_XML_B64_LEN={globals.MAX_RAW_XML_B64_LEN}")
    return [max_xml_b64_len]




def pl__1__bootstrap_video_index(pl):
  # ******************** start the pipeline, bootstrap video index, read it, apply schema: BEGIN ********************
  return (
    pl
    | "Beam PL: create initial pcoll containing information for boostrap_video_index" >> beam.Create(
        [ # one row containing dict of:
            # 1. url of video indexes archive
            # 2. local destination (path) for the downloaded archive
            # 3. local destination (path) which will receive the extracted archive csv files (there are more than one)
            # 4. final path to the selected videx index csv
            #   (note that the dict is not laid out in the above order)
          {
            'vid_indexes_dir': globals.VIDEO_INDEXES_DIR, 
            'sel_vid_index_path': globals.SELECTED_VIDEO_INDEX_PATH, 
            'video_indexes_archive': globals.VIDEO_INDEXES_ARCHIVE, 
            'tmp_dir': globals.TMP_DIR
          }
        ]
      )
    | "Beam PL: bootstrap video index" >> beam.Map(preprocessor__common.boostrap_video_index) # boostrap_video_index outputs SELECTED_VIDEO_INDEX_PATH but beam.Map() wraps this in a pcoll and is fed to...
    | "Beam PL: read video index into pcoll" >> beam.FlatMap(load_vid_index_csv) # outputs another pcoll but with each row as dict
    # note that we want rows as dicts since dicts help us apply a schema to the pcoll, which is what we want in the end

    # now we want to apply the schema so that we can ultimately use beam's SqlTransform (very similar to pandas sqldf) when necessary
    | "Beam PL: apply schema to video index pcoll" >> beam.Map(lambda x: beam.Row(
          filename=str(urllib.parse.quote(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),  # We MUST URL encode filenames since some of them sloppily contain spaces!
          video_seq_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                            
          perspective_cam_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                  
          compressed_mov_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),            # this is actually a list with ';' as delimiter)
          uncompressed_avi_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                     
          uncompressed_avi_mirror_1_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),   
          uncompressed_avi_mirror_2_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[6]])
        )
      )
    # | "Beam PL: print schemad video index pcoll" >> beam.ParDo(PipelinePcollPrinter())  # passthrough but comment out for production
  )
  # ******************** start the pipeline, bootstrap video index, read it, apply schema: END ********************

def pl__2__write_vid_index_to_storage(full_vid_index_schemad_pcoll):
  # ******************** write video index to storage as CSV: BEGIN ********************
  return (
    full_vid_index_schemad_pcoll
    | beam.Map(lambda vid_index_schemad_pcoll_row: row_to_string(vid_index_schemad_pcoll_row))
    | "Beam PL: write video index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_INDEXES_ARCHIVE.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=", ".join(globals.SCHEMA_COL_NAMES__VIDEO_INDEX)
      )
    | "Beam PL: print path to corpus index csv" >> beam.ParDo(PipelinePcollPrinter(msg="VIDEO INDEX CSV WRITTEN TO STORAGE"))
  )
  # ******************** write video index to storage as CSV: END ********************

def pl__1__read_vid_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing path to load the video index csv" >> beam.Create([os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_INDEXES_ARCHIVE.split('.')[0]+'.csv')])
    | "Beam PL: read video index into pcoll" >> beam.FlatMap(load_vid_index_csv) # outputs another pcoll but with each row as dict
    | "Beam PL: apply schema to video index pcoll" >> beam.Map(lambda x: beam.Row(
          filename=str(urllib.parse.quote(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),  # We MUST URL encode filenames since some of them sloppily contain spaces!
          video_seq_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                            
          perspective_cam_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                  
          compressed_mov_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),            # this is actually a list with ';' as delimiter)
          uncompressed_avi_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                     
          uncompressed_avi_mirror_1_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),   
          uncompressed_avi_mirror_2_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[6]])
        )
      )
  )

def pl__2__filter_vid_index(full_vid_index_schemad_pcoll):
  # ******************** filter schemad video index pcoll as desired (if necessary) using SqlTransform(), for example limiting size of pcoll data items to globals.MAX_DATA_FILES: BEGIN ********************
  return (
    full_vid_index_schemad_pcoll
    | SqlTransform(f"SELECT * FROM PCOLLECTION {'LIMIT '+str(globals.MAX_DATA_FILES) if globals.MAX_DATA_FILES is not None and globals.MAX_DATA_FILES>0 else ''}")
  )
  # ******************** filter schemad video index pcoll as desired (if necessary) using SqlTransform(), for example limiting size of pcoll data items to globals.MAX_DATA_FILES: END ********************

def pl__1__bootstrap_corpus_index(pl):
  # ******************** bootstrap SignStream corpus: BEGIN ********************
  corpus_documents_dir_path_schemad_pcoll = (
    pl
    | "Beam PL: create initial pcoll containing information for boostrap_signstream_corpus" >> beam.Create(
        [
          {
            'tmp_dir': globals.TMP_DIR,
            'data_dir': globals.DATA_ROOT_DIR,
            'corpus_archive': globals.CORPUS_ARCHIVE, 
            'corpus_ds_path': globals.CORPUS_DS_PATH,
            'document_asl_consultant_ds_path': globals.DOCUMENT_ASL_CONSULTANT_DS_PATH,
            'asl_consultant_ds_path': globals.ASL_CONSULTANT_DS_PATH,
            'video_ds_path': globals.VIDEO_DS_PATH,
            'utterance_ds_path': globals.UTTERANCE_DS_PATH,
            'utterance_video_ds_path': globals.UTTERANCE_VIDEO_DS_PATH,
            'utterance_token_ds_path': globals.UTTERANCE_TOKEN_DS_PATH,
            'vocabulary_ds_path': globals.VOCABULARY_DS_PATH
          }
        ]
      )
    | "Beam PL: bootstrap SignStream corpus" >> beam.FlatMap(boostrap_signstream_corpus) # boostrap_signstream_corpus outputs [os.path.join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'].split('.')[0])] if datasets do not yet exist, otherwise []
    | "Beam PL: apply schema to corpus document files path pcoll" >> beam.Map(lambda x: beam.Row(
          corpus_docs_dir=str(x)
        )
      )
    # | SqlTransform(f"SELECT * FROM PCOLLECTION")
    # | "Beam PL: print path to corpus dir" >> beam.ParDo(PipelinePcollPrinter())
  )
  return corpus_documents_dir_path_schemad_pcoll
  # ******************** bootstrap SignStream corpus: END ********************

def pl__1__corpus_document_file_structure_to_corpus_index(pl):
  return (
    pl
    | "Beam PL: get corpus documents" >> fileio.MatchFiles(os.path.join(os.path.join(globals.TMP_DIR, globals.CORPUS_ARCHIVE.split('.')[0]), "*"))
    | "Beam PL: read corpus documents" >> fileio.ReadMatches()
    | "Beam PL: create corpus index dataset" >> beam.ParDo(CorpusDocumentFileProcessor())
  ) # corpus_index_schemad_pcoll


def pl__2__write_corpus_index_to_storage(corpus_index_schemad_pcoll, global_var_value_assigner__raw_xml_b64_max_len):
  # ******************** write corpus index to storage as CSV: BEGIN ********************
  corpus_index_csv_path = (
    corpus_index_schemad_pcoll
    | beam.Map(lambda corpus_index_schemad_pcoll_row: row_to_string(corpus_index_schemad_pcoll_row))
    | "Beam PL: write corpus index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.CORPUS_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=", ".join(globals.SCHEMA_COL_NAMES__CORPUS_DS)
      )
    | "Beam PL: print path to corpus index csv" >> beam.ParDo(PipelinePcollPrinter(msg="CORPUS INDEX CSV WRITTEN TO STORAGE"))
  )
  max_xml_b64_len = (
    corpus_index_schemad_pcoll
    | "Beam PL: select LEN" >> beam.Map(lambda corpus_index_schemad_pcoll_row: corpus_index_schemad_pcoll_row.LEN)
    | beam.CombineGlobally(lambda corpus_index_b64_doc_length_rows: max(corpus_index_b64_doc_length_rows or [None]))
    # debug
    # | "Beam PL: print max (b64-encoded) length corpus doc" >> beam.ParDo(PipelinePcollPrinter(msg="MAX (b64-encoded) DOC LENGTH"))
  )
  corpus_index_csv_path_indexed = (
    corpus_index_csv_path
    | "Beam PL: apply RowIndex to corpus index csv path" >> beam.ParDo(RowIndexer(var_name_prefix="corpus_index_csv_path_id"))
    # debug
    # | "Beam PL: print indexed path to corpus index csv" >> beam.ParDo(PipelinePcollPrinter(msg="INDEXED CORPUS INDEX CSV PATH"))
  )
  max_xml_b64_len_indexed = (
    max_xml_b64_len
    | "Beam PL: assign to global var (globals.MAX_RAW_XML_B64_LEN)" >> beam.ParDo(global_var_value_assigner__raw_xml_b64_max_len) 
    | "Beam PL: apply RowIndex to maxlen" >> beam.ParDo(RowIndexer(var_name_prefix="max_xml_b64_len_id"))
    # debug
    # | "Beam PL: print indexed max (b64-encoded) length corpus doc" >> beam.ParDo(PipelinePcollPrinter(msg="INDEXED MAX (b64-encoded) DOC LENGTH"))
  )
  combined_results = (
    ({
      'corpus_index_csv_path': corpus_index_csv_path_indexed,
      'max_len': max_xml_b64_len_indexed
    })
    | "Beam PL: merge corpus_index_csv_path and max_len" >> beam.CoGroupByKey()
    # debug
    # | "Beam PL: print combined results" >> beam.ParDo(PipelinePcollPrinter(msg="READ CORPUS INDEX CSV TO PCOLL"))
  )
  return combined_results
  # ******************** write corpus index to storage as CSV: END ********************

def pl__1__read_corpus_index_csv(pl):
  # return (
  #   tpl_combined_results__corpus_index_csv_path__max_len
  #   # debug
  #   # | "Beam PL: read tpl_combined_results__corpus_index_csv_path__max_len" >> beam.ParDo(PipelinePcollPrinter(msg="READ tpl_combined_results__corpus_index_csv_path__max_len"))
  #   | "Beam PL: read corpus index into pcoll" >> beam.FlatMap(load_corpus_index_csv) # outputs another pcoll but with each row as dict (with globals.SCHEMA_COL_NAMES__CORPUS_DS keys)
  # )
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the corpus index csv" >> beam.Create(
        [{
          'corpus_index_csv_path': globals.CORPUS_DS_PATH,
          'max_len': globals.MAX_RAW_XML_B64_LEN
        }]
      )
    | "Beam PL: read corpus index into pcoll" >> beam.FlatMap(load_corpus_index_csv) # outputs another pcoll but with each row as dict (with globals.SCHEMA_COL_NAMES__CORPUS_DS keys)
  )

def pl__2__decode_XML(corpus_index_schemad_pcoll):
  # each row is of the form {'DocumentID': '37', 'Filename': ' biker.xml', 'XML_B64', 'LEN'}
  return (
    corpus_index_schemad_pcoll
    | "Beam PL: extract/decode base-64 encoded XML from corpus index document" >> beam.Map(decode_XML)
  )


def parse_signstream_database(corpus_index_decoded_XML_pcoll_row):
  d_corpus_index_decoded_XML = corpus_index_decoded_XML_pcoll_row[0]
  """
  require:
    d_corpus_index_decoded_XML:
    {
      'DocumentID': d_corpus_index_schemad_pcoll_row['DocumentID'], 
      'Filename': d_corpus_index_schemad_pcoll_row['Filename'],
      'XML': raw_xml,
      'LEN': len(raw_xml)
    }

  return:
    {
      'CORPUS_DOCUMENT_FILENAME': <corpus doc filename>, 

      'PARTICIPANT_SEQUENCE': [
        {
          'PARTICIPANT_NAME': <participant name>,
          'PARTICIPANT_AGE': <participant age>,
          'PARTICIPANT_GENDER': <participant gender>,

          'UTTERANCE_SEQUENCE': [
            {
              'UTTERANCE_ENGLISH_TRANSLATION': <utterance English translation>,
              'UTTERANCE_START_TIME': <utterance start time (time code)>,
              'UTTERANCE_END_TIME': <utterance end time (time code)>,

              'MEDIA_SEQUENCE': [
                {
                  'MEDIA_FNAME': <media fname>,
                  'MEDIA_CAMERA_PERSPECTIVE': <media camera perspective>,
                  'MEDIA_URL': <media url>
                }
              ]

              'TOKEN_SEQUENCE': [
                {
                  'TOKEN_LINGUSTIC_TEXT': <token linguistic text>,
                  'TOKEN_START_TIME': <token start time (time code)>,
                  'TOKEN_END_TIME': <token end time (time code)>,
                }
              ]
            }
          ]
        }
      ]
    }
  """

  document_record = {'CORPUS_DOCUMENT_FILENAME': d_corpus_index_decoded_XML['Filename']}
  participant_sequence = []
  # ********** parse (XML) document with SignStream: BEGIN **********
  # debug
  # print(f"length of (ASCII) XML document {d_corpus_index_decoded_XML['Filename']}: {d_corpus_index_decoded_XML['LEN']}")
  in_memory_xml_doc = io.StringIO(d_corpus_index_decoded_XML['XML'])
  ss_xml_db = ss.SignStreamDatabase.read_xml(in_memory_xml_doc)

  for participant in ss_xml_db.get_participants():
    participant_record = {}
    participant_record['PARTICIPANT_NAME'] = participant.get_name()
    participant_record['PARTICIPANT_AGE'] = participant.get_age()
    participant_record['PARTICIPANT_GENDER'] = participant.get_gender()

    utterance_sequence = []
    utterances = [utterance for utterance in participant.get_utterances()]
    for ui, utterance in enumerate(utterances):
      utterance_record = {}
      token_sequences = [token_sequence for token_sequence in utterance.get_tokens()]
      main_gloss_token_sequence = [token for token in utterance.get_tokens_for_field("main gloss")]
      utterance_main_gloss = ' '.join([token.get_text() for token in main_gloss_token_sequence])
      utterance_translation = ' '.join([token.get_text() for token in token_sequences[-1]])
      utterance_time_codes = utterance.get_timecodes()

      utterance_record['UTTERANCE_ENGLISH_TRANSLATION'] = utterance_translation
      utterance_record['UTTERANCE_START_TIME'] = utterance_time_codes[0]
      utterance_record['UTTERANCE_END_TIME'] = utterance_time_codes[1]

      media_sequence = []
      for media in utterance.get_media():
        media_record = {}
        media_fname = str(urllib.parse.quote(media.get_filename().split(':')[-1]))
        media_record['MEDIA_FNAME'] = media_fname
        media_camera_perspective = -1 # need to look this up!
        media_record['MEDIA_CAMERA_PERSPECTIVE'] = media_camera_perspective
        media_url = "<need to look this up!>"
        media_record['MEDIA_URL'] = media_url
        media_sequence.append(media_record)
      utterance_record['MEDIA_SEQUENCE'] = media_sequence

      token_sequence = []
      for ti, token in enumerate(main_gloss_token_sequence):
        token_record = {}
        token_linguistic_text = token.get_text().encode('utf-8') # must be encoded as binary since token can have punctuation and possibly other non-alphabetic characters
        token_record['TOKEN_LINGUSTIC_TEXT'] = token_linguistic_text
        token_time_codes = token.get_timecodes()
        token_record['TOKEN_START_TIME'] = token_time_codes[0]
        token_record['TOKEN_END_TIME'] = token_time_codes[1]
        token_sequence.append(token_record)
      utterance_record['TOKEN_SEQUENCE'] = token_sequence
      utterance_sequence.append(utterance_record)

    participant_record['UTTERANCE_SEQUENCE'] = utterance_sequence
    participant_sequence.append(participant_record)

  document_record['PARTICIPANT_SEQUENCE'] = participant_sequence
  # ********** parse (XML) document with SignStream: END **********
  return document_record

def pl__5__load_full_vid_index(corpus_index_decoded_XML_pcoll):
  return (
    corpus_index_decoded_XML_pcoll
    | "Beam PL: load vid index from csv into pcoll" >> beam.Create(
        [ # one row containing dict of:
            # 1. path to video index that was previously written to storage
          {
            'vid_index_path': os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_INDEXES_ARCHIVE.split('.')[0]+'.csv')
          }
        ]
      )
    # debug
    | "Beam PL: print saved vid index path" >> beam.ParDo(PipelinePcollPrinter(msg="READ SAVED VID INDEX PATH"))
  )

def pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll):
  return (
    corpus_index_decoded_XML_pcoll
    | "Beam PL: parse signstream corpus document" >> beam.Map(parse_signstream_database)
    # the above products pcoll with rows as:
      # {
      #   'CORPUS_DOCUMENT_FILENAME': <corpus doc filename>, 

      #   'PARTICIPANT_SEQUENCE': [
      #     {
      #       'PARTICIPANT_NAME': <participant name>,
      #       'PARTICIPANT_AGE': <participant age>,
      #       'PARTICIPANT_GENDER': <participant gender>,

      #       'UTTERANCE_SEQUENCE': [
      #         {
      #           'UTTERANCE_ENGLISH_TRANSLATION': <utterance English translation>,
      #           'UTTERANCE_START_TIME': <utterance start time (time code)>,
      #           'UTTERANCE_END_TIME': <utterance end time (time code)>,

      #           'MEDIA_SEQUENCE': [
      #             {
      #               'MEDIA_FNAME': <media fname>,
      #               'MEDIA_CAMERA_PERSPECTIVE': <media camera perspective>,
      #               'MEDIA_URL': <media url>
      #             }
      #           ]

      #           'TOKEN_SEQUENCE': [
      #             {
      #               'TOKEN_LINGUSTIC_TEXT': <token linguistic text>,
      #               'TOKEN_START_TIME': <token start time (time code)>,
      #               'TOKEN_END_TIME': <token end time (time code)>,
      #             }
      #           ]
      #         }
      #       ]
      #     }
      #   ]
      # }
  )

def debug_print_signstream_db(d_corpus_index_decoded_XML_row):
  d_corpus_index_decoded_XML = d_corpus_index_decoded_XML_row[0]
  """
  d_corpus_index_decoded_XML: {'DocumentID':d_corpus_index_decoded_XML['DocumentID'],'Filename':d_corpus_index_decoded_XML['Filename'],'ss_xml_db':ss_xml_db}
  """
  ss_xml_db = d_corpus_index_decoded_XML['ss_xml_db']
  # debug
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
  return [d_corpus_index_decoded_XML_row] # passthrough


def get_ss_xml_db_media(d_parsed_ss_xml_db_row):
  d_parsed_ss_xml_db = d_parsed_ss_xml_db_row[0]
  """
  d_parsed_ss_xml_db: {
    globals.SCHEMA_COL_NAMES__VIDEO_DS[0]:  <ss_xml_db documentID>,   # documentID from corpus index
    globals.SCHEMA_COL_NAMES__VIDEO_DS[3]:  <ss_xml_db filename>,     # filename from corpus index
    'ss_xml_db':                            <ss_xml_db object>        # this is the in-memory object (already parsed via the SignStream DOM API)
  }
  """
  ss_xml_db_docid = d_parsed_ss_xml_db[globals.SCHEMA_COL_NAMES__VIDEO_DS[0]]
  ss_xml_db_fname = d_parsed_ss_xml_db[globals.SCHEMA_COL_NAMES__VIDEO_DS[3]]
  ss_xml_db = d_parsed_ss_xml_db['ss_xml_db']
  media_list = []
  for media in ss_xml_db.get_media():
    media_fname = str(urllib.parse.quote(media.get_filename().split(':')[-1])) # there may be spaces in the fname
    media_list.append(media_fname)
  return [(ss_xml_db_docid, (ss_xml_db_fname, media_list))]


def get_ss_xml_db_participants(d_parsed_ss_xml_db_row):
  d_parsed_ss_xml_db = d_parsed_ss_xml_db_row[0]
  """
  d_parsed_ss_xml_db: {
    globals.SCHEMA_COL_NAMES__VIDEO_DS[0]:  <ss_xml_db documentID>,   # documentID from corpus index
    globals.SCHEMA_COL_NAMES__VIDEO_DS[3]:  <ss_xml_db filename>,     # filename from corpus index
    'ss_xml_db':                            <ss_xml_db object>        # this is the in-memory object (already parsed via the SignStream DOM API)
  }
  """
  ss_xml_db_docid = d_parsed_ss_xml_db[globals.SCHEMA_COL_NAMES__VIDEO_DS[0]]
  ss_xml_db_fname = d_parsed_ss_xml_db[globals.SCHEMA_COL_NAMES__VIDEO_DS[3]]
  ss_xml_db = d_parsed_ss_xml_db['ss_xml_db']

  participant_object_list = []
  for participant in ss_xml_db.get_participants():
    participant_data = {
      globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[1]: participant.get_name(),
      globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[2]: participant.get_age(),
      globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[3]: participant.get_gender()
      # , 'language': participant.get_language()
    }
    print(f"\tparticipant: {participant_data}")
    participant_object_list.append(copy.deepcopy(participant))

    # print(f"\t\tutterances:")
    # utterances = [utterance for utterance in participant.get_utterances()]
    # for ui, utterance in enumerate(utterances):
    #   token_sequences = [token_sequence for token_sequence in utterance.get_tokens()]
    #   main_gloss_token_sequence = [token for token in utterance.get_tokens_for_field("main gloss")]
    #   utterance_main_gloss = ' '.join([token.get_text() for token in main_gloss_token_sequence])
    #   utterance_translation = ' '.join([token.get_text() for token in token_sequences[-1]])
    #   utterance_time_codes = utterance.get_timecodes()
    #   print(f"\t\t\t#{utterance.get_id()} (time-codes: start={utterance_time_codes[0]}, end={utterance_time_codes[1]}):")
    #   print(f"\t\t\t\tEnglish Translation: {utterance_translation}")
    #   print(f"\t\t\t\tMain Gloss (Linguistic Tokens): {utterance_main_gloss}")

  return [(ss_xml_db_docid, (ss_xml_db_fname, participant_object_list))]


def validate_preprocess_asl_consultant_id_to_participant_object_mapping(tpl_aggregated_results):
  """
    tpl_aggregated_results: (
      <participant name>, # this is the key
      list_of(<participant object>)
    )
  """
  participant_name = tpl_aggregated_results[0]
  particpant_object_list = copy.deepcopy(tpl_aggregated_results[1])
  if len(particpant_object_list) > 0:
    age = -1
    gender = ""
    not_unique = False
    for participant_object in particpant_object_list:
      _age = participant_object.get_age()
      _age_list = list(map(int, re.findall(r'\d+', _age))) # we must parse using regex since it is possible to receive age string as '42 years' for example
      _age = int(_age_list[0]) if len(_age_list)>0 else -1 # -1 indicates no age provided
      _gender = participant_object.get_gender()
      if age is None:
        age = _age
        gender = _gender
      else:
        if _age != age:
          not_unique = True
          if _age > age:
            age = _age
            print(f"***WARNING!!!*** participant {participant_name} age is not unique; assigning greatest value (most recent): {age}")
        if _gender != gender:
          not_unique = True
          s_notification = "***WARNING!!!*** participant {participant_name} gender is not unique"
          if len(_gender)>0 and len(gender)==0:
            s_notification += f"; current gender is '{gender}'; assigning first (non-empty) gender: '{_gender}'"
            gender = _gender
    return [(
      participant_name, 
      {
        globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[2]:age,
        globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[3]:gender,
        'participant_object_list': particpant_object_list
      }
    )]
  else:
    print(f"***FATAL ERROR!!!*** participant {participant_name} does not have any associated info")
    return [tpl_aggregated_results] # passthrough

class DSASLConsultantIDToParticipantObjectMappingValidator(PipelinePcollElementProcessor):
  def __init__(self):
    super(DSASLConsultantIDToParticipantObjectMappingValidator, self).__init__(
      fn_pcoll_element_processor=validate_preprocess_asl_consultant_id_to_participant_object_mapping,
      kargs=None,
      return_result=True
    )

def validate_ds_document_asl_consultant_preprocessing(tpl_combined_results_row):
  # ('Michael Schlang', {'particpant_corpus_doc_mapping': [{'DocumentID': '28'}, {'DocumentID': '32'}, {'DocumentID': '36'}, {'DocumentID': '37'}, {'DocumentID': '1'}, {'DocumentID': '2'}, {'DocumentID': '3'}, {'DocumentID': '4'}, {'DocumentID': '6'}, {'DocumentID': '9'}, {'DocumentID': '12'}, {'DocumentID': '14'}, {'DocumentID': '15'}, {'DocumentID': '17'}, {'DocumentID': '19'}, {'DocumentID': '23'}], 'participant_asl_consultant_mapping': [3]})
  participant_name = tpl_combined_results_row[0]
  d_combined_results = tpl_combined_results_row[1]
  """
  d_combined_results: 
    {
      'particpant_corpus_doc_mapping': list_of({'DocumentID': <document id (as string)>}),
      'participant_asl_consultant_mapping': list_of(<asl consultant id (as int)>)
    }
  """
  particpant_corpus_doc_mapping = d_combined_results['particpant_corpus_doc_mapping']
  participant_asl_consultant_mapping = d_combined_results['participant_asl_consultant_mapping']
  set_doc_id = set()
  if len(particpant_corpus_doc_mapping) > 0:
    for d_particpant_corpus_doc_mapping_instance in particpant_corpus_doc_mapping:
      set_doc_id.add(int(d_particpant_corpus_doc_mapping_instance[globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0]]))
  else:
    print(f"***FATAL ERROR!!!*** participant {participant_name} is not associated with any corpus documents")
  asl_consultant_id = None
  if len(participant_asl_consultant_mapping) > 0:
    asl_consultant_id = None
    not_unique = False
    for _asl_consultant_id in participant_asl_consultant_mapping:
      if asl_consultant_id is None:
        asl_consultant_id = _asl_consultant_id
      else:
        if _asl_consultant_id != asl_consultant_id:
          not_unique = True
          print(f"***FATAL ERROR!!!*** participant {participant_name} asl consultant id is not unique (in document-asl-consultant mapping)! It has asl consultant ids: {asl_consultant_id} and {_asl_consultant_id}")
          break
  else:
    print(f"***FATAL ERROR!!!*** participant {participant_name} is not assigned an asl consultant id (in document-asl-consultant mapping)")
  return [(asl_consultant_id, sorted(list(set_doc_id)))]

class DSDocumentASLConsultantPreprocessingValidator(PipelinePcollElementProcessor):
  def __init__(self):
    super(DSDocumentASLConsultantPreprocessingValidator, self).__init__(
      fn_pcoll_element_processor=validate_ds_document_asl_consultant_preprocessing,
      kargs=None,
      return_result=True
    )

def validate_preprocess_asl_consultant_to_document_participant_list_mapping(tpl_combined_results):
  """
  tpl_combined_results:
    (
      'Lana Cook', 
      {
        'particpant_corpus_doc_mapping': [{'DocumentID': '8', 'participant_object': <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object at 0x7fd3c5144610>}], 
        'participant_asl_consultant_mapping': [{'ASLConsultantID': 4}]
      }
    )
  """
  participant_name = tpl_combined_results[0]
  d_combined_results = tpl_combined_results[1]
  """
  d_combined_results: 
    {
      'particpant_corpus_doc_mapping': [{'DocumentID': '8', 'participant_object': <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object at 0x7fd3c5144610>}], 
      'participant_asl_consultant_mapping': [{'ASLConsultantID': 4}]
    }
  """
  particpant_corpus_doc_mapping = d_combined_results['particpant_corpus_doc_mapping']
  participant_asl_consultant_mapping = d_combined_results['participant_asl_consultant_mapping']
  list_document_to_participant_object_mapping = []
  if len(particpant_corpus_doc_mapping) > 0:
    for d_particpant_corpus_doc_mapping_instance in particpant_corpus_doc_mapping:
      list_document_to_participant_object_mapping.append(
        (
          int(d_particpant_corpus_doc_mapping_instance[globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0]]), # DocumentID
          copy.deepcopy(d_particpant_corpus_doc_mapping_instance['participant_object'])
        )
      )
  else:
    print(f"***FATAL ERROR!!!*** participant {participant_name} is not associated with any corpus documents")
  asl_consultant_id = None
  if len(participant_asl_consultant_mapping) > 0:
    asl_consultant_id = None
    not_unique = False
    for d_asl_consultant_id in participant_asl_consultant_mapping:
      _asl_consultant_id = d_asl_consultant_id[globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[1]]
      if asl_consultant_id is None:
        asl_consultant_id = _asl_consultant_id
      else:
        if _asl_consultant_id != asl_consultant_id:
          not_unique = True
          print(f"***FATAL ERROR!!!*** participant {participant_name} asl consultant id is not unique (in document-asl-consultant mapping)! It has asl consultant ids: {asl_consultant_id} and {_asl_consultant_id}")
          break
  else:
    print(f"***FATAL ERROR!!!*** participant {participant_name} is not assigned an asl consultant id (in document-asl-consultant mapping)")
  return [
    (
      asl_consultant_id, 
      sorted(
        list_document_to_participant_object_mapping,
        key=lambda document_to_participant_object_mapping_instance: document_to_participant_object_mapping_instance[0]
      )
    )
  ]

class ASLConsultantToDocumentParticipantListMappingValidator(PipelinePcollElementProcessor):
  def __init__(self):
    super(ASLConsultantToDocumentParticipantListMappingValidator, self).__init__(
      fn_pcoll_element_processor=validate_preprocess_asl_consultant_to_document_participant_list_mapping,
      kargs=None,
      return_result=True
    )


def create_document_asl_consultant_index_schemad_pcoll(doc_id_to_participant_object_list_mapping, asl_consultant_id_to_participant_object_mapping):
  participant_doc_mapping_pcoll = (
    doc_id_to_participant_object_list_mapping # pcoll with rows as (ss_xml_db_docid, (ss_xml_db_fname, participant_object_list))
    | "Beam PL: 'explode' associated participant object from doc_id_to_participant_object_list_mapping to participant_object_list tuple pcoll" >> beam.Map(
          lambda tpl_doc_particpant_list_row: [
            (
              participant_object.get_name(), # key
              {
                globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0]: tpl_doc_particpant_list_row[0][0],  # DocumentID
                'participant_object': copy.deepcopy(participant_object)
              }
            ) for participant_object in tpl_doc_particpant_list_row[0][1][1]
          ]
        ) # yields tuples in the form of ('Michael Schlang', {'DocumentID': '37'})
    | "Beam PL: 'explode' list_doc_id_to_participant_object_tpl to pcoll" >> beam.FlatMap(lambda list_doc_id_to_participant_object_tpl: list_doc_id_to_participant_object_tpl)
    # debug
    # | "Beam PL: print explosion of list_doc_id_to_participant_object_tpl tuple pcoll" >> beam.ParDo(PipelinePcollPrinter("\tparticipant_doc_mapping_pcoll entry"))
  )

  # from asl_consultant_id_to_participant_object_mapping, we have tuples 
  #   of the form (1, ('Norma Bowers Tourangeau', {'Age': 29, 'Gender': 'female', 'participant_object_list': particpant_object_list}))
  #   but we want to extract tuple of the form (<participant name>, {'ASLConsultantID': <asl consultant id>, 'Age': <participant max age>, 'Gender': <participant gender>})
  #   so that we can merge to final doc_id_to_participant_object_mapping
  participant_asl_consultant_id_mapping = (
    asl_consultant_id_to_participant_object_mapping
    | "Beam PL: extract/transform tuples of validated participant info (with assigned ASLConsultantID), keyed by participant name" >> beam.Map(
        lambda tpl_asl_consultant_id_to_participant_object_mapping: (
          tpl_asl_consultant_id_to_participant_object_mapping[1][0],                                                                                                    # <participant name> (key)
          {
            globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[0]: tpl_asl_consultant_id_to_participant_object_mapping[0],                                                     # ASLConsultantID
            # globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[2]: tpl_asl_consultant_id_to_participant_object_mapping[1][1][globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[2]],  # Age
            # globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[3]: tpl_asl_consultant_id_to_participant_object_mapping[1][1][globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[3]]   # Gender
          }
        )
      )
    # debug
    # | "Beam PL: print extracted/transformed asl_consultant_id_to_participant_object_mapping entries for merge" >> beam.ParDo(PipelinePcollPrinter("extracted/transformed asl_consultant_id_to_participant_object_mapping entry"))
  )

  document_id_to_asl_consultant_id_participant_mapping = (
    ({
      'particpant_corpus_doc_mapping': participant_doc_mapping_pcoll,
      'participant_asl_consultant_mapping': participant_asl_consultant_id_mapping
    })
    | "Beam PL: merge participant_doc_mapping_pcoll and participant_asl_consultant_id_mapping for document_id_to_asl_consultant_id_participant_mapping" >> beam.CoGroupByKey() # the key in this case is the participant's name
    # the above outputs rows as (for example)
    #   (
    #     'Lana Cook', 
    #     {
    #       'particpant_corpus_doc_mapping': [{'DocumentID': '8', 'participant_object': <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object at 0x7fd3c5144610>}], 
    #       'participant_asl_consultant_mapping': [{'ASLConsultantID': 4}]
    #     }
    #   )
    | "Beam PL: validate/preprocess asl-consultant-to-document-participant-list mapping for document_id_to_asl_consultant_id_participant_mapping" >> beam.ParDo(ASLConsultantToDocumentParticipantListMappingValidator()) 
    # the above outputs rows as (<asl consultant id>, [(<corpus document id>, <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object>)])
    #   but we want (<corpus document id>, [(<asl consultant id>, <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object>)]) 
    | "Beam PL: 'explode' asl-consultant-to-document-participant-list to pcoll for document_id_to_asl_consultant_id_participant_mapping" >> beam.Map(
          lambda tpl_asl_consultant_id_to_document_id_participant_object_list: [
            (
              tpl_document_id_participant_object[0],                              # corpus document id
              (
                tpl_asl_consultant_id_to_document_id_participant_object_list[0],  # asl consultant id
                copy.deepcopy(tpl_document_id_participant_object[1])              # signstreamxmlparser-refactored.analysis.signstream.dom.Participant object
              )
            ) for tpl_document_id_participant_object in tpl_asl_consultant_id_to_document_id_participant_object_list[1]
          ]
        ) # yields list of tuples in the form of (<corpus document id>, (<asl consultant id>, <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object>))
      | "Beam PL: 'explode' list_document_id_asl_consultant_participant_object_tpl to pcoll for document_id_to_asl_consultant_id_participant_mapping" >> beam.FlatMap(
          lambda list_document_id_asl_consultant_participant_object_tpl: copy.deepcopy(list_document_id_asl_consultant_participant_object_tpl)
        ) # yields pcoll with rows as (<corpus document id>, (<asl consultant id>, <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object>))
    
    # debug
    # | "Beam PL: print merged document_id_to_asl_consultant_id_participant_mapping entries" >> beam.ParDo(PipelinePcollPrinter("document_id_to_asl_consultant_id_participant_mapping entry"))
  )

  # for now we need to do it this way for the sake of thread-safety: I'm running into a deadlock
  #   question: do I really need the participant object in the mapping after it has been validated??
  document_id_to_asl_consultant_id_mapping = (
    ({
      'particpant_corpus_doc_mapping': participant_doc_mapping_pcoll,
      'participant_asl_consultant_mapping': participant_asl_consultant_id_mapping
    })
    | "Beam PL: merge participant_doc_mapping_pcoll and participant_asl_consultant_id_mapping for document_id_to_asl_consultant_id_mapping" >> beam.CoGroupByKey() # the key in this case is the participant's name
    # the above outputs rows as (for example)
    #   (
    #     'Lana Cook', 
    #     {
    #       'particpant_corpus_doc_mapping': [{'DocumentID': '8', 'participant_object': <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object at 0x7fd3c5144610>}], 
    #       'participant_asl_consultant_mapping': [{'ASLConsultantID': 4}]
    #     }
    #   )
    | "Beam PL: validate/preprocess asl-consultant-to-document-participant-list mapping for document_id_to_asl_consultant_id_mapping" >> beam.ParDo(ASLConsultantToDocumentParticipantListMappingValidator()) 
    # the above outputs rows as (<asl consultant id>, [(<corpus document id>, <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object>)])
    #   but we want (<corpus document id>, [(<asl consultant id>, <signstreamxmlparser-refactored.analysis.signstream.dom.Participant object>)]) 
    | "Beam PL: 'explode' asl-consultant-to-document-participant-list to pcoll for document_id_to_asl_consultant_id_mapping" >> beam.Map(
          lambda tpl_asl_consultant_id_to_document_id_participant_object_list: [
            (
              tpl_document_id_participant_object[0],                              # corpus document id
              tpl_asl_consultant_id_to_document_id_participant_object_list[0],    # asl consultant id
            ) for tpl_document_id_participant_object in tpl_asl_consultant_id_to_document_id_participant_object_list[1]
          ]
        ) # yields list of tuples in the form of (<corpus document id>, <asl consultant id>)
      | "Beam PL: 'explode' list_document_id_asl_consultant_id to pcoll for document_id_to_asl_consultant_id_mapping" >> beam.FlatMap(
          lambda list_document_id_asl_consultant_id_tpl: list_document_id_asl_consultant_id_tpl
        ) # yields pcoll with rows as (<corpus document id>, <asl consultant id>)
    
    # debug
    # | "Beam PL: print merged document_id_to_asl_consultant_id_mapping entries" >> beam.ParDo(PipelinePcollPrinter("document_id_to_asl_consultant_id_mapping entry"))
  )

  document_asl_consultant_index_schemad_pcoll = (
    document_id_to_asl_consultant_id_mapping
    | "Beam PL: apply schema to extracted document_asl_consultant_index_schemad_pcoll tuples" >> beam.Map(lambda x: beam.Row(
          # SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS = [
          #   'DocumentID',
          #   'ASLConsultantID'
          # ]
          DocumentID=int(x[0]),
          ASLConsultantID=int(x[1])
        )
      )
    # debug
    | "Beam PL: print document_asl_consultant_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_index_schemad_pcoll entry"))
  )

  return (
    document_id_to_asl_consultant_id_participant_mapping, 
    document_asl_consultant_index_schemad_pcoll
  )


def validate_preprocess_participant_to_asl_consultant_id(tpl_participant_info_grouped_by_name):
    """
      tpl_participant_info_grouped_by_name: (<participant name>, [(<participant age (as string)>, participant_gender)])
    """
    participant_name = tpl_participant_info_grouped_by_name[0]
    particpant_info_tpl_list = tpl_participant_info_grouped_by_name[1]
    if len(particpant_info_tpl_list) > 0:
      age = -1
      gender = ""
      not_unique = False
      for participant_info_tpl in particpant_info_tpl_list:
        _age = participant_info_tpl[0]
        _age_list = list(map(int, re.findall(r'\d+', _age))) # we must parse using regex since it is possible to receive age string as '42 years' for example
        _age = int(_age_list[0]) if len(_age_list)>0 else -1 # -1 indicates no age provided
        _gender = participant_info_tpl[1]
        if age is None:
          age = _age
          gender = _gender
        else:
          if _age != age:
            not_unique = True
            if _age > age:
              age = _age
              print(f"***WARNING!!!*** participant {participant_name} age is not unique; assigning greatest value (most recent): {age}")
          if _gender != gender:
            not_unique = True
            s_notification = "***WARNING!!!*** participant {participant_name} gender is not unique"
            if len(_gender)>0 and len(gender)==0:
              s_notification += f"; current gender is '{gender}'; assigning first (non-empty) gender: '{_gender}'"
              gender = _gender
      return [(participant_name, age, gender)]
    else:
      print(f"***FATAL ERROR!!!*** participant {participant_name} does not have any associated info")
      return [tpl_participant_info_grouped_by_name] # passthrough
      
def pl__4__create_asl_consultant_index_schemad_pcoll(ss_parsed_xmldb_pcoll):
  return (
    ss_parsed_xmldb_pcoll
    | "Beam PL: extract/transform participant records list" >> beam.Map(
      lambda d_ss_parsed_xmldb_entry: [
        (
          d_participant['PARTICIPANT_NAME'],
          (
            d_participant['PARTICIPANT_AGE'],
            d_participant['PARTICIPANT_GENDER']
          )
        ) for d_participant in d_ss_parsed_xmldb_entry['PARTICIPANT_SEQUENCE']
      ]
    )
    | "Beam PL: 'explode' participant list into pcoll of individual participant records, keyed by name" >> beam.FlatMap(lambda participant_tpl: participant_tpl)
    # debug
    # | "Beam PL: print participant record for document" >> beam.ParDo(PipelinePcollPrinter(msg="participant record"))
    | "Beam PL: group participants keyed by named" >> beam.GroupByKey()
    # the above produces tuples of the form:
    #   (<participant name>, [(<participant age (as string)>, participant_gender)])
    | "Beam PL: validate/preprocess participant_to_asl_consultant_id mapping" >> beam.FlatMap(validate_preprocess_participant_to_asl_consultant_id) # outputs (<participant name>, <participant age (most recent)>, <participant gender>)
    | "Beam PL: apply RowIndex to validated participant-name-to-participant-object mapping" >> beam.ParDo(RowIndexer(var_name_prefix="asl_consultant_id")) 
    # the above produces tuples of the form:
    #   (<asl consultant id (unique)>, (participant name>, <participant age (most recent)>, <participant gender>))
    | "Beam PL: apply schema to particpant_list pcoll" >> beam.Map(lambda tpl_asl_consultant_id_validated_participant_info: beam.Row(
        # SCHEMA_COL_NAMES__ASL_CONSULTANT_DS = [
        #   'ASLConsultantID',
        #   'Name',
        #   'Age',
        #   'Gender'
        # ]
        ASLConsultantID=int(tpl_asl_consultant_id_validated_participant_info[0]),
        Name=str(tpl_asl_consultant_id_validated_participant_info[1][0]),
        Age=int(tpl_asl_consultant_id_validated_participant_info[1][1]),                  
        Gender=str(tpl_asl_consultant_id_validated_participant_info[1][2])
      )
    )
    # debug
    # | "Beam PL: print asl_consultant_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter(msg="asl_consultant_index_schemad_pcoll entry"))
  )

def pl__5__create_document_asl_consultant_index_schemad_pcoll(ss_parsed_xmldb_pcoll, corpus_index_schemad_pcoll, asl_consultant_index_schemad_pcoll):
  document_participant_pcoll = (
    ss_parsed_xmldb_pcoll
    | "Beam PL: extract/transform document-participant records list" >> beam.Map(
      lambda d_ss_parsed_xmldb_entry: [
        (
          d_ss_parsed_xmldb_entry['CORPUS_DOCUMENT_FILENAME'], 
          d_participant['PARTICIPANT_NAME']
        ) for d_participant in d_ss_parsed_xmldb_entry['PARTICIPANT_SEQUENCE']
      ]
    )
    | "Beam PL: 'explode' document-participant list into pcoll of individual document-participant records, keyed by name" >> beam.FlatMap(lambda document_participant_tpl: document_participant_tpl)
    # debug
    # | "Beam PL: print document-participant record" >> beam.ParDo(PipelinePcollPrinter(msg="document-participant record"))
    | "Beam PL: group document-participants keyed by document filename" >> beam.GroupByKey()
    # the above produces tuples of the form:
    #   (<document filename>, [<participant name>])
    | "Beam PL: 'explode' document-participant-list into pcoll where each row has a list of (<document filename>, <participant name>)" >> beam.Map(
        lambda document_participant_list_tpl: [
          (
            document_participant_list_tpl[0],
            participant_name
          ) for participant_name in document_participant_list_tpl[1]
        ]
      ) # outputs [(<document filename>, <participant name>)]
    | "Beam PL: 'explode' row as list of (<document filename>, <participant name>) into a pcoll where each row is an individual (<document filename>, <participant name>)" >> beam.FlatMap(lambda list_document_participant: list_document_participant)
    # now we have a pcoll with rows as (<document filename>, <participant name>)
    # debug
    # | "Beam PL: print document-participant records" >> beam.ParDo(PipelinePcollPrinter(msg="document-participant entry"))
  )

  document_id_pcoll = (
    corpus_index_schemad_pcoll
    | "Beam PL: extract (<document filename>, <document id>) from corpus_index_schemad_pcoll" >> beam.Map(
        lambda corpus_index_dict: (
          corpus_index_dict['Filename'],
          corpus_index_dict['DocumentID']
        )
      )
    # | "Beam PL: print document-id-to-filename records" >> beam.ParDo(PipelinePcollPrinter(msg="document-id-to-filename entry"))
  )

  participant_name_doc_id = (
    ({
      'document_id_pcoll': document_id_pcoll,
      'document_participant_pcoll': document_participant_pcoll
    })
    | "Beam PL: merge document_id_pcoll and document_participant_pcoll" >> beam.CoGroupByKey()
    # the above produces tuples of the form:
    #   ('ncslgr10e.xml', {'document_id_pcoll': ['7'], 'document_participant_pcoll': ['Norma Bowers Tourangeau', 'Benjamin Bahan']})
    | "Beam PL: extract (<participant name>, <doc id>, <doc filename>) from merged document_id_pcoll and document_participant_pcoll" >> beam.Map(
        lambda tpl: [
          (
            participant_name,
            (
              tpl[1]['document_id_pcoll'][0],
              tpl[0]
            )
          ) for participant_name in tpl[1]['document_participant_pcoll']
        ]
      )
    | "Beam PL: 'explode' doc-id-to-participant-name lists" >> beam.FlatMap(lambda list_doc_id_to_participant_name_tpl: list_doc_id_to_participant_name_tpl)
    # debug
    # | "Beam PL: print merged document_id_pcoll and document_participant_pcoll" >> beam.ParDo(PipelinePcollPrinter(msg="merged document_id_pcoll and document_participant_pcoll entry"))
  )

  participant_name_asl_constultant_id = (
    asl_consultant_index_schemad_pcoll # Row(ASLConsultantID=0, Age=-1, Gender='female', Name='Lana Cook')
    | "Beam PL: extract (<participant name>, <asl consultant id>) from asl_consultant_index_schemad_pcoll" >> beam.Map(
        lambda asl_consultant_index_schemad_pcoll_row: (
          asl_consultant_index_schemad_pcoll_row.Name,
          asl_consultant_index_schemad_pcoll_row.ASLConsultantID
        )
      )
      # debug
    # | "Beam PL: print extracted (<participant name>, <asl consultant id>) from asl_consultant_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter(msg="extracted (<participant name>, <asl consultant id>) from asl_consultant_index_schemad_pcoll"))
  )

  document_asl_consultant_index_schemad_pcoll = (
    ({
      'participant_name_doc_id': participant_name_doc_id,
      'participant_name_asl_constultant_id': participant_name_asl_constultant_id
    })
    | "Beam PL: merge participant_name_doc_id and participant_name_asl_constultant_id" >> beam.CoGroupByKey()
    # the above produces tuples of the form:
    #   ('Norma Bowers Tourangeau', {'participant_name_doc_id': [('24', 'ncslgr10l.xml'), ('33', 'ncslgr10i.xml'), ('7', 'ncslgr10e.xml'), ('29', 'ncslgr10k.xml'), ('13', 'ncslgr10f.xml'), ('21', 'ncslgr10m.xml'), ('30', 'ncslgr10j.xml'), ('18', 'ncslgr10c.xml'), ('5', 'ncslgr10d.xml'), ('25', 'ncslgr10n.xml')], 'participant_name_asl_constultant_id': [3]})
    | "Beam PL: 'explode' participant-asl-consultant-id-doc-list into pcoll where each row has a list of (<doc id>, <asl consultant_id>)" >> beam.Map(
        lambda participant_doc_id_list_tpl: [
          (
            int(corpus_doc_id_tpl[0]),                                                # DocumentID
            corpus_doc_id_tpl[1],                                                     # (corpus document) Filename
            participant_doc_id_list_tpl[1]['participant_name_asl_constultant_id'][0], # ASLConsultantID
            participant_doc_id_list_tpl[0]                                            # <participant name>
          ) for corpus_doc_id_tpl in participant_doc_id_list_tpl[1]['participant_name_doc_id']
        ]
      ) # outputs [(<corpus doc id>, <corpus doc filename>, <asl consultant_id>, <participant name>)]
    | "Beam PL: 'explode' (<corpus doc id>, <corpus doc filename>, <asl consultant_id>, <participant name>) lists" >> beam.FlatMap(lambda list_doc_id_to_asl_consultant_id_tpl: list_doc_id_to_asl_consultant_id_tpl)
    | "Beam PL: apply schema to extracted document_asl_consultant_index_schemad_pcoll tuples" >> beam.Map(lambda document_asl_consultant_mapping_tpl: beam.Row(
          DocumentID=int(document_asl_consultant_mapping_tpl[0]),
          Filename=document_asl_consultant_mapping_tpl[1],
          ASLConsultantID=int(document_asl_consultant_mapping_tpl[2]),
          ParticipantName=document_asl_consultant_mapping_tpl[3]
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_mapping" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_mapping entry"))
  )

  return document_asl_consultant_index_schemad_pcoll

def pl__6__create_document_asl_consultant_utterance_index_schemad_pcoll(ss_parsed_xmldb_pcoll, document_asl_consultant_index_schemad_pcoll):
  corpus_document_participant_utterance_mapping = (
    ss_parsed_xmldb_pcoll
    | "Beam PL: 'explode' ss_parsed_xmldb_pcoll_row_dict 'UTTERANCE_SEQUENCE'" >> beam.Map(
        lambda ss_parsed_xmldb_pcoll_row_dict: [
          (
            (
              ss_parsed_xmldb_pcoll_row_dict['CORPUS_DOCUMENT_FILENAME'],     # <corpus document filename>
              participant_utterance_sequence[0],                              # <participant name>
            ),
            participant_utterance_sequence[1],                                # <participant utterance sequence>
          ) for participant_utterance_sequence in [
              (participant['PARTICIPANT_NAME'], participant['UTTERANCE_SEQUENCE']) for participant in ss_parsed_xmldb_pcoll_row_dict['PARTICIPANT_SEQUENCE']
            ]
        ]
      )
    | "Beam PL: 'explode' ((<participant name>, <corpus document filename>), <participant utterance sequence>) lists" >> beam.FlatMap(lambda list_participant_utterance_sequence_doc_tpl: list_participant_utterance_sequence_doc_tpl)
    | "Beam PL: 'explode' <participant utterance sequence> from ((<participant name>, <corpus document filename>), <participant utterance sequence>)" >> beam.Map(
        lambda participant_doc_utterance_sequence_tpl: [
          (
            (
              utterance_seq_id, 
              ' '.join([d_token['TOKEN_LINGUSTIC_TEXT'].decode('ascii') for d_token in participant_utterance['TOKEN_SEQUENCE']]),   # <participant utterance linguistic token sequence text>
              participant_utterance['UTTERANCE_ENGLISH_TRANSLATION'],                                                               # <participant utterance English translation>
              participant_utterance['UTTERANCE_START_TIME'],                                                                        # <participant utterance start time>
              participant_utterance['UTTERANCE_END_TIME'],                                                                          # <participant utterance end time>
            ),
            (
              participant_doc_utterance_sequence_tpl[0][0],           # <participant name>
              participant_doc_utterance_sequence_tpl[0][1],           # <corpus document filename>
            )
          ) for utterance_seq_id, participant_utterance in enumerate(participant_doc_utterance_sequence_tpl[1])
        ]
      )
    | "Beam PL: 'explode' participant_utterance_name_doc_tpl lists" >> beam.FlatMap(
        lambda list_participant_utterance_name_doc_tpl: list_participant_utterance_name_doc_tpl
      )
    | "Beam PL: transform participant_utterance_name_doc_tpl to be keyed by (<corpus document filename>, <participant name>)" >> beam.Map(
        lambda participant_utterance_name_doc_keyed_by_utterance_seq_id_tpl: (
          participant_utterance_name_doc_keyed_by_utterance_seq_id_tpl[1],          # (<corpus document filename>, <participant name>)
          (
            participant_utterance_name_doc_keyed_by_utterance_seq_id_tpl[0][0],     # <utterance seq id>
            participant_utterance_name_doc_keyed_by_utterance_seq_id_tpl[0][1],     # <utterance linguistic token sequence text>
            participant_utterance_name_doc_keyed_by_utterance_seq_id_tpl[0][2],     # <utterance English translation>
            participant_utterance_name_doc_keyed_by_utterance_seq_id_tpl[0][3],     # <utterance start time>
            participant_utterance_name_doc_keyed_by_utterance_seq_id_tpl[0][4]      # <utterance end time>
          )
        )
      )
    # debug
    # | "Beam PL: print corpus_document_participant_utterance_mapping" >> beam.ParDo(PipelinePcollPrinter("corpus_document_participant_utterance_mapping entry"))
  )

  corpus_document_participant_doc_id_asl_consultant_id_mapping = (
    document_asl_consultant_index_schemad_pcoll
    | "Beam PL: extract ((<corpus document filename>, <participant name>), (<corpus document id>, <asl consultant id>)) from document_asl_consultant_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_index_schemad_pcoll_row: (
          (document_asl_consultant_index_schemad_pcoll_row.Filename, document_asl_consultant_index_schemad_pcoll_row.ParticipantName),
          (document_asl_consultant_index_schemad_pcoll_row.DocumentID, document_asl_consultant_index_schemad_pcoll_row.ASLConsultantID)
        )
      )
    # debug
    # | "Beam PL: print corpus_document_participant_doc_id_asl_consultant_id_mapping" >> beam.ParDo(PipelinePcollPrinter("corpus_document_participant_doc_id_asl_consultant_id_mapping entry"))
  )

  document_asl_consultant_utterance_index_schemad_pcoll = (
    ({
      'corpus_document_participant_doc_id_asl_consultant_id_mapping': corpus_document_participant_doc_id_asl_consultant_id_mapping,
      'corpus_document_participant_utterance_mapping': corpus_document_participant_utterance_mapping
    })
    | "Beam PL: merge corpus_document_participant_doc_id_asl_consultant_id_mapping and corpus_document_participant_utterance_mapping" >> beam.CoGroupByKey()
    # the above produces tuples of the form:
    #   (
    #     (<corpus doc filename>, <participant name>), 
    #     {
    #       'corpus_document_participant_doc_id_asl_consultant_id_mapping': [(<corpus doc id>, <asl consultant id>)], # note that this list should always only have a single tuple
    #  
    #       'corpus_document_participant_utterance_mapping': [
    #         (<utterance seq id>, <utterance linguistic token sequence text>, <utterance English translation>, <utterance start time>, <utterance end time>) # there are many of these
    #       ]
    #     }
    #   )
    | "Beam PL: 'explode' corpus_document_participant_utterance_mapping list from merge result" >> beam.Map(
        lambda merged_mapping_tpl: [
          (
            merged_mapping_tpl[0],
            (
              merged_mapping_tpl[1]['corpus_document_participant_doc_id_asl_consultant_id_mapping'][0],
              corpus_document_participant_utterance_mapping
            ), 
          ) for corpus_document_participant_utterance_mapping in merged_mapping_tpl[1]['corpus_document_participant_utterance_mapping']
        ]
      )
    | "Beam PL: 'explode' doc_participant_utterances lists" >> beam.FlatMap(
        lambda list_doc_participant_utterances_tpl: list_doc_participant_utterances_tpl
      ) # produces tuples of the form (('football.xml', 'Michael Schlang'), ((32, 1), (54, '...it can create many things.', 189800, 191733)))
    | "Beam PL: apply schema to doc_participant_utterance rows" >> beam.Map(
        lambda doc_participant_utterances_tpl: beam.Row(
          Filename=doc_participant_utterances_tpl[0][0],
          DocumentID=int(doc_participant_utterances_tpl[1][0][0]),
          ParticipantName=doc_participant_utterances_tpl[0][1],
          ASLConsultantID=int(doc_participant_utterances_tpl[1][0][1]),
          UtteranceSequence=doc_participant_utterances_tpl[1][1][0],
          StartTime=doc_participant_utterances_tpl[1][1][3],
          EndTime=doc_participant_utterances_tpl[1][1][4],
          Tokens=doc_participant_utterances_tpl[1][1][1],
          Translation=doc_participant_utterances_tpl[1][1][2]
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_utterance_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_utterance_index_schemad_pcoll entry"))
  )

  return document_asl_consultant_utterance_index_schemad_pcoll


def pl__5__write_asl_consultant_index_csv(asl_consultant_index_schemad_pcoll):
  return ( # asl_consultant_index_csv_path
    asl_consultant_index_schemad_pcoll
    | beam.Map(lambda asl_consultant_index_schemad_pcoll_row: row_to_string(asl_consultant_index_schemad_pcoll_row))
    | "Beam PL: write asl consultant index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.ASL_CONSULTANT_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=", ".join(globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS)
      )
    | "Beam PL: print path to asl consultant index csv" >> beam.ParDo(PipelinePcollPrinter(msg="ASL CONSULTANT INDEX CSV WRITTEN TO STORAGE"))
  )


def pl__6__write_document_asl_consultant_index_csv(document_asl_consultant_index_schemad_pcoll):
  return (
    document_asl_consultant_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS columns from document_asl_consultant_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_index_schemad_pcoll_row: beam.Row(
          # SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS = [
          #   'DocumentID',
          #   'ASLConsultantID'
          # ]
          DocumentID=int(document_asl_consultant_index_schemad_pcoll_row.DocumentID),
          ASLConsultantID=int(document_asl_consultant_index_schemad_pcoll_row.ASLConsultantID)
        )
      )
    | beam.Map(lambda document_asl_consultant_index_schemad_pcoll_row: row_to_string(document_asl_consultant_index_schemad_pcoll_row))
    | "Beam PL: write document-asl-consultant index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.DOCUMENT_ASL_CONSULTANT_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=", ".join(globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS)
      )
    | "Beam PL: print path to document-asl-consultant index csv" >> beam.ParDo(PipelinePcollPrinter(msg="DOCUMENT-ASL-CONSULTANT INDEX CSV WRITTEN TO STORAGE"))
  ) # document_asl_consultant_index_csv_path

def pl__7__write_document_asl_consultant_utterance_index_csv(document_asl_consultant_utterance_index_schemad_pcoll):
  return (
    document_asl_consultant_utterance_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__UTTERANCE_DS columns from document_asl_consultant_utterance_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_utterance_index_schemad_pcoll_row: beam.Row(
          # SCHEMA_COL_NAMES__UTTERANCE_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'UtteranceSequence',
          #   'StartTime',
          #   'EndTime',
          #   'Tokens',
          #   'Translation'
          # ]
          DocumentID=int(document_asl_consultant_utterance_index_schemad_pcoll_row.DocumentID),
          ASLConsultantID=int(document_asl_consultant_utterance_index_schemad_pcoll_row.ASLConsultantID),
          UtteranceSequence=int(document_asl_consultant_utterance_index_schemad_pcoll_row.UtteranceSequence),
          StartTime=int(document_asl_consultant_utterance_index_schemad_pcoll_row.StartTime),
          EndTime=int(document_asl_consultant_utterance_index_schemad_pcoll_row.EndTime),
          Tokens=document_asl_consultant_utterance_index_schemad_pcoll_row.Tokens,
          Translation=document_asl_consultant_utterance_index_schemad_pcoll_row.Translation
        )
      )
    | beam.Map(lambda document_asl_consultant_utterance_index_schemad_pcoll_row: row_to_string(document_asl_consultant_utterance_index_schemad_pcoll_row))
    | "Beam PL: write document-asl-consultant-utterance index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=", ".join(globals.SCHEMA_COL_NAMES__UTTERANCE_DS)
      )
    | "Beam PL: print path to document-asl-consultant-utterance index csv" >> beam.ParDo(PipelinePcollPrinter(msg="DOCUMENT-ASL-CONSULTANT-UTTERANCE INDEX CSV WRITTEN TO STORAGE"))
  ) # document_asl_consultant_utterance_index_csv_path

def pl__7__write_document_asl_consultant_video_index_csv(document_asl_consultant_video_index_schemad_pcoll):
  # beam.Row(
  #   # SCHEMA_COL_NAMES__VIDEO_DS = [
  #   #   'DocumentID',
  #   #   'ASLConsultantID',
  #   #   'CameraPerspective',
  #   #   'Filename'
  #   # ]
  #   DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
  #   DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
  #   ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
  #   ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
  #   CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][3]),                  
  #   MediaFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][2])
  # )
  return (
    document_asl_consultant_video_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__VIDEO_DS columns from document_asl_consultant_video_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_video_index_schemad_pcoll_row: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'Filename'
          # ]
          DocumentID=int(document_asl_consultant_video_index_schemad_pcoll_row.DocumentID),
          ASLConsultantID=int(document_asl_consultant_video_index_schemad_pcoll_row.ASLConsultantID),
          CameraPerspective=int(document_asl_consultant_video_index_schemad_pcoll_row.CameraPerspective),
          Filename=str(document_asl_consultant_video_index_schemad_pcoll_row.MediaFilename)
        )
      )
    | beam.Map(lambda document_asl_consultant_video_index_schemad_pcoll_row: row_to_string(document_asl_consultant_video_index_schemad_pcoll_row))
    | "Beam PL: write document-asl-consultant-video index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=", ".join(globals.SCHEMA_COL_NAMES__VIDEO_DS)
      )
    | "Beam PL: print path to document-asl-consultant-video index csv" >> beam.ParDo(PipelinePcollPrinter(msg="DOCUMENT-ASL-CONSULTANT-VIDEO INDEX CSV WRITTEN TO STORAGE"))
  ) # document_asl_consultant_video_index_csv_path


def validate_ds_video_preprocessing(tpl_combined_results_row):
  d_combined_results = tpl_combined_results_row[1]
  media_fname = tpl_combined_results_row[0]
  media_corpus_doc_mapping = d_combined_results['media_corpus_doc_mapping']
  media_camera_perspective_mapping = d_combined_results['media_camera_perspective']
  if len(media_corpus_doc_mapping) > 0:
    doc_id = None
    doc_fname = None
    not_unique = False
    for d_media_corpus_doc_mapping_instance in media_corpus_doc_mapping:
      _doc_id = d_media_corpus_doc_mapping_instance['DocumentID']
      _doc_fname = d_media_corpus_doc_mapping_instance['Filename']
      if doc_id is None:
        doc_id = _doc_id
        doc_fname = _doc_fname
      else:
        if _doc_id != doc_id:
          not_unique = True
          print(f"***WARNING!!!*** media {media_fname} document occurrence is not unique! It occurs in documents: {doc_fname} (doc id {doc_id}) and {_doc_fname} (doc id {_doc_id})")
          break
  if len(media_camera_perspective_mapping) > 0:
    camera_perspective = None
    not_unique = False
    for d_media_camera_perspectivec_mapping_instance in media_camera_perspective_mapping:
      _camera_perspective = d_media_camera_perspectivec_mapping_instance['CameraPerspective']
      if camera_perspective is None:
        camera_perspective = _camera_perspective
      else:
        if _camera_perspective != camera_perspective:
          not_unique = True
          print(f"***FATAL ERROR!!!*** media {media_fname} camera perspective not unique! It has camera perspectives: {camera_perspective} and {_camera_perspective}")
          break
  return [tpl_combined_results_row] # passthrough
class DSVideoPreprocessingValidator(PipelinePcollElementProcessor):
  def __init__(self):
    super(DSVideoPreprocessingValidator, self).__init__(
      fn_pcoll_element_processor=validate_ds_video_preprocessing,
      kargs=None,
      return_result=True
    )

def validate_preprocess_merged_media_doc_participant_camera_perspective_mapping(merged_media_doc_participant_camera_perspective_mapping_row_tpl):
  """
  merged_media_doc_participant_camera_perspective_mapping_row_tpl:
    (<media fname>, {'media_doc_participant_mapping': [(<corpus doc filename>, <participant_name>)], 'media_camera_perspective_mapping': [{'CameraPerspective': <camera perspective>}]})

  return:
    listof(
      ((<document fname>, <participant name>), (<media fname>, <camera perspective>))
    )
  """
  media_fname = merged_media_doc_participant_camera_perspective_mapping_row_tpl[0]
  media_doc_participant_mapping = merged_media_doc_participant_camera_perspective_mapping_row_tpl[1]['media_doc_participant_mapping']
  media_camera_perspective_mapping = merged_media_doc_participant_camera_perspective_mapping_row_tpl[1]['media_camera_perspective_mapping']

  validated_results = []

  # there should always only be ONE camera perspective per media file
  camera_perspective = None
  if len(media_camera_perspective_mapping) > 0:
    not_unique = False
    for d_media_camera_perspectivec_mapping_instance in media_camera_perspective_mapping:
      _camera_perspective = d_media_camera_perspectivec_mapping_instance['CameraPerspective']
      if camera_perspective is None:
        camera_perspective = _camera_perspective
      else:
        if _camera_perspective != camera_perspective:
          not_unique = True
          print(f"***FATAL ERROR!!!*** media {media_fname} camera perspective not unique! It has camera perspectives: {camera_perspective} and {_camera_perspective}")
          break
  else:
    print(f"***FATAL ERROR!!!*** media {media_fname} has no associated camera perspective!")
  
  doc_fname = None
  participant_name = None
  if len(media_doc_participant_mapping) > 0:
    not_unique = False
    for media_doc_participant_mapping_instance in media_doc_participant_mapping:
      _doc_fname = media_doc_participant_mapping_instance[0]
      _participant_name = media_doc_participant_mapping_instance[1]
      if doc_fname is None:
        doc_fname = _doc_fname
      else:
        if _doc_fname != doc_fname:
          not_unique = True
          print(f"***WARNING!!!*** media {media_fname} document occurrence is not unique! It occurs in documents: {doc_fname} and {_doc_fname}")
      if participant_name is None:
        participant_name = _participant_name
      else:
        if _participant_name != participant_name:
          not_unique = True
          print(f"***WARNING!!!*** media {media_fname} participant occurrence is not unique! It has participants: {participant_name} and {_participant_name}")
      validated_results.append(
        (
          (_doc_fname, _participant_name), 
          (media_fname, camera_perspective)
        )
      )
  else:
    print(f"***FATAL ERROR!!!*** media {media_fname} is not associated with a corpus document!")
    validated_results.append(
      (None, participant_name),
      (media_fname, camera_perspective)
    )

  return validated_results

def validate_preprocess_media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl(media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl):
  """
  media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl:
    ((<doc filename>, <participant name>), {'document_participant_with_ids_mapping': [(<doc id>, <asl consultant id>)], 'merged_media_doc_participant_camera_perspective_mapping': [(<media filename>, <camera perspective>)]})

  return:
    listof(
      ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <media filename>, <camera perspective>))
    )
  """
  doc_fname = media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl[0][0]
  if doc_fname is None or len(doc_fname)==0:
    print(f"***FATAL ERROR!!!*** media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl {media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl} has no associated corpus document filename")
    return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  participant_name = media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl[0][1]
  if participant_name is None or len(participant_name)==0:
    print(f"***FATAL ERROR!!!*** media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl {media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl} has no associated participant name")
    return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  document_participant_with_ids_mapping = media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl[1]['document_participant_with_ids_mapping']
  merged_media_doc_participant_camera_perspective_mapping = media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl[1]['merged_media_doc_participant_camera_perspective_mapping']

  validated_results = []

  # there should always only be ONE (<doc id>, <asl consultant id>) per mapping
  doc_id = None
  asl_consultant_id = None
  if len(document_participant_with_ids_mapping) > 0:
    not_unique = False
    for document_participant_with_ids_mapping_instance in document_participant_with_ids_mapping:
      _doc_id = document_participant_with_ids_mapping_instance[0]
      _asl_consultant_id = document_participant_with_ids_mapping_instance[1]

      if doc_id is None:
        doc_id = _doc_id
      else:
        if _doc_id != doc_id:
          not_unique = True
          print(f"***FATAL ERROR!!!*** document {doc_fname} doc_id is not unique! It has doc ids: {doc_id} and {_doc_id}")
          return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

      if asl_consultant_id is None:
        asl_consultant_id = _asl_consultant_id
      else:
        if _asl_consultant_id != asl_consultant_id:
          not_unique = True
          print(f"***FATAL ERROR!!!*** document {doc_fname} asl_consultant_id is not unique! It has doc ids: {asl_consultant_id} and {_asl_consultant_id}")
          return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows
  else:
    print(f"***FATAL ERROR!!!*** document {doc_fname} has no document_participant_with_ids_mapping!")
    return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows
  
  if len(merged_media_doc_participant_camera_perspective_mapping) > 0:
    not_unique = False
    for merged_media_doc_participant_camera_perspective_mapping_instance in merged_media_doc_participant_camera_perspective_mapping:
      _media_fname = merged_media_doc_participant_camera_perspective_mapping_instance[0]
      if _media_fname is None or len(_media_fname)==0:
        print(f"***FATAL ERROR!!!*** document {doc_fname} has an empty (or None) media filename")
        return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

      _camera_perspective = merged_media_doc_participant_camera_perspective_mapping_instance[1]
      if _camera_perspective is None or not isinstance(_camera_perspective, int) or _camera_perspective<0:
        print(f"***FATAL ERROR!!!*** document {doc_fname} has an invalid camera perspective: {_camera_perspective}")
        return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

      # ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <media filename>, <camera perspective>))
      validated_results.append(
        (
          (doc_id, asl_consultant_id), 
          (doc_fname, participant_name, _media_fname, _camera_perspective)
        )
      )
  else:
    print(f"***FATAL ERROR!!!*** document {doc_fname} has no merged_media_doc_participant_camera_perspective_mapping entries")
    return media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  return validated_results

def pl__6__create_document_asl_consultant_video_index_schemad_pcoll(ss_parsed_xmldb_pcoll, document_asl_consultant_index_schemad_pcoll, full_vid_index_schemad_pcoll):
  # get list of media
  #   note that we separate 'explosion' into a separate step since we will use ss_doc_media_list_pcoll again later
  media_doc_participant_mapping = (
    ss_parsed_xmldb_pcoll
    | "Beam PL: get media associated with this ss_parsed_xmldb, keyed by media filename" >> beam.Map(
        lambda ss_parsed_xmldb_pcoll_row_dict: [
          (
            str(urllib.parse.quote(media_participant_tpl[0])), # there may be spaces!
            (
              ss_parsed_xmldb_pcoll_row_dict['CORPUS_DOCUMENT_FILENAME'],
              media_participant_tpl[1]
            )
          ) for media_participant_tpl in [
              (d_media['MEDIA_FNAME'], d_participant['PARTICIPANT_NAME']) 
                for d_participant in ss_parsed_xmldb_pcoll_row_dict['PARTICIPANT_SEQUENCE']
                  for d_utterance in d_participant['UTTERANCE_SEQUENCE'] 
                    for d_media in d_utterance['MEDIA_SEQUENCE']
            ] 
        ]
      ) # outputs pcoll with each row list of (<media filename>, (<corpus doc filename>, <participant_name>))
    | "Beam PL: 'explode' list of media_doc_participant_mapping tuples" >> beam.FlatMap(lambda list_media_doc_participant_mapping_tpl: list_media_doc_participant_mapping_tpl)
    | "Beam PL: select distinct media_doc_participant_mapping tuples" >> beam.Distinct()
    # debug
    # | "Beam PL: print pl__6__create_document_asl_consultant_video_index_schemad_pcoll result" >> beam.ParDo(PipelinePcollPrinter("pl__6__create_document_asl_consultant_video_index_schemad_pcoll result"))
  )

  # now extract distinct media fnames from media_doc_participant_mapping
  media_list_pcoll = (
    media_doc_participant_mapping
    | "Beam PL: extract media fname" >> beam.Map(lambda media_doc_participant_mapping_row_tpl: media_doc_participant_mapping_row_tpl[0])
    | "Beam PL: select distinct media filenames" >> beam.Distinct()
    # debug
    # | "Beam PL: print media associated with this ss_parsed_xmldb" >> beam.ParDo(PipelinePcollPrinter("\tmedia"))
  )

  # now we need to filter from full_vid_index_schemad_pcoll for each media_fname in media_list_pcoll
    # recall, vid_index_entry is "schemad":
    #   beam.Row(
    #     filename=str(urllib.parse.quote(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),  # We MUST URL encode filenames since some of them sloppily contain spaces!
    #     video_seq_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                            
    #     perspective_cam_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                  
    #     compressed_mov_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),            # this is actually a list with ';' as delimiter)
    #     uncompressed_avi_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                     
    #     uncompressed_avi_mirror_1_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),   
    #     uncompressed_avi_mirror_2_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[6]])
    #   )
  media_camera_perspective_mapping = (
    full_vid_index_schemad_pcoll
    | "Beam PL: filter matching rows from vid index" >> beam.Filter(
        lambda vid_index_entry, matching_media_fnames: vid_index_entry.filename in matching_media_fnames,
        matching_media_fnames=beam.pvalue.AsIter(media_list_pcoll),
      )
    | "Beam PL: extract column vals from matching vid index entries for new video dataset" >> beam.Map(
        lambda matching_vid_index_entry: (
          matching_vid_index_entry.filename, # key
          {globals.SCHEMA_COL_NAMES__VIDEO_DS[2]: matching_vid_index_entry.perspective_cam_id}
        )
      )
    # debug
    # | "Beam PL: print vid index entries matching media associated with this ss_parsed_xmldb" >> beam.ParDo(PipelinePcollPrinter("\tmatching vid index media"))
  )

  # merge doc, participant, and camera perspective keyed by media filename
  merged_media_doc_participant_camera_perspective_mapping = (
    ({
      'media_doc_participant_mapping': media_doc_participant_mapping,
      'media_camera_perspective_mapping': media_camera_perspective_mapping
    })
    | "Beam PL: merge media_doc_participant_mapping and media_camera_perspective_mapping" >> beam.CoGroupByKey()
    # the above produces tuples in the form:
      # (<media fname>, {'media_doc_participant_mapping': [(<corpus doc filename>, <participant_name>)], 'media_camera_perspective_mapping': [{'CameraPerspective': <camera perspective>}]})
    | "Beam PL: validate/preprocess merged_media_doc_participant_camera_perspective_mapping" >> beam.FlatMap(validate_preprocess_merged_media_doc_participant_camera_perspective_mapping)
    # the above produces tuples in the form:
    #   ((<document fname>, <participant name>), (<media fname>, <camera perspective>))
    # debug
    # | "Beam PL: print merged media_doc_participant_mapping and media_camera_perspective_mapping" >> beam.ParDo(PipelinePcollPrinter("merged_media_doc_participant_camera_perspective_mapping entry"))
  )

  # now use document_asl_consultant_index_schemad_pcoll in order to associate doc id and asl consultant id:
  document_participant_with_ids_mapping = (
    document_asl_consultant_index_schemad_pcoll
    | "Beam PL: transform document_asl_consultant_index_schemad_pcoll rows into ((<document fname>, <participant name>), (<document id>, <asl consultant id>))" >> beam.Map(
        lambda document_asl_consultant_index_schemad_pcoll_row: (
          (document_asl_consultant_index_schemad_pcoll_row.Filename, document_asl_consultant_index_schemad_pcoll_row.ParticipantName),
          (document_asl_consultant_index_schemad_pcoll_row.DocumentID, document_asl_consultant_index_schemad_pcoll_row.ASLConsultantID)
        )
      )
  )

  # merge doc, participant, and camera perspective keyed by media filename
  document_asl_consultant_video_index_schemad_pcoll = (
    ({
      'document_participant_with_ids_mapping': document_participant_with_ids_mapping,
      'merged_media_doc_participant_camera_perspective_mapping': merged_media_doc_participant_camera_perspective_mapping
    })
    | "Beam PL: merge document_participant_with_ids_mapping and merged_media_doc_participant_camera_perspective_mapping" >> beam.CoGroupByKey()
    # the above produces tuples in the form:
      # ((<doc filename>, <participant name>), {'document_participant_with_ids_mapping': [(<doc id>, <asl consultant id>)], 'merged_media_doc_participant_camera_perspective_mapping': [(<media filename>, <camera perspective>)]})
    | "Beam PL: validate/preprocess media_doc_participant_camera_perspective_with_ids_pcoll" >> beam.FlatMap(validate_preprocess_media_doc_participant_camera_perspective_with_ids_pcoll_row_tpl)
    # the above produces tuples in the form:
    #   ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <media filename>, <camera perspective>))
    | "Beam PL: apply schema to create final document_asl_consultant_video_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_video_index_pcoll_row_tpl: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'Filename'
          # ]
          DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
          DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
          ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
          ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
          CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][3]),                  
          MediaFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][2])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_video_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_video_index_schemad_pcoll entry"))
  )

  return document_asl_consultant_video_index_schemad_pcoll


def pl__4__debug_print_signstream_db(ss_parsed_xmldb_pcoll):
  return (
    ss_parsed_xmldb_pcoll
    | "Beam PL: debug print parsed signstream xmldb" >> beam.Map(debug_print_signstream_db)
  )


def pl__3__parallel_download_videos(vid_index_schemad_pcoll, n_partitions=8):
  # ******************** DOWNLOAD VIDEOS IN PARALLEL: BEGIN ********************
  # this is just for debugging - comment out for production
  # (
  #   vid_index_schemad_pcoll
  #   | 'Count videos queued for download' >> beam.combiners.Count.Globally()
  #   | 'Print result' >> beam.Map(lambda count_pcol_element: print(f"Videos queued for download: {count_pcol_element}"))
  # )

  # this does the job but is much much slower than parallel downloads since each item is processed sequentially
  # (
  #   vid_index_schemad_pcoll
  #   | "Beam PL: gather download info for video segments" >> beam.ParDo(VideoSegmentInfoGatherer())
  #   # | "Beam PL: print download info for video segments" >> beam.ParDo(PipelinePcollPrinter())  # comment out for production
  #   | "Beam PL: download video segments" >> beam.ParDo(VideoSegmentDownloader())
  # )

  # create as many partitions as we have workers (cores for DirectRunner) available
  #   this is done so that downloads can occur in parallel
  #   we randomly assign each data item to one of the num_partitions partitions
  download_partitions = (
    vid_index_schemad_pcoll

    # Partition accepts a function that receives the number of partitions, and returns the index of the desired partition for the element. 
    # The number of partitions passed must be a positive integer, and it must return an integer in the range 0 to num_partitions-1.
    | "Beam PL: partition schemad video index for download parallelization" >> beam.Partition(
        lambda vid_index_row, num_partitions: random.randint(0,num_partitions-1), 
        # lambda vid_index_row, num_partitions: np.random.uniform(0,num_partitions), # not working yet
        n_partitions
      )
  )

  # here, we download in parallel by partition
  partition_download_results = [None for i in range(n_partitions)]
  for i, p in enumerate(download_partitions):
    p_label = f"p{i+1}"
    p_label_indented = f"\t{p_label}"

    p_dl_results = (
      p
      | f"Beam PL: {p_label} gather download info for video segments" >> beam.ParDo(VideoSegmentInfoGatherer())
      | f"Beam PL: {p_label} download video segments" >> beam.ParDo(VideoSegmentDownloader(f"{p_label_indented}")) # outputs a pcoll with each row as {'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}
    )
    partition_download_results[i] = p_dl_results

    # # note that this depends on the DAG - i.e. will not occur until p_dl_results are ready which, of course, does not occur until all videos have been downloaded
    # (
    #   p_dl_results
    #   | f"Beam PL: {p_label} count videos downloaded" >> beam.combiners.Count.Globally() 
    #   | f"Beam PL: {p_label} print videos downloaded count" >> beam.ParDo(PipelinePcollPrinter(label=p_label_indented, msg="videos downloaded/found"))
    # )

  # now merge all download results
  merged_download_results = (
    (p_dl_r for p_dl_r in partition_download_results) 
    | f"Beam PL: merge download results" >> beam.Flatten() 
  )

  return merged_download_results
  # ******************** DOWNLOAD VIDEOS IN PARALLEL: END ********************

def pl__4__parallel_extract_target_video_frames(merged_download_results, n_partitions=8):
  # ******************** EXTRACT SEGMENT-FRAMES IN PARALLEL: BEGIN ********************
  #   NOTE! THIS IS A CRUCIAL PIECE SO PAY ATTENTION TO THE FOLLOWING!!
  #   ********** --> IMPORTANT VIDEO-FRAME EXTRACTION PROCESSING INFORMATION<-- (BEGIN) **********
  #     We partitioned vid_index_schemad_pcoll so that video-SEGMENT downloads can occur independently.
  #     Downloading segments can occur independently since there is no correlation between each segment
  #       AS FAR AS DOWNLOADING IS CONCERNED.
  #
  #     However, AS FAR AS EXTRACTION IS CONCERNED, each segment is related by the target video composed
  #       of each segment.  The segment-videos themselves are ordered as they compose the final target
  #       video corresponding of ordered segment videos. For example, if a target video is composed of
  #       three segment videos, those segments occur in a certain order, as specified by the video index.
  #       Expanding upon this example, suppose target video "some_story_given_by_john_doe_0.mov", was recorded
  #       and saved in three corresponding video segments (to save space, I guess?) 
  #       "some_story_given_by_john_doe_0_1.mov", "some_story_given_by_john_doe_0_2.mov", and
  #       "some_story_given_by_john_doe_0_3.mov". Note that the trailing "0" in the TARGET VIDEO filename
  #       indicates the camera perspective... all stories are potentially filmed from multiple synchronized
  #       camera perspectives/angles - there were obvioiusly multiple synchronized video recorders used in
  #       in that case.  However, for this example, we are focusing on the target video for camera perspective 0.
  #       Anyway, as said, there are three segments which compose the target video.  THESE SEGMENT VIDEOS
  #       ARE ORDERED (in time).  THEREFORE, THE FRAMES COMPOSING EACH SEGMENT VIDEO ARE CONSEQUENTLY ORDERED
  #       (in time).  THE BOTTOM LINE IS THAT WE NOW NEED TO GROUP SEGMENT VIDEOS, KEYED BY CORRESPONDING
  #       TARGET VIDEO.  FURTHERMORE, THE COLLECTION OF SEGMENT VIDEOS FOR EACH TARGET VIDEO MUST BE ORDERED.
  #       THAT IS, WE MUST EXTRACT SEGMENT FRAMES AND SAVE THEM TO THE FILE SYSTEM WITH A FILE NAMING SCHEME
  #       THAT REFLECTS FRAME ORDER OF THE UNION OF ALL SEGMENT FRAMES.  IF WE EXTRACT THE FRAMES OF EACH
  #       ORDERED SEGMENT, THEN A SIMPLE NUMERIC INDEX AS SEGMENT-FRAME FILENAME WILL DO THE TRICK.
  #   ********** --> IMPORTANT VIDEO-FRAME EXTRACTION PROCESSING INFORMATION<-- (END) **********

  # GROUP segment videos by target video
  #   note that this depends on the DAG - i.e. will not occur until partition_download_results are ready which, of course, does not occur until all videos have been downloaded
  frame_extraction_partitions = (
    merged_download_results # pcoll with each row as {'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}
    | f"Beam PL: group extraction info for video segments by target video" >> beam.GroupBy(lambda d: d['video_fname']) # yields pcoll of rows as (video_fname, list({'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}))
    | f"Beam PL: partition target video segment info for extraction parallelization" >> beam.Partition(
        lambda vid_index_row, num_partitions: random.randint(0,num_partitions-1), 
        # lambda vid_index_row, num_partitions: np.random.uniform(0,num_partitions), # not working yet
        n_partitions
      )
  )

  partition_extraction_results = [None for i in range(n_partitions)]
  for i, p in enumerate(frame_extraction_partitions):
    p_label = f"p{i+1}"
    p_label_indented = f"\t{p_label}"

    p_extraction_results = (
      p
      | f"Beam PL: {p_label} extract frames of each segment per target video" >> beam.ParDo(SegmentFrameExtractor(f"{p_label_indented}")) # passthrough: pcoll of rows as (video_fname, n_stitched_frames, list({'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1], 'n_frames_extracted': n_frames_extracted}))
    )
    partition_extraction_results[i] = p_extraction_results

    # (
    #   p_extraction_results
    #   | f"Beam PL: {p_label} count target videos processed" >> beam.combiners.Count.Globally() 
    #   | f"Beam PL: {p_label} print target videos processed count" >> beam.ParDo(PipelinePcollPrinter(label=p_label_indented, msg="target videos processed"))
    # )
  
  merged_extraction_results = (
    (p_extraction_results for p_extraction_results in partition_extraction_results) 
    | f"Beam PL: merge extraction results" >> beam.Flatten() # outputs pcoll of rows as tpl_target_video_extraction_info: (video_fname, n_stitched_frames, list({'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1], 'n_frames_extracted': n_frames_extracted}))
    # | f"Beam PL: print merged extraction results" >> beam.ParDo(PipelinePcollPrinter(label="\t"))
  )
  _ = (
    merged_extraction_results
    | "Beam PL: apply schema to merged extraction results pcoll" >> beam.Map(lambda x: beam.Row(
          video_fname=str(x[0]),
          n_stitched_frames=int(x[1])
        ))
    # | "Beam PL: count total frames extracted" >> SqlTransform(f"SELECT SUM(n_stitched_frames) AS total_frames_extracted FROM PCOLLECTION") # this is VERY, VERY SLOW
    | "Beam PL: select n_stitched_frames" >> beam.Map(lambda extraction_results_row: extraction_results_row.n_stitched_frames) # on DirectRunner, this is literally about 100 times faster!
    | "Beam PL: count total frames extracted" >> beam.CombineGlobally(sum)
    | f"Beam PL: print total frames extracted" >> beam.ParDo(PipelinePcollPrinter(msg="TOTAL FRAMES EXTRACTED"))
  )

  return merged_extraction_results
  # ******************** EXTRACT SEGMENT-FRAMES IN PARALLEL: END ********************




def run():
  # pipeline_options = PipelineOptions(
  #   save_main_session=True,
  #   runner='DirectRunner',
  #   direct_num_workers=0,
  #   direct_running_mode='multi_threading', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
  #   streaming=False,
  # )
  options = {
    'project': 'my-project', # change
    'runner': 'DirectRunner',
    'direct_num_workers': 0, # 0 is use all available cores
    'direct_running_mode': 'multi_threading', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
    'streaming': False # set to True if data source is unbounded (e.g. GCP PubSub)
  }
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")

  n_partitions = 8 # hardcoded for now but we need to retrieve this from beam to be the number of workers

  with beam.Pipeline(options=pipeline_options) as pl:
    full_vid_index_schemad_pcoll = pl__1__bootstrap_video_index(pl)
    pl__2__write_vid_index_to_storage(full_vid_index_schemad_pcoll)

  with beam.Pipeline(options=pipeline_options) as pl:
    pl__1__bootstrap_corpus_index(pl)
  # writing the corpus index needs to be in a separate pipeline, which will execute sequentially after the download completes
  #   note that if we don't do it this way, it is HIGHLY probable that file structure will not be ready
  #   for reading yet
  with beam.Pipeline(options=pipeline_options) as pl:
    corpus_index_schemad_pcoll = pl__1__corpus_document_file_structure_to_corpus_index(pl)
    pl__2__write_corpus_index_to_storage(corpus_index_schemad_pcoll, GlobalVarValueAssigner(fn_assign_to_global=assign_to_global__raw_xml_b64_max_len))
  # debug
  # print(f"globals.CORPUS_DS_PATH={globals.CORPUS_DS_PATH}, globals.MAX_RAW_XML_B64_LEN={globals.MAX_RAW_XML_B64_LEN}")

  with beam.Pipeline(options=pipeline_options) as pl:
    full_vid_index_schemad_pcoll = pl__1__read_vid_index_csv(pl)
    corpus_index_schemad_pcoll = pl__1__read_corpus_index_csv(pl)
    corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll)
    ss_parsed_xmldb_pcoll = pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll)
    # pl__4__debug_print_signstream_db(ss_parsed_xmldb_pcoll)
    asl_consultant_index_schemad_pcoll = pl__4__create_asl_consultant_index_schemad_pcoll(ss_parsed_xmldb_pcoll)
    pl__5__write_asl_consultant_index_csv(asl_consultant_index_schemad_pcoll)
    document_asl_consultant_index_schemad_pcoll = pl__5__create_document_asl_consultant_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      corpus_index_schemad_pcoll, 
      asl_consultant_index_schemad_pcoll
    )
    pl__6__write_document_asl_consultant_index_csv(document_asl_consultant_index_schemad_pcoll)
    document_asl_consultant_utterance_index_schemad_pcoll = pl__6__create_document_asl_consultant_utterance_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      document_asl_consultant_index_schemad_pcoll
    )
    pl__7__write_document_asl_consultant_utterance_index_csv(document_asl_consultant_utterance_index_schemad_pcoll)
    document_asl_consultant_video_index_schemad_pcoll = pl__6__create_document_asl_consultant_video_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      document_asl_consultant_index_schemad_pcoll, 
      full_vid_index_schemad_pcoll
    )
    pl__7__write_document_asl_consultant_video_index_csv(document_asl_consultant_video_index_schemad_pcoll)

  # with beam.Pipeline(options=pipeline_options) as pl:
  #   full_vid_index_schemad_pcoll = pl__1__read_vid_index_csv(pl)
  #   vid_index_schemad_pcoll = pl__2__filter_vid_index(full_vid_index_schemad_pcoll)
  #   merged_download_results = pl__3__parallel_download_videos(vid_index_schemad_pcoll, n_partitions)
  #   merged_extraction_results = pl__4__parallel_extract_target_video_frames(merged_download_results, n_partitions)

  # TO DO: use results bootstraps to populate datasets (using extractor to globals classes)
    
  print(f"Beam PL: ALL DONE!")
  # df_video_index = vid_index_df_converter.df_video_index # this doesn't work since it's not thread-safe!
  df_video_index = None
