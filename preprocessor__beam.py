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

from importlib import import_module
sxa = import_module('.analysis', 'signstreamxmlparser-refactored')
ss = import_module('.signstream', 'signstreamxmlparser-refactored.analysis')
import cv2
import time
import base64
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


# class SignstreamCorpusBootsrapper(beam.DoFn):
#   def __init__(self, label=""):
#     self.label = label

#   def process(self, d_corpus_info):
#     return boostrap_signstream_corpus(d_corpus_info, self.label)


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


def get_video_segment_download_info(vid_index_schemad_pcoll_row):
  """
  vid_index_schemad_pcoll_row:
    beam.Row(
      filename=str(urllib.parse.quote(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),
      video_seq_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                            
      perspective_cam_id=int(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                  
      compressed_mov_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),
      uncompressed_avi_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                     
      uncompressed_avi_mirror_1_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),   
      uncompressed_avi_mirror_2_url=str(x[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[6]])
    )
  """
  video_fname = vid_index_schemad_pcoll_row.filename
  frames_dir = os.path.join(globals.STICHED_VIDEO_FRAMES_DIR, video_fname.split('.')[0])
  urls = vid_index_schemad_pcoll_row.compressed_mov_url.split(';') # this can be a list, separated by ';'
  return [{'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]} for url in urls]


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
          print(f"{label+': ' if len(label)>0 else ''}{globals.VALIDATION_WARNING_TEXT} Cannot stitch together target video {video_fname} since {_n_frames_expected} frames were expected from segment {seg_fname} ({seg_path}) but only {n_frames} were successfully extracted")
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
    print(f"\t{globals.VALIDATION_WARNING_TEXT} Cannot stitch together target video {video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segments have zero frames")
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
  return ", ". join([str(d_row[k]).replace(',','') for k in d_row.keys()])


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
  ) # full_vid_index_schemad_pcoll
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
        header=",".join(globals.SCHEMA_COL_NAMES__VIDEO_INDEX)
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
        header=",".join(globals.SCHEMA_COL_NAMES__CORPUS_DS)
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


def validate_preprocess_participant_to_asl_consultant_id(tpl_participant_info_grouped_by_name):
    """
      tpl_participant_info_grouped_by_name: (<participant name>, [(<participant age (as string)>, participant_gender)])
    """
    participant_name = tpl_participant_info_grouped_by_name[0]
    particpant_info_tpl_list = tpl_participant_info_grouped_by_name[1]
    if len(particpant_info_tpl_list) > 0:
      age = -1
      multiple_ages = []
      gender = ""
      multiple_genders = []

      for participant_info_tpl in particpant_info_tpl_list:
        _age = participant_info_tpl[0]
        _age_list = list(map(int, re.findall(r'\d+', _age))) # we must parse using regex since it is possible to receive age string as '42 years' for example
        _age = int(_age_list[0]) if len(_age_list)>0 else -1 # -1 indicates no age provided
        multiple_ages.append(_age)
        _gender = participant_info_tpl[1]
        multiple_genders.append(_gender)

      multiple_ages = set(multiple_ages)
      if len(multiple_ages) > 0:
        age = max(multiple_ages)
        if len(multiple_ages) > 1:
          print(f"{globals.VALIDATION_WARNING_TEXT} participant {participant_name} age is not unique: {multiple_ages}; assigning greatest value (most recent): {age}")
      else:
        print(f"{globals.VALIDATION_WARNING_TEXT} participant {participant_name} age info does not exist; assigning default age (-1)")
        age = -1

      multiple_genders = set(multiple_genders)
      if len(multiple_genders) > 0 and (gender is None or len(gender)==0):
        for _gender in multiple_genders:
          if len(_gender)>0:
            gender = _gender
            if len(multiple_genders) > 1:
              print(f"{globals.VALIDATION_WARNING_TEXT} participant {participant_name} gender is not unique: {multiple_genders}; current gender is {gender}; assigning first (non-empty) gender: {_gender}")              
            break

      return [(participant_name, age, gender)]
    else:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} participant {participant_name} does not have any associated info")
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


def pl__5__write_asl_consultant_index_csv(asl_consultant_index_schemad_pcoll):
  return ( # asl_consultant_index_csv_path
    asl_consultant_index_schemad_pcoll
    | beam.Map(lambda asl_consultant_index_schemad_pcoll_row: row_to_string(asl_consultant_index_schemad_pcoll_row))
    | "Beam PL: write asl consultant index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.ASL_CONSULTANT_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=",".join(globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS)
      )
    | "Beam PL: print path to asl consultant index csv" >> beam.ParDo(PipelinePcollPrinter(msg="ASL CONSULTANT INDEX CSV WRITTEN TO STORAGE"))
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


def pl__6__write_document_asl_consultant_index_csv(document_asl_consultant_index_schemad_pcoll):
  return (
    document_asl_consultant_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS columns from document_asl_consultant_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_index_schemad_pcoll_row: (
          document_asl_consultant_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_index_schemad_pcoll_row.ASLConsultantID
        )
      )
    | "Beam PL: select distinct document_asl_consultant_index rows" >> beam.Distinct()
    | "Beam PL: apply minimal schema to create final document_asl_consultant_index_schemad_pcoll of distinct rows" >> beam.Map(
        lambda distinct_document_asl_consultant_index_row: beam.Row(
          DocumentID=int(distinct_document_asl_consultant_index_row[0]),
          ASLConsultantID=int(distinct_document_asl_consultant_index_row[1])
        )
      )
    | beam.Map(lambda distinct_document_asl_consultant_index_schemad_pcoll_row: row_to_string(distinct_document_asl_consultant_index_schemad_pcoll_row))
    | "Beam PL: write document-asl-consultant index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.DOCUMENT_ASL_CONSULTANT_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=",".join(globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS)
      )
    | "Beam PL: print path to document-asl-consultant index csv" >> beam.ParDo(PipelinePcollPrinter(msg="DOCUMENT-ASL-CONSULTANT INDEX CSV WRITTEN TO STORAGE"))
  ) # document_asl_consultant_index_csv_path


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


def pl__7__write_document_asl_consultant_utterance_index_csv(document_asl_consultant_utterance_index_schemad_pcoll):
  return (
    document_asl_consultant_utterance_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__UTTERANCE_DS columns from document_asl_consultant_utterance_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_utterance_index_schemad_pcoll_row: (
          # SCHEMA_COL_NAMES__UTTERANCE_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'UtteranceSequence',
          #   'StartTime',
          #   'EndTime',
          #   'Tokens',
          #   'Translation'
          # ]
          document_asl_consultant_utterance_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_utterance_index_schemad_pcoll_row.ASLConsultantID,
          document_asl_consultant_utterance_index_schemad_pcoll_row.UtteranceSequence,
          document_asl_consultant_utterance_index_schemad_pcoll_row.StartTime,
          document_asl_consultant_utterance_index_schemad_pcoll_row.EndTime,
          document_asl_consultant_utterance_index_schemad_pcoll_row.Tokens,
          document_asl_consultant_utterance_index_schemad_pcoll_row.Translation
        )
      )
    | "Beam PL: select distinct document_asl_consultant_utterance_index rows" >> beam.Distinct()
    | "Beam PL: apply minimal schema to create final document_asl_consultant_utterance_index_schemad_pcoll of distinct rows" >> beam.Map(
        lambda distinct_document_asl_consultant_utterance_index_row: beam.Row(
          DocumentID=int(distinct_document_asl_consultant_utterance_index_row[0]),
          ASLConsultantID=int(distinct_document_asl_consultant_utterance_index_row[1]),
          UtteranceSequence=int(distinct_document_asl_consultant_utterance_index_row[2]),
          StartTime=int(distinct_document_asl_consultant_utterance_index_row[3]),
          EndTime=int(distinct_document_asl_consultant_utterance_index_row[4]),
          Tokens=distinct_document_asl_consultant_utterance_index_row[5],
          Translation=distinct_document_asl_consultant_utterance_index_row[6]
        )
      )
    | beam.Map(lambda distinct_document_asl_consultant_utterance_index_row: row_to_string(distinct_document_asl_consultant_utterance_index_row))
    | "Beam PL: write document-asl-consultant-utterance index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=",".join(globals.SCHEMA_COL_NAMES__UTTERANCE_DS)
      )
    | "Beam PL: print path to document-asl-consultant-utterance index csv" >> beam.ParDo(PipelinePcollPrinter(msg="DOCUMENT-ASL-CONSULTANT-UTTERANCE INDEX CSV WRITTEN TO STORAGE"))
  ) # document_asl_consultant_utterance_index_csv_path


def validate_preprocess_merged_corpus_doc_asl_consultant_utterance_token(merged_doc_participant_utterance_token):
  """
  merged_doc_participant_utterance_token:
    (
      (<document fname>, <participant name>),   # key
      {
        'utterance_token_mapping': [(
          <utterance seq id>, 
          <token linguistic text>, 
          <token (new) seq id>, 
          <token start time>, 
          <token end time>
        )], 
        'document_id_asl_consultant_id_mapping': [(
          <corpus doc id>, 
          <asl consultant id>
        )]
      }
    )

  return: (
    (<corpus doc id>, <asl consultant id>, <utterance seq id>, <token seq id>), # key

    # associated data (validated)
    (<document fname>, <participant name>, <token linguistic text>, <token start time>, <token end time>)
  )
  """
  doc_fname = merged_doc_participant_utterance_token[0][0]
  if len(doc_fname)==0:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid doc_fname {doc_fname}!")
    return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
  participant_name = merged_doc_participant_utterance_token[0][1]
  if len(participant_name)==0:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid participant_name {participant_name}!")
    return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped

  utterance_token_mapping = merged_doc_participant_utterance_token[1]['utterance_token_mapping']
  document_id_asl_consultant_id_mapping = merged_doc_participant_utterance_token[1]['document_id_asl_consultant_id_mapping']

  validated_results = []

  multiple_docs = []
  multiple_asl_consultants = []

  # there should always only be ONE (<corpus doc id>, <asl consultant id>) in document_id_asl_consultant_id_mapping
  doc_id = None
  asl_consultant_id = None
  if len(document_id_asl_consultant_id_mapping) > 0:
    for document_id_asl_consultant_id_mapping_instance in document_id_asl_consultant_id_mapping:
      _doc_id = document_id_asl_consultant_id_mapping_instance[0]
      if isinstance(_doc_id, int) and _doc_id>-1 and _doc_id not in multiple_docs:
        multiple_docs.append(_doc_id)
      _asl_consultant_id = document_id_asl_consultant_id_mapping_instance[1]
      if isinstance(_asl_consultant_id, int) and _asl_consultant_id>-1 and _asl_consultant_id not in multiple_asl_consultants:
        multiple_asl_consultants.append(_asl_consultant_id)
    if len(multiple_docs)>1 or len(multiple_asl_consultants)>1:
      multiple_associations = zip(multiple_docs, multiple_asl_consultants)
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} (<corpus doc id>, <asl consultant id>) association is not unique! It occurs has the following (<corpus doc id>, <asl consultant id>) associations: {multiple_associations}")
      return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
    else:
      doc_id = multiple_docs[0]
      asl_consultant_id = multiple_asl_consultants[0]
  else:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} does not have a (<corpus doc id>, <asl consultant id>) association!")
    return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
  
  if len(utterance_token_mapping) > 0:
    for utterance_token_mapping_instance in utterance_token_mapping:
      _utterance_seq_id = utterance_token_mapping_instance[0]
      if not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _utterance_seq_id {_utterance_seq_id} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_ling_text = utterance_token_mapping_instance[1]
      if len(_token_ling_text)==0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_ling_text {_token_ling_text} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_new_seq_id = utterance_token_mapping_instance[2]
      if not isinstance(_token_new_seq_id, int) or _token_new_seq_id<0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_new_seq_id {_token_new_seq_id} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_start_time = utterance_token_mapping_instance[3]
      if not isinstance(_token_start_time, int) or _token_start_time<0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_start_time {_token_start_time} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_end_time = utterance_token_mapping_instance[4]
      if not isinstance(_token_end_time, int) or _token_end_time<0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_end_time {_token_end_time} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      
      validated_results.append(
        (
          (doc_id, asl_consultant_id, _utterance_seq_id, _token_new_seq_id), 
          (doc_fname, participant_name, _token_ling_text, _token_start_time, _token_end_time)
        )
      )
  else:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} is not associated with an utterance_token_mapping!")
    return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped

  return validated_results 


def validate_preprocess_document_asl_consultant_utterance_token_tpl(document_asl_consultant_utterance_token_tpl):
  """
  document_asl_consultant_utterance_token_tpl:
    document_asl_consultant_utterance_token_index_schemad_pcoll
    (
      <token linguistic text>, 
      {
        'vocabulary_token_id_map': [<vocab token id>], 
        'doc_participant_utterance_token_info_map': [(
          <corpus doc id>, 
          <document fname>, 
          <asl consultant id>, 
          <participant name>, 
          <utterance seq id>, 
          <token (new) seq id>, 
          <token start time>, 
          <token end time>
        )]
      }
    )

  return:
    listof(
      (
        <corpus doc id>,
        <corpus document fname>,
        <asl consultant id>,
        <participant name>,
        <utterance seq id>,
        <vocab token id>,
        <token linguistic text>,
        <token (new) seq id>,
        <token start time>,
        <token end time>
      )
    )
  """
  token_ling_text = document_asl_consultant_utterance_token_tpl[0]
  if len(token_ling_text)==0:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl key is invalid: {token_ling_text}!")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped

  vocabulary_token_id_map = document_asl_consultant_utterance_token_tpl[1]['vocabulary_token_id_map']
  if len(vocabulary_token_id_map) == 0:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) does not have a <vocab token id> association!")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped

  doc_participant_utterance_token_info_map = document_asl_consultant_utterance_token_tpl[1]['doc_participant_utterance_token_info_map']
  if len(doc_participant_utterance_token_info_map) == 0:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) does not have a doc_participant_utterance_token_info_map!")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped

  validated_results = []

  multiple_token_ids = []

  vocab_token_id = None
  for vocabulary_token_id_map_instance in vocabulary_token_id_map:
    _vocab_token_id = vocabulary_token_id_map_instance
    if isinstance(_vocab_token_id, int) and _vocab_token_id>-1 and _vocab_token_id not in multiple_token_ids:
      multiple_token_ids.append(_vocab_token_id)
  if len(multiple_token_ids) > 1:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) <vocab token id> association is not unique! It occurs has the following <vocab token id> associations: {multiple_token_ids}")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
  else:
    vocab_token_id = multiple_token_ids[0]

  for doc_participant_utterance_token_info_map_instance in doc_participant_utterance_token_info_map:
    _corpus_doc_id = doc_participant_utterance_token_info_map_instance[0]
    if not isinstance(_corpus_doc_id, int) or _corpus_doc_id<0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _corpus_doc_id {_corpus_doc_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _doc_fname = doc_participant_utterance_token_info_map_instance[1]
    if len(_doc_fname)==0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _doc_fname {_doc_fname} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _asl_consultant_id = doc_participant_utterance_token_info_map_instance[2]
    if not isinstance(_asl_consultant_id, int) or _asl_consultant_id<0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _asl_consultant_id {_asl_consultant_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _participant_name = doc_participant_utterance_token_info_map_instance[3]
    if len(_participant_name)==0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _participant_name {_participant_name} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _utterance_seq_id = doc_participant_utterance_token_info_map_instance[4]
    if not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _utterance_seq_id {_utterance_seq_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _token_new_seq_id = doc_participant_utterance_token_info_map_instance[5]
    if not isinstance(_token_new_seq_id, int) or _token_new_seq_id<0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _token_new_seq_id {_token_new_seq_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _token_start_time = doc_participant_utterance_token_info_map_instance[6]
    if not isinstance(_token_start_time, int) or _token_start_time<0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _token_start_time {_token_start_time} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _token_end_time = doc_participant_utterance_token_info_map_instance[7]
    if not isinstance(_token_end_time, int) or _token_end_time<0:
      print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _token_end_time {_token_end_time} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped

    validated_results.append(
      (
        _corpus_doc_id,
        _doc_fname,
        _asl_consultant_id,
        _participant_name,
        _utterance_seq_id,
        vocab_token_id,
        token_ling_text,
        _token_new_seq_id,
        _token_start_time,
        _token_end_time
      )
    )

  return validated_results


def pl__6__create_document_asl_consultant_utterance_token_index_schemad_pcoll(ss_parsed_xmldb_pcoll, document_asl_consultant_index_schemad_pcoll):
  doc_participant_utterance_token_mapping = (
    ss_parsed_xmldb_pcoll
    | "Beam PL: get token associated with this ss_parsed_xmldb, participant, utterance, keyed by doc filename, participant name, utterance seq id, token linguistic text" >> beam.Map(
        lambda ss_parsed_xmldb_pcoll_row_dict: [
          (
            (
              doc_participant_utterance_token_tpl[0], # <corpus doc filename>
              doc_participant_utterance_token_tpl[1], # <participant name>
              doc_participant_utterance_token_tpl[2], # <utterance seq id>
              doc_participant_utterance_token_tpl[4], # <token linguistic text> (ascii representation of the byte-string)
            ), # key

            (
              doc_participant_utterance_token_tpl[3], # <token (new) seq id>
              doc_participant_utterance_token_tpl[5], # <token start time>
              doc_participant_utterance_token_tpl[6], # <token end time>
            ) # token data
          ) for doc_participant_utterance_token_tpl in [
              (
                ss_parsed_xmldb_pcoll_row_dict['CORPUS_DOCUMENT_FILENAME'],
                d_participant['PARTICIPANT_NAME'],
                utterance_seq_id,
                token_new_seq_id, 
                d_token['TOKEN_LINGUSTIC_TEXT'], # we get an ascii representation of the byte-string
                d_token['TOKEN_START_TIME'], 
                d_token['TOKEN_END_TIME']
              ) for d_participant in ss_parsed_xmldb_pcoll_row_dict['PARTICIPANT_SEQUENCE']
                  for utterance_seq_id, d_utterance in enumerate(d_participant['UTTERANCE_SEQUENCE']) 
                    for token_new_seq_id, d_token in enumerate(d_utterance['TOKEN_SEQUENCE'])
            ] 
        ]
      ) # outputs pcoll with each row list of ((<corpus doc filename>, <participant name>, <utterance seq id>, <token linguistic text>), (<token (new) seq id>, <token start time>, <token end time>))
    | "Beam PL: 'explode' list of doc_participant_utterance_token_mapping tuples" >> beam.FlatMap(lambda list_doc_participant_utterance_token_mapping_tpl: list_doc_participant_utterance_token_mapping_tpl)
    # the above produces a pcoll with rows as:
    #   ((<corpus doc filename>, <participant name>, <utterance seq id>, <token linguistic text>), (<token (new) seq id>, <token start time>, <token end time>))
    | "Beam PL: select distinct doc_participant_utterance_token_mapping tuples" >> beam.Distinct()
    # debug
    # | "Beam PL: print doc_participant_utterance_token_mapping" >> beam.ParDo(PipelinePcollPrinter("doc_participant_utterance_token_mapping entry"))
  )

  # now extract distinct token linguistic text from doc_participant_utterance_token_mapping to build the final vocabulary index
  vocabulary_index_pcoll = (
    doc_participant_utterance_token_mapping
    # ((<corpus doc filename>, <participant name>, <utterance seq id>, <token linguistic text>), (<token (new) seq id>, <token start time>, <token end time>))
    | "Beam PL: extract token linguistic text" >> beam.Map(lambda doc_participant_utterance_token_mapping_row_tpl: doc_participant_utterance_token_mapping_row_tpl[0][3])
    | "Beam PL: select distinct token linguistic text" >> beam.Distinct()
    | "Beam PL: apply RowIndex to vocab index" >> beam.ParDo(RowIndexer(var_name_prefix="vocab_token_id"))
    # the above produces tuples of the form:
      # (<vocab token id>, <vocab token linguistic text>)
    | "Beam PL: apply schema to vocabulary_index_pcoll rows" >> beam.Map(
        lambda vocabulary_index_pcoll_row_tpl: beam.Row(
          # SCHEMA_COL_NAMES__VOCABULARY_DS = [
          #   'TokenID',
          #   'Token'
          # ]
          TokenID=int(vocabulary_index_pcoll_row_tpl[0]),
          Token=vocabulary_index_pcoll_row_tpl[1]
        )
      )
    # debug
    # | "Beam PL: print vocabulary" >> beam.ParDo(PipelinePcollPrinter("vocabulary token"))
  )

  document_asl_consultant_mapping = (
    document_asl_consultant_index_schemad_pcoll
    | beam.Map(
        lambda document_asl_consultant_index_schemad_pcoll_row: (
          (document_asl_consultant_index_schemad_pcoll_row.Filename, document_asl_consultant_index_schemad_pcoll_row.ParticipantName),
          (document_asl_consultant_index_schemad_pcoll_row.DocumentID, document_asl_consultant_index_schemad_pcoll_row.ASLConsultantID)
        )
      ) # outputs rows as ((<corpus doc filename>, <participant name>), (<corpus doc id>, <asl consultant id>))
  )

  doc_participant_utterance_token_mapping_2 = (
    doc_participant_utterance_token_mapping 
    # have: ((<corpus doc filename>, <participant name>, <utterance seq id>, <token linguistic text>), (<token (new) seq id>, <token start time>, <token end time>))
    # need: ((<corpus doc filename>, <participant name>), (<utterance seq id>, <token linguistic text>, <token (new) seq id>, <token start time>, <token end time>))
    | beam.Map(
        lambda doc_participant_utterance_token_mapping_row_tpl: (
          (
            doc_participant_utterance_token_mapping_row_tpl[0][0],  # <corpus doc filename>
            doc_participant_utterance_token_mapping_row_tpl[0][1]   # <participant name>
          ),
          (
            doc_participant_utterance_token_mapping_row_tpl[0][2],   # <utterance seq id>
            doc_participant_utterance_token_mapping_row_tpl[0][3],   # <token linguistic text>
            doc_participant_utterance_token_mapping_row_tpl[1][0],   # <token (new) seq id>
            doc_participant_utterance_token_mapping_row_tpl[1][1],   # <token start time>
            doc_participant_utterance_token_mapping_row_tpl[1][2],   # <token end time>
          )
        )
      )
  )

  # merge <corpus doc id>, <asl consultant id>, <token (new) seq id>, <token start time>, <token end time>
  merged_doc_participant_utterance_token = (
    ({
      'utterance_token_mapping': doc_participant_utterance_token_mapping_2,
      'document_id_asl_consultant_id_mapping': document_asl_consultant_mapping
    })
    | "Beam PL: merge utterance_token_mapping and document_id_asl_consultant_id_mapping" >> beam.CoGroupByKey()
    # (
    #   (<document fname>, <participant name>),   # key
    #   {
    #     'utterance_token_mapping': [(
    #       <utterance seq id>, 
    #       <token linguistic text>, 
    #       <token (new) seq id>, 
    #       <token start time>, 
    #       <token end time>
    #     )], 
    #     'document_id_asl_consultant_id_mapping': [(
    #       <corpus doc id>, 
    #       <asl consultant id>
    #     )]
    #   }
    # )
    | "Beam PL: validate/preprocess merged_doc_participant_utterance_token" >> beam.FlatMap(validate_preprocess_merged_corpus_doc_asl_consultant_utterance_token)
    # the above produces tuples in the form:
    #   ((<corpus doc id>, <asl consultant id>, <utterance seq id>, <token (new) seq id>), (<document fname>, <participant name>, <token linguistic text>, <token start time>, <token end time>))
    # debug
    # | "Beam PL: print validated merged_doc_participant_utterance_token" >> beam.ParDo(PipelinePcollPrinter("merged_doc_participant_utterance_token (validated) entry"))
  )

  # transform merged_doc_participant_utterance_token tuples:
  #   have:
  #     ((<corpus doc id>, <asl consultant id>, <utterance seq id>, <token (new) seq id>), (<document fname>, <participant name>, <token linguistic text>, <token start time>, <token end time>))
  #   need:
  #     (<token linguistic text>, (<corpus doc id>, <document fname>, <asl consultant id>, <participant name>, <utterance seq id>, <token (new) seq id>, <token start time>, <token end time>))
  doc_participant_utterance_by_token_ling_text = (
    merged_doc_participant_utterance_token
    | beam.Map(
        lambda merged_doc_participant_utterance_token_row_tpl: (
          merged_doc_participant_utterance_token_row_tpl[1][2],   # <token linguistic text> (key)
          (
            merged_doc_participant_utterance_token_row_tpl[0][0], # <corpus doc id>
            merged_doc_participant_utterance_token_row_tpl[1][0], # <document fname>
            merged_doc_participant_utterance_token_row_tpl[0][1], # <asl consultant id>
            merged_doc_participant_utterance_token_row_tpl[1][1], # <participant name>
            merged_doc_participant_utterance_token_row_tpl[0][2], # <utterance seq id>
            merged_doc_participant_utterance_token_row_tpl[0][3], # <token (new) seq id>
            merged_doc_participant_utterance_token_row_tpl[1][3], # <token start time>
            merged_doc_participant_utterance_token_row_tpl[1][4], # <token end time>
          )
        )
      )
  )

  # transform vocabulary_index_pcoll tuples
    # have:
      # beam.Row(
      #   # SCHEMA_COL_NAMES__VOCABULARY_DS = [
      #   #   'TokenID',
      #   #   'Token'
      #   # ]
      #   TokenID=int(vocabulary_index_pcoll_row_tpl[0]),
      #   Token=vocabulary_index_pcoll_row_tpl[1]
      # )
    # need:
      # (<token linguistic text>, <vocab token id>)
  vocabulary_by_token_ling_text = (
    vocabulary_index_pcoll
    | beam.Map(
        lambda vocabulary_index_pcoll_row: (
          vocabulary_index_pcoll_row.Token,
          vocabulary_index_pcoll_row.TokenID
        )
      )
  )

  document_asl_consultant_utterance_token_index_schemad_pcoll = (
    ({
      'vocabulary_token_id_map': vocabulary_by_token_ling_text,
      'doc_participant_utterance_token_info_map': doc_participant_utterance_by_token_ling_text
    })
    | "Beam PL: merge vocabulary_by_token_ling_text and doc_participant_utterance_by_token_ling_text" >> beam.CoGroupByKey()
    # the above produces tuples in the form:
      # (
      #   <token linguistic text>, 
      #   {
      #     'vocabulary_token_id_map': [1057], 
      #     'doc_participant_utterance_token_info_map': [(
      #       <corpus doc id>, 
      #       <document fname>, 
      #       <asl consultant id>, 
      #       <participant name>, 
      #       <utterance seq id>, 
      #       <token (new) seq id>, 
      #       <token start time>, 
      #       <token end time>
      #     )]
      #   }
      # )
    | "Beam PL: validate/preprocess document_asl_consultant_utterance_token_tpl" >> beam.FlatMap(validate_preprocess_document_asl_consultant_utterance_token_tpl)
    # the above produces tuples in the form:
      # (
      #   <corpus doc id>,
      #   <document fname>,
      #   <asl consultant id>,
      #   <participant name>,
      #   <utterance seq id>,
      #   vocab_token_id,
      #   token_ling_text,
      #   _token_new_seq_id,
      #   _token_start_time,
      #   _token_end_time
      # )
    | "Beam PL: apply schema to document_asl_consultant_utterance_token rows" >> beam.Map(
        lambda document_asl_consultant_utterance_token_tpl: beam.Row(
          # SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'UtteranceSequence',
          #   'TokenSequence',
          #   'StartTime',
          #   'EndTime',
          #   'TokenID',
          #   'Field',
          #   'FieldValue'
          # ]
          DocumentID=document_asl_consultant_utterance_token_tpl[0],
          DocumentFilename=document_asl_consultant_utterance_token_tpl[1],
          ASLConsultantID=document_asl_consultant_utterance_token_tpl[2],
          ParticipantName=document_asl_consultant_utterance_token_tpl[3],
          UtteranceSequence=document_asl_consultant_utterance_token_tpl[4],
          TokenSequence=document_asl_consultant_utterance_token_tpl[7],
          StartTime=document_asl_consultant_utterance_token_tpl[8],
          EndTime=document_asl_consultant_utterance_token_tpl[9],
          TokenID=document_asl_consultant_utterance_token_tpl[5],
          Field='', # blank for now
          FieldValue='' # blank for now
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_utterance_token_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_utterance_token_index_schemad_pcoll entry"))
  )

  return vocabulary_index_pcoll, document_asl_consultant_utterance_token_index_schemad_pcoll


def pl__7__write_vocabulary_index_csv(vocabulary_index_pcoll):
  """
  vocabulary_index_pcoll:
    beam.Row(
      # SCHEMA_COL_NAMES__VOCABULARY_DS = [
      #   'TokenID',
      #   'Token'
      # ]
      TokenID=int(vocabulary_index_pcoll_row_tpl[0]),
      Token=vocabulary_index_pcoll_row_tpl[1]
    )
  """
  return (
    vocabulary_index_pcoll
    | beam.Map(lambda vocabulary_index_pcoll_row: row_to_string(vocabulary_index_pcoll_row))
    | "Beam PL: write vocabulary index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.VOCABULARY_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=",".join(globals.SCHEMA_COL_NAMES__VOCABULARY_DS)
      )
    | "Beam PL: print path to vocabulary index csv" >> beam.ParDo(PipelinePcollPrinter(msg="VOCABULARY INDEX CSV WRITTEN TO STORAGE"))
  ) # vocabulary_index_csv_path


def pl__7__write_document_asl_consultant_utterance_token_index_csv(document_asl_consultant_utterance_token_index_schemad_pcoll):
  """
  document_asl_consultant_utterance_token_index_schemad_pcoll:
    beam.Row(
      # SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS = [
      #   'DocumentID',
      #   'ASLConsultantID',
      #   'UtteranceSequence',
      #   'TokenSequence',
      #   'StartTime',
      #   'EndTime',
      #   'TokenID',
      #   'Field',
      #   'FieldValue'
      # ]
      DocumentID=document_asl_consultant_utterance_token_tpl[0],
      DocumentFilename=document_asl_consultant_utterance_token_tpl[1],
      ASLConsultantID=document_asl_consultant_utterance_token_tpl[2],
      ParticipantName=document_asl_consultant_utterance_token_tpl[3],
      UtteranceSequence=document_asl_consultant_utterance_token_tpl[4],
      TokenSequence=document_asl_consultant_utterance_token_tpl[7],
      StartTime=document_asl_consultant_utterance_token_tpl[8],
      EndTime=document_asl_consultant_utterance_token_tpl[9],
      TokenID=document_asl_consultant_utterance_token_tpl[5],
      Field='', # blank for now
      FieldValue='' # blank for now
    )
  """
  return (
    document_asl_consultant_utterance_token_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS columns from document_asl_consultant_utterance_token_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_utterance_token_index_schemad_pcoll_row: (
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.ASLConsultantID,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.UtteranceSequence,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.TokenSequence,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.StartTime,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.EndTime,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.TokenID,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.Field,
          document_asl_consultant_utterance_token_index_schemad_pcoll_row.FieldValue,
        )
      )
    | "Beam PL: select distinct document_asl_consultant_utterance_token_index rows" >> beam.Distinct()
    | "Beam PL: apply minimal schema to create final document_asl_consultant_utterance_token_index_schemad_pcoll of distinct rows" >> beam.Map(
        lambda distinct_document_asl_consultant_utterance_token_index_row: beam.Row(
          DocumentID=int(distinct_document_asl_consultant_utterance_token_index_row[0]),
          ASLConsultantID=int(distinct_document_asl_consultant_utterance_token_index_row[1]),
          UtteranceSequence=int(distinct_document_asl_consultant_utterance_token_index_row[2]),
          TokenSequence=int(distinct_document_asl_consultant_utterance_token_index_row[3]),
          StartTime=int(distinct_document_asl_consultant_utterance_token_index_row[4]),
          EndTime=int(distinct_document_asl_consultant_utterance_token_index_row[5]),
          TokenID=int(distinct_document_asl_consultant_utterance_token_index_row[6]),
          Field=str(distinct_document_asl_consultant_utterance_token_index_row[7]),
          FieldValue=str(distinct_document_asl_consultant_utterance_token_index_row[8]),
        )
      )
    | beam.Map(lambda distinct_document_asl_consultant_utterance_token_index_schemad_pcoll_row: row_to_string(distinct_document_asl_consultant_utterance_token_index_schemad_pcoll_row))
    | "Beam PL: write document-asl-consult-utterance-token index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.UTTERANCE_TOKEN_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=",".join(globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS)
      )
    | "Beam PL: print path to document-asl-consult-utterance-token index csv" >> beam.ParDo(PipelinePcollPrinter(msg="DOCUMENT-ASL-CONSULTANT-UTTERANCE-TOKEN INDEX CSV WRITTEN TO STORAGE"))
  ) # document_asl_consultant_utterance_token_index_csv_path


def pl__7__create_document_asl_consultant_utterance_video_index_schemad_pcoll(
  ss_parsed_xmldb_pcoll,
  document_asl_consultant_utterance_index_schemad_pcoll, 
  document_asl_consultant_video_index_schemad_pcoll
):
  pass


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
          print(f"{globals.VALIDATION_WARNING_TEXT} media {media_fname} document occurrence is not unique! It occurs in documents: {doc_fname} (doc id {doc_id}) and {_doc_fname} (doc id {_doc_id})")
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
          print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} media {media_fname} camera perspective not unique! It has camera perspectives: {camera_perspective} and {_camera_perspective}")
          break
  return [tpl_combined_results_row] # passthrough
class DSVideoPreprocessingValidator(PipelinePcollElementProcessor):
  def __init__(self):
    super(DSVideoPreprocessingValidator, self).__init__(
      fn_pcoll_element_processor=validate_ds_video_preprocessing,
      kargs=None,
      return_result=True
    )


def validate_preprocess_merged_video_doc_participant_utterance_camera_perspective_mapping(merged_video_doc_participant_utterance_camera_perspective_mapping_row_tpl):
  """
  merged_video_doc_participant_utterance_camera_perspective_mapping_row_tpl:
    (<media fname>, {'video_doc_participant_utterance_mapping': [(<corpus doc filename>, <participant_name>, <utterance seq id>)], 'video_camera_perspective_mapping': [{'CameraPerspective': <camera perspective>}]})

  return:
    listof(
      ((<document fname>, <participant name>), (<utterance seq id>, <media fname>, <camera perspective>))
    )
  """
  video_fname = merged_video_doc_participant_utterance_camera_perspective_mapping_row_tpl[0]
  video_doc_participant_utterance_mapping = merged_video_doc_participant_utterance_camera_perspective_mapping_row_tpl[1]['video_doc_participant_utterance_mapping']
  video_camera_perspective_mapping = merged_video_doc_participant_utterance_camera_perspective_mapping_row_tpl[1]['video_camera_perspective_mapping']

  validated_results = []

  # there should always only be ONE camera perspective per video_fname file
  camera_perspective = None
  if len(video_camera_perspective_mapping) > 0:
    not_unique = False
    for d_video_camera_perspectivec_mapping_instance in video_camera_perspective_mapping:
      _camera_perspective = d_video_camera_perspectivec_mapping_instance['CameraPerspective']
      if camera_perspective is None:
        camera_perspective = _camera_perspective
      else:
        if _camera_perspective != camera_perspective:
          not_unique = True
          print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} video {video_fname} camera perspective not unique! It has camera perspectives: {camera_perspective} and {_camera_perspective}")
          break
  else:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} video {video_fname} has no associated camera perspective!")
  
  doc_fname = None
  participant_name = None
  utterance_seq_id = None
  if len(video_doc_participant_utterance_mapping) > 0:
    multiple_docs = []
    multiple_participants = []
    multiple_utterances = []
    for video_doc_participant_utterance_mapping_instance in video_doc_participant_utterance_mapping:
      _doc_fname = video_doc_participant_utterance_mapping_instance[0]
      _participant_name = video_doc_participant_utterance_mapping_instance[1]
      _utterance_seq_id = video_doc_participant_utterance_mapping_instance[2]

      if doc_fname is None or len(doc_fname)==0:
        doc_fname = _doc_fname
        multiple_docs.append(_doc_fname)
      else:
        if _doc_fname != doc_fname:
          multiple_docs.append(_doc_fname)

      if participant_name is None or len(participant_name)==0:
        participant_name = _participant_name
        multiple_participants.append(_participant_name)
      else:
        if _participant_name != participant_name:
          multiple_participants.append(_participant_name)

      if utterance_seq_id is None or not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
        utterance_seq_id = _utterance_seq_id
        multiple_utterances.append(_utterance_seq_id)
      else:
        if _utterance_seq_id != utterance_seq_id:
          multiple_utterances.append(_utterance_seq_id)

      validated_results.append(
        (
          (_doc_fname, _participant_name), 
          (_utterance_seq_id, video_fname, camera_perspective)
        )
      )
    multiple_docs = set(multiple_docs)
    if len(multiple_docs) > 1:
      print(f"{globals.VALIDATION_WARNING_TEXT} video {video_fname} document occurrence is not unique! It occurs in documents: {multiple_docs}")

    multiple_participants = set(multiple_participants)
    if len(multiple_participants) > 1:
      print(f"{globals.VALIDATION_WARNING_TEXT} video {video_fname} participant occurrence is not unique! It has participants: {multiple_participants}")
    # if len(multiple_utterances) > 1: # this is actually expected
    #   print(f"{globals.VALIDATION_WARNING_TEXT}video {video_fname} utterance seq id occurrence is not unique! It has utterance seq ids: {multiple_utterances}")

  else:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} video {video_fname} is not associated with a corpus document!")
    validated_results.append(
      (None, participant_name),
      (utterance_seq_id, video_fname, camera_perspective)
    )

  return validated_results


def validate_preprocess_video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl(video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl):
  """
  video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl:
    ((<doc filename>, <participant name>), {'document_participant_with_ids_mapping': [(<doc id>, <asl consultant id>)], 'merged_video_doc_participant_utterance_camera_perspective_mapping': [(<utterance seq id>, <video fname>, <camera perspective>)]})

  return:
    listof(
      ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <utterance seq id>, <media filename>, <camera perspective>))
    )
  """
  doc_fname = video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl[0][0]
  if doc_fname is None or len(doc_fname)==0:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl {video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl} has no associated corpus document filename")
    return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  participant_name = video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl[0][1]
  if participant_name is None or len(participant_name)==0:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl {video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl} has no associated participant name")
    return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  document_participant_with_ids_mapping = video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl[1]['document_participant_with_ids_mapping']
  merged_video_doc_participant_utterance_camera_perspective_mapping = video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl[1]['merged_video_doc_participant_utterance_camera_perspective_mapping']

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
          print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} doc_id is not unique! It has doc ids: {doc_id} and {_doc_id}")
          return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

      if asl_consultant_id is None:
        asl_consultant_id = _asl_consultant_id
      else:
        if _asl_consultant_id != asl_consultant_id:
          not_unique = True
          print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} asl_consultant_id is not unique! It has doc ids: {asl_consultant_id} and {_asl_consultant_id}")
          return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows
  else:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has no document_participant_with_ids_mapping!")
    return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows
  
  if len(merged_video_doc_participant_utterance_camera_perspective_mapping) > 0:
    not_unique = False
    for merged_video_doc_participant_utterance_camera_perspective_mapping_instance in merged_video_doc_participant_utterance_camera_perspective_mapping:
      # (<utterance seq id>, <video fname>, <camera perspective>)
      _utterance_seq_id = merged_video_doc_participant_utterance_camera_perspective_mapping_instance[0]
      if _utterance_seq_id is None or not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has an invalid utterance seq id: {_utterance_seq_id}")
        return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

      _media_fname = merged_video_doc_participant_utterance_camera_perspective_mapping_instance[1]
      if _media_fname is None or len(_media_fname)==0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has an empty (or None) media filename")
        return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

      _camera_perspective = merged_video_doc_participant_utterance_camera_perspective_mapping_instance[2]
      if _camera_perspective is None or not isinstance(_camera_perspective, int) or _camera_perspective<0:
        print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has an invalid camera perspective: {_camera_perspective}")
        return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

      # ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <utterance seq id>, <media filename>, <camera perspective>))
      validated_results.append(
        (
          (doc_id, asl_consultant_id), 
          (doc_fname, participant_name, _utterance_seq_id, _media_fname, _camera_perspective)
        )
      )
  else:
    print(f"{globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has no merged_video_doc_participant_utterance_camera_perspective_mapping entries")
    return video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  return validated_results


def pl__6__create_document_asl_consultant_video_index_schemad_pcoll(ss_parsed_xmldb_pcoll, document_asl_consultant_index_schemad_pcoll, full_vid_index_schemad_pcoll):
  # get list of media
  video_doc_participant_utterance_mapping = (
    ss_parsed_xmldb_pcoll
    | "Beam PL: get doc filename, participant name, utterance seq id associated with this video, keyed by video filename" >> beam.Map(
        lambda ss_parsed_xmldb_pcoll_row_dict: [
          ( # key
            doc_participant_utterance_video_tpl[3], # <video filename>
            (
              doc_participant_utterance_video_tpl[0], # <corpus doc filename>
              doc_participant_utterance_video_tpl[1], # <participant name>
              doc_participant_utterance_video_tpl[2]  # <utterance seq id>
            )
          ) for doc_participant_utterance_video_tpl in [
              (
                ss_parsed_xmldb_pcoll_row_dict['CORPUS_DOCUMENT_FILENAME'],
                d_participant['PARTICIPANT_NAME'],
                utterance_seq_id,
                str(urllib.parse.quote(d_media['MEDIA_FNAME'])) # there may be spaces!
              ) for d_participant in ss_parsed_xmldb_pcoll_row_dict['PARTICIPANT_SEQUENCE']
                  for utterance_seq_id, d_utterance in enumerate(d_participant['UTTERANCE_SEQUENCE']) 
                    for d_media in d_utterance['MEDIA_SEQUENCE']
            ] 
        ]
      ) # outputs pcoll with each row list of (<video filename>, (<corpus doc filename>, <participant name>, <utterance seq id>))
    | "Beam PL: 'explode' list of video_doc_participant_utterance_mapping tuples" >> beam.FlatMap(lambda list_video_doc_participant_utterance_mapping_mapping_tpl: list_video_doc_participant_utterance_mapping_mapping_tpl)
    | "Beam PL: select distinct list_video_doc_participant_utterance_mapping_tpl tuples" >> beam.Distinct()
    # debug
    # | "Beam PL: print pl__6__create_document_asl_consultant_video_index_schemad_pcoll result" >> beam.ParDo(PipelinePcollPrinter("pl__6__create_document_asl_consultant_video_index_schemad_pcoll result"))
  )

  # now extract distinct media fnames from media_doc_participant_mapping
  video_list_pcoll = (
    video_doc_participant_utterance_mapping
    | "Beam PL: extract media fname" >> beam.Map(lambda video_doc_participant_utterance_mapping_row_tpl: video_doc_participant_utterance_mapping_row_tpl[0])
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
  video_camera_perspective_mapping = (
    full_vid_index_schemad_pcoll
    | "Beam PL: filter matching rows from vid index" >> beam.Filter(
        lambda vid_index_entry, matching_media_fnames: vid_index_entry.filename in matching_media_fnames,
        matching_media_fnames=beam.pvalue.AsIter(video_list_pcoll),
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
  merged_video_doc_participant_utterance_camera_perspective_mapping = (
    ({
      'video_doc_participant_utterance_mapping': video_doc_participant_utterance_mapping,
      'video_camera_perspective_mapping': video_camera_perspective_mapping
    })
    | "Beam PL: merge video_doc_participant_utterance_mapping and video_camera_perspective_mapping" >> beam.CoGroupByKey()
    # the above produces tuples in the form:
      # (<media fname>, {'video_doc_participant_utterance_mapping': [(<corpus doc filename>, <participant_name>, <utterance seq id>)], 'video_camera_perspective_mapping': [{'CameraPerspective': <camera perspective>}]})
    | "Beam PL: validate/preprocess merged_video_doc_participant_utterance_camera_perspective_mapping" >> beam.FlatMap(validate_preprocess_merged_video_doc_participant_utterance_camera_perspective_mapping)
    # the above produces tuples in the form:
    #   ((<document fname>, <participant name>), (<utterance seq id>, <video fname>, <camera perspective>))
    # debug
    # | "Beam PL: print merged video_doc_participant_utterance_mapping and video_camera_perspective_mapping" >> beam.ParDo(PipelinePcollPrinter("merged_video_doc_participant_utterance_camera_perspective_mapping entry"))
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
      'merged_video_doc_participant_utterance_camera_perspective_mapping': merged_video_doc_participant_utterance_camera_perspective_mapping
    })
    | "Beam PL: merge document_participant_with_ids_mapping and merged_video_doc_participant_utterance_camera_perspective_mapping" >> beam.CoGroupByKey()
    # the above produces tuples in the form:
      # ((<doc filename>, <participant name>), {'document_participant_with_ids_mapping': [(<doc id>, <asl consultant id>)], 'merged_video_doc_participant_utterance_camera_perspective_mapping': [(<utterance seq id>, <video fname>, <camera perspective>)]})
    | "Beam PL: validate/preprocess video_doc_participant_utterance_camera_perspective_with_ids_pcoll" >> beam.FlatMap(validate_preprocess_video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl)
    # the above produces tuples in the form:
    #   ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <utterance seq id>, <media filename>, <camera perspective>))
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
          UtteranceSequence=int(document_asl_consultant_video_index_pcoll_row_tpl[1][2]),
          CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][4]),                  
          MediaFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][3])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_video_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_video_index_schemad_pcoll entry"))
  )

  return document_asl_consultant_video_index_schemad_pcoll


def pl__7__write_document_asl_consultant_video_index_csv(document_asl_consultant_video_index_schemad_pcoll):
  """
  document_asl_consultant_video_index_schemad_pcoll:
    beam.Row(
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
  """
  return (
    document_asl_consultant_video_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__VIDEO_DS columns from document_asl_consultant_video_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_video_index_schemad_pcoll_row: (
          document_asl_consultant_video_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_video_index_schemad_pcoll_row.ASLConsultantID,
          document_asl_consultant_video_index_schemad_pcoll_row.CameraPerspective,
          document_asl_consultant_video_index_schemad_pcoll_row.MediaFilename
        )
      )
    | "Beam PL: select distinct document_asl_consultant_video_index rows" >> beam.Distinct()
    | "Beam PL: apply minimal schema to create final document_asl_consultant_video_index_schemad_pcoll of distinct rows" >> beam.Map(
        lambda distinct_document_asl_consultant_video_index_row: beam.Row(
          DocumentID=int(distinct_document_asl_consultant_video_index_row[0]),
          ASLConsultantID=int(distinct_document_asl_consultant_video_index_row[1]),
          CameraPerspective=int(distinct_document_asl_consultant_video_index_row[2]),
          Filename=str(distinct_document_asl_consultant_video_index_row[3])
        )
      )
    | beam.Map(lambda distinct_document_asl_consultant_video_index_schemad_pcoll_row: row_to_string(distinct_document_asl_consultant_video_index_schemad_pcoll_row))
    | "Beam PL: write document-asl-consultant-video index to storage as csv" >> beam.io.WriteToText(
        os.path.join(globals.DATA_ROOT_DIR, globals.VIDEO_DS_FNAME.split('.')[0]), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=",".join(globals.SCHEMA_COL_NAMES__VIDEO_DS)
      )
    | "Beam PL: print path to document-asl-consultant-video index csv" >> beam.ParDo(PipelinePcollPrinter(msg="DOCUMENT-ASL-CONSULTANT-VIDEO INDEX CSV WRITTEN TO STORAGE"))
  ) # document_asl_consultant_video_index_csv_path


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
  vid_index_schemad_pcoll_download_partitions = (
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
  for i, vid_index_schemad_pcoll_partition in enumerate(vid_index_schemad_pcoll_download_partitions):
    p_label = f"p{i+1}"
    p_label_indented = f"\t{p_label}"

    p_dl_results = (
      vid_index_schemad_pcoll_partition
      # | f"Beam PL: {p_label} gather download info for video segments" >> beam.ParDo(VideoSegmentInfoGatherer()) # get_video_segment_download_info(schemad_pcoll_element)
      | f"Beam PL: {p_label} get download info for video segments" >> beam.FlatMap(get_video_segment_download_info)
      | f"Beam PL: {p_label} download video segments" >> beam.ParDo(VideoSegmentDownloader(f"{p_label_indented}")) # outputs a pcoll with each row as [{'video_fname': video_fname, 'frames_dir': frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}]
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

    (
      p_extraction_results
      | f"Beam PL: {p_label} count target videos processed" >> beam.combiners.Count.Globally() 
      | f"Beam PL: {p_label} print target videos processed count" >> beam.ParDo(PipelinePcollPrinter(label=p_label_indented, msg="target videos processed"))
    )
  
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

    vocabulary_index_pcoll, document_asl_consultant_utterance_token_index_schemad_pcoll = pl__6__create_document_asl_consultant_utterance_token_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      document_asl_consultant_index_schemad_pcoll
    )
    pl__7__write_vocabulary_index_csv(vocabulary_index_pcoll)
    pl__7__write_document_asl_consultant_utterance_token_index_csv(document_asl_consultant_utterance_token_index_schemad_pcoll)

    pl__7__create_document_asl_consultant_utterance_video_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      document_asl_consultant_utterance_index_schemad_pcoll,
      document_asl_consultant_video_index_schemad_pcoll
    )


  with beam.Pipeline(options=pipeline_options) as pl:
    full_vid_index_schemad_pcoll = pl__1__read_vid_index_csv(pl)
    vid_index_schemad_pcoll = pl__2__filter_vid_index(full_vid_index_schemad_pcoll)
    merged_download_results = pl__3__parallel_download_videos(vid_index_schemad_pcoll, n_partitions)
    merged_extraction_results = pl__4__parallel_extract_target_video_frames(merged_download_results, n_partitions)
    

  print(f"Beam PL: ALL DONE!")
  # df_video_index = vid_index_df_converter.df_video_index # this doesn't work since it's not thread-safe!
  df_video_index = None
