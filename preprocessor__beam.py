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
    print(f"length of base-64 encoded XML document {xml_db_fname}: {len(raw_xml_b64)}")
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
  def __init__(self, label=""):
    self.label = label
    self.next_id = 0

  def process(self, element):
    tpl = (self.next_id, element)
    self.next_id += 1
    return [tpl]


def decode_XML(d_corpus_index_schemad_pcoll_row):
  """
  d_corpus_index_schemad_pcoll_row: {'DocumentID': '37', 'Filename': ' biker.xml', 'XML_B64', 'LEN'}
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
    print(f"ASSIGNED globals.MAX_RAW_XML_B64_LEN={globals.MAX_RAW_XML_B64_LEN}")
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

def pl__1__load_vid_index_csv(pl):
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
  )


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
    | "Beam PL: apply RowIndex to corpus index csv path" >> beam.ParDo(RowIndexer())
    # debug
    # | "Beam PL: print indexed path to corpus index csv" >> beam.ParDo(PipelinePcollPrinter(msg="INDEXED CORPUS INDEX CSV PATH"))
  )
  max_xml_b64_len_indexed = (
    max_xml_b64_len
    | "Beam PL: assign to global var (globals.MAX_RAW_XML_B64_LEN)" >> beam.ParDo(global_var_value_assigner__raw_xml_b64_max_len) 
    | "Beam PL: apply RowIndex to maxlen" >> beam.ParDo(RowIndexer())
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

def pl__1__load_corpus_index_csv(pl):
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
  """
  require pcollection or list of:

    d_corpus_index_decoded_XML_pcoll_row:
    {
      'DocumentID': d_corpus_index_schemad_pcoll_row['DocumentID'], 
      'Filename': d_corpus_index_schemad_pcoll_row['Filename'],
      'XML': raw_xml,
      'LEN': len(raw_xml)
    }
  """
  d_corpus_index_decoded_XML = corpus_index_decoded_XML_pcoll_row[0]
  # ********** parse (XML) document with SignStream: BEGIN **********
  print(f"length of (base-64 decoded) XML document {corpus_index_decoded_XML_pcoll_row[0]['Filename']}: {corpus_index_decoded_XML_pcoll_row[0]['LEN']}")
  in_memory_xml_doc = io.StringIO(d_corpus_index_decoded_XML['XML'])
  ss_xml_db = ss.SignStreamDatabase.read_xml(in_memory_xml_doc)
  # ********** parse (XML) document with SignStream: END **********
  return [ss_xml_db]

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

def debug_print_signstream_db(ss_parsed_xmldb_pcoll_row):
  ss_xml_db = ss_parsed_xmldb_pcoll_row[0]
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
  return [ss_xml_db] # passthrough

def append_initial_video_dataset(ss_parsed_xmldb_pcoll_row):
  ss_xml_db = ss_parsed_xmldb_pcoll_row[0]
  """
  Note that we don't have all information at this point to populate every column
    of the videos dataset.  For now, we only populate the DocumentID, CameraPerspective,
    and Filename columns.
  """
  print(f"\tmedia:")
  for media in ss_xml_db.get_media():
    # ********** populate df_video: BEGIN **********
    # video_id = append_corpus__video(media, doc_id, df_video_index, df_video, debug=debug)

    fname = str(urllib.parse.quote(media.get_filename().split(':')[-1])) # there may be spaces in the fname

    # we need access to the video index here:
    #   we would preferably have it in memory already
    #   if not, we need to load it again from the filesystem
    # df_video_index_lookup = df_video_index.query(f"{globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]}=='{fname}'")
    # _ = (

    # )

    # camera_perspective = None if df_video_index_lookup.empty else df_video_index_lookup[globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]].values[0]
    camera_perspective = None # for now
    video_id = None
    try:
      # if camera_perspective is None:
      #   if debug:
      #     print(f"\t\t{fname}\t\t*** ValueError: video '{fname}' is not in the video index, has no valid camera perspective ***")
      # else:
      #   df_video_lookup = df_video.query(f"{globals.SCHEMA_COL_NAMES__VIDEO_DS[3]}=='{fname}'")
      #   if df_video_lookup.empty:
      #     df_video.reset_index(inplace=True)
      #     data = {
      #       globals.SCHEMA_COL_NAMES__VIDEO_DS[0]: doc_id,
      #       globals.SCHEMA_COL_NAMES__VIDEO_DS[2]: camera_perspective,
      #       globals.SCHEMA_COL_NAMES__VIDEO_DS[3]: fname
      #     }
      #     video_id = len(df_video)
      #     df_video.loc[video_id] = data
      #     df_video.columns = globals.SCHEMA_COL_NAMES__VIDEO_DS
      #     df_video.set_index(globals.SCHEMA_PK__VIDEO_DS, inplace=True)
      #     df_video.sort_index(ascending=[True for c in globals.SCHEMA_PK__VIDEO_DS], inplace=True)
      #   else:
      #     # if debug:
      #     #   print(f"KeyError: video '{fname}' has already been inserted")
      #     video_id = df_video_lookup.index.values[0]
      #   if debug:
      print(f"\t\t{fname} (camera perspective {camera_perspective})")
    except Exception as e:
      print(e)
    # return video_id
    pass
    # ********** populate df_video: BEGIN **********
  return [ss_xml_db] # passthrough

def pl__4__append_initial_video_dataset(ss_parsed_xmldb_pcoll, full_vid_index_schemad_pcoll): # TO DO: make use of full_vid_index_schemad_pcoll within append_initial_video_dataset()
  return (
    ss_parsed_xmldb_pcoll
    | "Beam PL: append initial video dataset" >> beam.Map(append_initial_video_dataset)
  )

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
  print(f"globals.CORPUS_DS_PATH={globals.CORPUS_DS_PATH}, globals.MAX_RAW_XML_B64_LEN={globals.MAX_RAW_XML_B64_LEN}")

  with beam.Pipeline(options=pipeline_options) as pl:
    full_vid_index_schemad_pcoll = pl__1__load_vid_index_csv(pl)
    corpus_index_schemad_pcoll = pl__1__load_corpus_index_csv(pl)
    corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll)
    ss_parsed_xmldb_pcoll = pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll)
    # pl__4__debug_print_signstream_db(ss_parsed_xmldb_pcoll)
    pl__4__append_initial_video_dataset(ss_parsed_xmldb_pcoll, full_vid_index_schemad_pcoll)

  with beam.Pipeline(options=pipeline_options) as pl:
    full_vid_index_schemad_pcoll = pl__1__load_vid_index_csv(pl)
    vid_index_schemad_pcoll = pl__2__filter_vid_index(full_vid_index_schemad_pcoll)
    merged_download_results = pl__3__parallel_download_videos(vid_index_schemad_pcoll, n_partitions)
    merged_extraction_results = pl__4__parallel_extract_target_video_frames(merged_download_results, n_partitions)

  # TO DO: use results bootstraps to populate datasets (using extractor to globals classes)
    
  print(f"Beam PL: ALL DONE!")
  # df_video_index = vid_index_df_converter.df_video_index # this doesn't work since it's not thread-safe!
  df_video_index = None
  