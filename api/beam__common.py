from __future__ import absolute_import

import csv
import io
import logging
import sys
import urllib

import apache_beam as beam
import tensorflow as tf
from apache_beam.io.filesystems import FileSystems, GCSFileSystem
from apache_beam.options.pipeline_options import PipelineOptions
from tensorflow.python.lib.io.file_io import FileIO

from api import fidscs_globals, fileio


def opt_name_to_command_line_opt(opt_name):
  return opt_name.replace('_', '-')

def command_line_opt_to_opt_name(command_line_opt):
  return command_line_opt.replace('-', '_')


class FIDSCapstonePipelineOptions(PipelineOptions):
  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_MAX_TARGET_VIDEOS)}',
      default=None
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_WORK_DIR)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_DATA_DIR)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_TMP_DIR)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_VIDEO_DIR)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_STITCHED_VIDEO_FRAMES_DIR)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_CORPUS_DIR)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_CORPUS_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_DOCUMENT_ASL_CONSULTANT_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_ASL_CONSULTANT_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_VIDEO_INDEXES_DIR)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_SELECTED_VIDEO_INDEX_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_VIDEO_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_VIDEO_SEGMENT_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_VIDEO_FRAME_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_UTTERANCE_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_UTTERANCE_VIDEO_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_FRAME_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_VOCABULARY_DS_PATH)}',
      default=None,
    )

class FIDSCapstonePreprocessingPipelineOptions(FIDSCapstonePipelineOptions):
  @classmethod
  def _add_argparse_args(cls, parser):
    super(FIDSCapstonePreprocessingPipelineOptions, cls)._add_argparse_args(parser)
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_TRAIN_ASSOC_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_VAL_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_TRAIN_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_COMPLETE_UTTERANCES_TRAIN_ASSOC_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_COMPLETE_UTTERANCES_VAL_DS_PATH)}',
      default=None,
    )
    parser.add_argument(
      f'--{opt_name_to_command_line_opt(fidscs_globals.OPT_NAME_COMPLETE_UTTERANCES_TRAIN_DS_PATH)}',
      default=None,
    )


class PipelinePcollElementProcessor(beam.DoFn):
  def __init__(self, fn_pcoll_element_processor, kargs=None, return_result=False):
    self.fn_pcoll_element_processor = fn_pcoll_element_processor
    self.kargs = kargs
    self.return_result = return_result

  def process(self, pcoll_element):
    result = self.fn_pcoll_element_processor(pcoll_element, **self.kargs) if self.kargs is not None else self.fn_pcoll_element_processor(pcoll_element)
    ret_val = result if self.return_result else [pcoll_element]
    return ret_val


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


class GlobalVarValueAssigner(PipelinePcollElementProcessor):
  def __init__(self, fn_assign_to_global, kargs=None):
    super(GlobalVarValueAssigner, self).__init__(
      fn_pcoll_element_processor=fn_assign_to_global,
      kargs=kargs,
      return_result=True
    )


def make_fids_options_dict(work_dir, max_target_videos=-1, beam_gcp_project=fidscs_globals.GCP_PROJECT):
  data_dir = fileio.path_join(work_dir, fidscs_globals.DATA_DIR_NAME)
  tmp_dir = fileio.path_join(data_dir, fidscs_globals.TMP_DIR_NAME)
  videos_dir = fileio.path_join(data_dir, fidscs_globals.VIDEO_DIR_NAME)
  stitched_video_frames_dir = fileio.path_join(data_dir, fidscs_globals.STICHED_VIDEO_FRAMES_DIR_NAME)
  corpus_dir = fileio.path_join(tmp_dir, fidscs_globals.CORPUS_BASE)
  corpus_ds_path = fileio.path_join(data_dir, fidscs_globals.CORPUS_DS_FNAME)
  document_asl_cconsultant_ds_path = fileio.path_join(data_dir, fidscs_globals.DOCUMENT_ASL_CONSULTANT_DS_FNAME)
  asl_consultant_ds_path = fileio.path_join(data_dir, fidscs_globals.ASL_CONSULTANT_DS_FNAME)
  video_indexes_dir = fileio.path_join(tmp_dir, fidscs_globals.VIDEO_INDEX_BASE)
  selected_video_index_path = fileio.path_join(video_indexes_dir, fidscs_globals.SELECTED_VIDEO_INDEX)
  video_ds_path = fileio.path_join(data_dir, fidscs_globals.VIDEO_DS_FNAME)
  video_segment_ds_path = fileio.path_join(data_dir, fidscs_globals.VIDEO_SEGMENT_DS_FNAME)
  video_frame_ds_path = fileio.path_join(data_dir, fidscs_globals.VIDEO_FRAME_DS_FNAME)
  utterance_ds_path = fileio.path_join(data_dir, fidscs_globals.UTTERANCE_DS_FNAME)
  utterance_video_ds_path = fileio.path_join(data_dir, fidscs_globals.UTTERANCE_VIDEO_DS_FNAME)
  utterance_token_ds_path = fileio.path_join(data_dir, fidscs_globals.UTTERANCE_TOKEN_DS_FNAME)
  utterance_token_frame_ds_path = fileio.path_join(data_dir, fidscs_globals.UTTERANCE_TOKEN_FRAME_DS_FNAME)
  vocabulary_ds_path = fileio.path_join(data_dir, fidscs_globals.VOCABULARY_DS_FNAME)
  return {
    fidscs_globals.OPT_NAME_PROJECT:                          beam_gcp_project,
    fidscs_globals.OPT_NAME_MAX_TARGET_VIDEOS:                max_target_videos,
    fidscs_globals.OPT_NAME_WORK_DIR:                         work_dir,
    fidscs_globals.OPT_NAME_DATA_DIR:                         data_dir,
    fidscs_globals.OPT_NAME_TMP_DIR:                          tmp_dir,
    fidscs_globals.OPT_NAME_VIDEO_DIR:                        videos_dir,
    fidscs_globals.OPT_NAME_STITCHED_VIDEO_FRAMES_DIR:        stitched_video_frames_dir,
    fidscs_globals.OPT_NAME_CORPUS_DIR:                       corpus_dir,
    fidscs_globals.OPT_NAME_CORPUS_DS_PATH:                   corpus_ds_path,
    fidscs_globals.OPT_NAME_DOCUMENT_ASL_CONSULTANT_DS_PATH:  document_asl_cconsultant_ds_path,
    fidscs_globals.OPT_NAME_ASL_CONSULTANT_DS_PATH:           asl_consultant_ds_path,
    fidscs_globals.OPT_NAME_VIDEO_INDEXES_DIR:                video_indexes_dir,
    fidscs_globals.OPT_NAME_SELECTED_VIDEO_INDEX_PATH:        selected_video_index_path,
    fidscs_globals.OPT_NAME_VIDEO_DS_PATH:                    video_ds_path,
    fidscs_globals.OPT_NAME_VIDEO_SEGMENT_DS_PATH:            video_segment_ds_path,
    fidscs_globals.OPT_NAME_VIDEO_FRAME_DS_PATH:              video_frame_ds_path,
    fidscs_globals.OPT_NAME_UTTERANCE_DS_PATH:                utterance_ds_path,
    fidscs_globals.OPT_NAME_UTTERANCE_VIDEO_DS_PATH:          utterance_video_ds_path,
    fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_DS_PATH:          utterance_token_ds_path,
    fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_FRAME_DS_PATH:    utterance_token_frame_ds_path,
    fidscs_globals.OPT_NAME_VOCABULARY_DS_PATH:               vocabulary_ds_path
  }


def dataset_csv_files_exist(d_pl_options, dataset_csv_paths=None):
  if dataset_csv_paths is None or len(dataset_csv_paths)==0:
    dataset_csv_paths = [
      d_pl_options[fidscs_globals.OPT_NAME_ASL_CONSULTANT_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_VIDEO_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_UTTERANCE_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_UTTERANCE_VIDEO_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_CORPUS_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_DOCUMENT_ASL_CONSULTANT_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_FRAME_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_VOCABULARY_DS_PATH]
    ]
  for dataset_csv_path in dataset_csv_paths:
    if not fileio.file_path_exists(dataset_csv_path, d_pl_options)[0]:
      # print(f"Dataset {dataset_csv_path} not found")
      return False
    else:
      print(f"Found dataset {dataset_csv_path}")
  return True


def train_val_csv_files_exist(d_pl_options, train_val_csv_paths=None):
  if train_val_csv_paths is None or len(train_val_csv_paths)==0:
    train_val_csv_paths = [
      d_pl_options[fidscs_globals.OPT_NAME_TRAIN_ASSOC_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_VAL_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_TRAIN_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_COMPLETE_UTTERANCES_TRAIN_ASSOC_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_COMPLETE_UTTERANCES_VAL_DS_PATH],
      d_pl_options[fidscs_globals.OPT_NAME_COMPLETE_UTTERANCES_TRAIN_DS_PATH]
    ]
  for train_val_csv_path in train_val_csv_paths:
    if not fileio.file_path_exists(train_val_csv_path, d_pl_options)[0]:
      # print(f"Dataset {dataset_csv_path} not found")
      return False
    else:
      print(f"Found train/val dataset {train_val_csv_path}")
  return True


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
    fileio.open_file_read(sel_csv_path), 
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
  if sel_csv_path is not None and len(sel_csv_path)>0:
    return load_csv(sel_csv_path, rows_to_dicts=True, dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX)
  else:
    return []


def rmdir_if_exists(path_coll_row, d_pl_options):
  path = path_coll_row
  if fileio.path_exists(path, d_pl_options, is_dir=True):
    n_files = len(fileio.list_dir(path, d_pl_options))
    if n_files > 0 and fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
      print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} directory {path} is not empty!")
    fs = FileSystems.get_filesystem(path)
    if type(fs) == GCSFileSystem:
      path = fileio.gcs_correct_dir_path_form(path, d_pl_options)
    fileio.delete_file(path, d_pl_options, recursive=True)
    return [path]
  else:
    return []

class DirectoryDeleter(PipelinePcollElementProcessor):
  def __init__(self, d_pl_options):
    super(DirectoryDeleter, self).__init__(
      fn_pcoll_element_processor=rmdir_if_exists,
      kargs={'d_pl_options':d_pl_options},
      return_result=True
    )

def pl__X__rmdir(path, path_label, d_pl_options):
  return (
    path
    | f"Beam PL: remove {path_label}" >> beam.ParDo(DirectoryDeleter(d_pl_options))
    | f"Beam PL: print {path_label} deleted message" >> beam.ParDo(PipelinePcollPrinter(msg=f"DIRECTORY DELETED"))
  )


def sort_keyed_tuple_data(keyed_tuple):
  """
  keyed_tuple:
    (<key>, <data to be sorted (must be a list)>)
  """
  key = keyed_tuple[0]
  data_list = list(keyed_tuple[1])
  data_list.sort() # in-place
  return data_list # now sorted

def pl__X__sort_pcoll(pcoll, pcoll_label):
  return (
    pcoll
    | f"Beam PL: key {pcoll_label} for sort" >> beam.Map(
        lambda pcoll_row: (1, pcoll_row)
      )
    | f"Beam PL: 'implode' keyed {pcoll_label} for sort" >> beam.GroupByKey()
    | f"Beam PL: sort and 'explode' the 'imploded' keyed {pcoll_label}" >> beam.FlatMap(sort_keyed_tuple_data)
  )


def index_keyed_tuple_data(keyed_tuple):
  """
  keyed_tuple:
    (<key>, <data to be indexed (must be a list)>)
  """
  key = keyed_tuple[0]
  data_list = list(keyed_tuple[1])
  indexed_data_list = [(i, data_item) for i, data_item in enumerate(data_list)]
  return indexed_data_list # now indexed

def pl__X__index_pcoll(pcoll, pcoll_label):
  return (
    pcoll
    | f"Beam PL: key {pcoll_label} for index" >> beam.Map(
        lambda pcoll_row: (1, pcoll_row)
      )
    | f"Beam PL: 'implode' keyed {pcoll_label} for subset" >> beam.GroupByKey()
    | f"Beam PL: index and 'explode' the 'imploded' keyed {pcoll_label}" >> beam.FlatMap(index_keyed_tuple_data)
  )


def pl__X__subset_pcoll(pcoll, pcoll_label, n, d_pl_options):
  # if (d_pl_options['runner']=='DirectRunner' or d_pl_options['runner']=='InteractiveRunner') and \
  #   (d_pl_options['direct_running_mode']=='in_memory' or d_pl_options['direct_running_mode']=='multi_threading'):
  if d_pl_options['runner']=='DirectRunner':
    return (
      pcoll
      | beam.transforms.sql.SqlTransform(f"SELECT * FROM PCOLLECTION {'LIMIT '+str(n) if n is not None and n>0 else ''}") # we do this on DirectRunner since it will result in doing work in Docker worker nodes, which is ends up being much faster
    )
  else:
    if n is not None and n>0:
      indexed_pcoll = pl__X__index_pcoll(pcoll, pcoll_label)
      return (
        indexed_pcoll
        | f"Beam PL: filter {pcoll_label} tuples with index val less than {n}" >> beam.Filter(lambda tpl: tpl[0] < n)
        | f"Beam PL: discard {pcoll_label} index, leaving only data" >> beam.Map(lambda tpl: tpl[1])
      )
    else:
      return pcoll


def beam_row_to_csv_string(row):
  d_row = row.as_dict()
  return ", ". join([str(d_row[k]).replace(',','') for k in d_row.keys()])
  
def pl__X__write_pcoll_to_csv(pcoll, pcoll_label, csv_fname, schema_col_names, d_pl_options):
  return (
    pcoll
    | f"Beam PL: write {pcoll_label} to storage as csv" >> beam.io.WriteToText(
        fileio.path_join(
          d_pl_options[fidscs_globals.OPT_NAME_DATA_DIR],
          csv_fname.split('.')[0]
        ), 
        file_name_suffix=".csv", 
        append_trailing_newlines=True,
        shard_name_template="",
        header=",".join(schema_col_names)
      )
    | f"Beam PL: print path to {pcoll_label} csv" >> beam.ParDo(PipelinePcollPrinter(msg=f"{pcoll_label} CSV WRITTEN TO STORAGE"))
  ) # returns pcoll with single entry, the full path to the file written


def pl__1__read_target_vid_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing path to load the video index csv" >> beam.Create(
        [
          fileio.path_join(
            pl._options._all_options[fidscs_globals.OPT_NAME_DATA_DIR], 
            fidscs_globals.VIDEO_INDEXES_ARCHIVE.split('.')[0]+'.csv'
          )
        ]
      )
    | "Beam PL: read video index into pcoll" >> beam.FlatMap(load_vid_index_csv) # outputs another pcoll but with each row as dict
    | "Beam PL: apply schema to video index pcoll" >> beam.Map(lambda x: beam.Row(
          target_video_filename=str(urllib.parse.quote(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),  # We MUST URL encode filenames since some of them sloppily contain spaces!
          video_seq_id=int(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                            
          perspective_cam_id=int(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                  
          compressed_mov_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),            # this is actually a list with ';' as delimiter)
          uncompressed_avi_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                     
          uncompressed_avi_mirror_1_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),   
          uncompressed_avi_mirror_2_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[6]])
        )
      )
  ) # full_target_vid_index_schemad_pcoll


def load_corpus_index_csv(d_corpus_info):
  """
  this function simply wraps the call to load_csv() to produce a "schema'd" pcoll
  so we fix the definition of dict_field_names to fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS

  d_corpus_info: {
    'corpus_index_csv_path': fidscs_globals.CORPUS_DS_PATH,
    'max_len': fidscs_globals.MAX_RAW_XML_B64_LEN
  }
  """
  return load_csv(
    d_corpus_info['corpus_index_csv_path'],
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS,
    max_len=d_corpus_info['max_len']+4 # note that we need 4 more bytes due to base-64 encoding
  )

def pl__1__read_corpus_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the corpus index csv" >> beam.Create(
        [{
          'corpus_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_CORPUS_DS_PATH],
          'max_len': fidscs_globals.MAX_RAW_XML_B64_LEN # NOTE! This is likely running in a different pipeline (and likely a different worker altogether in the cloud, thus, we rely on a high enough default setting)
        }]
      )
    | "Beam PL: read corpus index into pcoll" >> beam.FlatMap(load_corpus_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS keys)
    | "Beam PL: apply schema to extracted corpus document info dicts" >> beam.Map(
        lambda d_corpus_document_info: beam.Row(
          # SCHEMA_COL_NAMES__CORPUS_DS = [
          #   'DocumentID',
          #   'Filename',
          #   'XML_B64',
          #   'LEN'
          # ]
          DocumentID=int(d_corpus_document_info[fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS[0]]),
          Filename=str(d_corpus_document_info[fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS[1]]),
          XML_B64=d_corpus_document_info[fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS[2]],
          LEN=int(d_corpus_document_info[fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS[3]])
        )
      )
    # debug
    # | "Beam PL: print corpus_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded corpus_index_schemad_pcoll entry"))
  ) # corpus_index_schemad_pcoll


def load_asl_consultant_index_csv(d_asl_consultant_index_info):
  return load_csv(
    d_asl_consultant_index_info['asl_consultant_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS
  )

def pl__1__read_asl_consultant_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the asl consultant index csv" >> beam.Create(
        [{
          'asl_consultant_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_ASL_CONSULTANT_DS_PATH]
        }]
      )
    # | "Beam PL: print path to asl_consultant_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("asl_consultant_index path"))
    | "Beam PL: read asl consultant index into pcoll" >> beam.FlatMap(load_asl_consultant_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS keys)
    | "Beam PL: apply schema to extracted document asl consultant index info dicts" >> beam.Map(
        lambda d_document_asl_consultant: beam.Row(
          # SCHEMA_COL_NAMES__ASL_CONSULTANT_DS = [
          #   'ASLConsultantID',
          #   'Name',
          #   'Age',
          #   'Gender'
          # ]
          ASLConsultantID=int(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[0]]),
          Name=str(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[1]]),
          Age=int(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[2]]),
          Gender=str(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS[3]])
        )
      )
    # debug
    # | "Beam PL: print asl_consultant_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded asl_consultant_index_schemad_pcoll entry"))
  ) # asl_consultant_index_schemad_pcoll


def load_document_asl_consultant_index_csv(d_document_asl_consultant_index_info):
  return load_csv(
    d_document_asl_consultant_index_info['document_asl_consultant_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS
  )

def pl__1__read_document_asl_consultant_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant index csv" >> beam.Create(
        [{
          'document_asl_consultant_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_DOCUMENT_ASL_CONSULTANT_DS_PATH]
        }]
      )
    # | "Beam PL: print path to asl_consultant_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_index path"))
    | "Beam PL: document asl consultant index into pcoll" >> beam.FlatMap(load_document_asl_consultant_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS keys)
    | "Beam PL: apply schema to extracted document asl consultant index info dicts" >> beam.Map(
        lambda d_document_asl_consultant: beam.Row(
          # SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'Filename',
          #   'ParticipantName'
          # ]
          DocumentID=int(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[0]]),
          ASLConsultantID=int(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[1]]),
          Filename=str(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[2]]),
          ParticipantName=str(d_document_asl_consultant[fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS[3]])
        )
      )
    # debug
    # | "Beam PL: print asl_consultant_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded asl_consultant_index_schemad_pcoll entry"))
  ) # document_asl_consultant_index_schemad_pcoll


def load_document_asl_consultant_utterance_index_csv(d_document_asl_consultant_utterance_index_info):
  return load_csv(
    d_document_asl_consultant_utterance_index_info['document_asl_consultant_utterance_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS
  )

def pl__1__read_document_asl_consultant_utterance_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant utterance index csv" >> beam.Create(
        [{
          'document_asl_consultant_utterance_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_UTTERANCE_DS_PATH]
        }]
      )
    # | "Beam PL: print path to document_asl_consultant_utterance_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_utterance_index path"))
    | "Beam PL: read document asl consultant utterance index into pcoll" >> beam.FlatMap(load_document_asl_consultant_utterance_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS keys)
    | "Beam PL: apply schema to extracted document asl consultant utterance index info dicts" >> beam.Map(
        lambda d_document_asl_consultant_utterance: beam.Row(
          # SCHEMA_COL_NAMES__UTTERANCE_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'UtteranceSequence',
          #   'StartTime',
          #   'EndTime',
          #   'Tokens',
          #   'Translation'
          # ]
          DocumentID=int(d_document_asl_consultant_utterance[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[0]]),
          ASLConsultantID=int(d_document_asl_consultant_utterance[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[1]]),
          UtteranceSequence=int(d_document_asl_consultant_utterance[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[2]]),
          StartTime=int(d_document_asl_consultant_utterance[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[3]]),
          EndTime=int(d_document_asl_consultant_utterance[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[4]]),
          Tokens=str(d_document_asl_consultant_utterance[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[5]]),
          Translation=str(d_document_asl_consultant_utterance[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS[6]])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_utterance_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_utterance_index_schemad_pcoll entry"))
  ) # document_asl_consultant_utterance_index_schemad_pcoll


def load_document_asl_consultant_target_video_index_csv(d_document_asl_consultant_target_video_index_info):
  return load_csv(
    d_document_asl_consultant_target_video_index_info['document_asl_consultant_target_video_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS
  )

def pl__1__read_document_asl_consultant_target_video_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant target video index csv" >> beam.Create(
        [{
          'document_asl_consultant_target_video_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_DS_PATH]
        }]
      )
    # | "Beam PL: print path to document_asl_consultant_target_video_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_target_video_index path"))
    | "Beam PL: read document asl consultant target video index into pcoll" >> beam.FlatMap(load_document_asl_consultant_target_video_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS keys)
    | "Beam PL: apply schema to extracted document asl consultant target video index info dicts" >> beam.Map(
        lambda d_document_asl_consultant_target_video_index_info: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'TargetVideoFilename'
          # ]
          DocumentID=int(d_document_asl_consultant_target_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[0]]),
          ASLConsultantID=int(d_document_asl_consultant_target_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[1]]),
          CameraPerspective=int(d_document_asl_consultant_target_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[2]]),
          TargetVideoFilename=str(d_document_asl_consultant_target_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS[3]])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_target_video_index_schemad_pcoll entry"))
  ) # document_asl_consultant_target_video_index_schemad_pcoll


def load_document_asl_consultant_target_video_utterance_index_csv(d_document_asl_consultant_target_video_utterance_index_info):
  return load_csv(
    d_document_asl_consultant_target_video_utterance_index_info['document_asl_consultant_target_video_utterance_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=['DocumentID', 'ASLConsultantID', 'CameraPerspective', 'TargetVideoFilename', 'UtteranceSequence']
  )

def pl__1__read_document_asl_consultant_target_video_utterance_index_csv(pl):
  # VIDEO_DS_FNAME = 'document-consultant-targetvideo-index.csv'
  # VIDEO_UTTERANCE_DS_FNAME = 'document-consultant-targetvideo-utterance-index.csv'
  # this is a last minute hack!
  video_utterance_ds_path = pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_DS_PATH].replace(fidscs_globals.VIDEO_DS_FNAME, fidscs_globals.VIDEO_UTTERANCE_DS_FNAME)
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant target video utterance index csv" >> beam.Create(
        [{
          'document_asl_consultant_target_video_utterance_index_csv_path': video_utterance_ds_path
        }]
      )
    # | "Beam PL: print path to document_asl_consultant_target_video_utterance_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_target_video_utterance_index path"))
    | "Beam PL: read document asl consultant target video utterance into pcoll" >> beam.FlatMap(load_document_asl_consultant_target_video_utterance_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS keys)
    # | "Beam PL: print loaded document_asl_consultant_target_video_utterance_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_target_video_utterance_index_schemad_pcoll entry"))
    | "Beam PL: apply schema to extracted document asl consultant target video utterance index info dicts" >> beam.Map(
        lambda d_ddocument_asl_consultant_target_video_utterance_index_info: beam.Row(
          DocumentID=int(d_ddocument_asl_consultant_target_video_utterance_index_info['DocumentID']),
          ASLConsultantID=int(d_ddocument_asl_consultant_target_video_utterance_index_info['ASLConsultantID']),
          CameraPerspective=int(d_ddocument_asl_consultant_target_video_utterance_index_info['CameraPerspective']),
          TargetVideoFilename=str(d_ddocument_asl_consultant_target_video_utterance_index_info['TargetVideoFilename']),
          UtteranceSequence=int(d_ddocument_asl_consultant_target_video_utterance_index_info['UtteranceSequence'])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_utterance_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_target_video_utterance_index_schemad_pcoll entry"))
  ) # document_asl_consultant_target_video_utterance_index_schemad_pcoll


def load_document_asl_consultant_utterance_video_index_csv(d_document_asl_consultant_utterance_video_index_info):
  return load_csv(
    d_document_asl_consultant_utterance_video_index_info['document_asl_consultant_utterance_video_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS
  )

def pl__1__read_document_asl_consultant_utterance_video_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant utterance target video index csv" >> beam.Create(
        [{
          'document_asl_consultant_utterance_video_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_UTTERANCE_VIDEO_DS_PATH]
        }]
      )
    # | "Beam PL: print path to document_asl_consultant_target_video_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_target_video_index path"))
    | "Beam PL: read document asl consultant utterance target video index into pcoll" >> beam.FlatMap(load_document_asl_consultant_utterance_video_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS keys)
    | "Beam PL: apply schema to extracted document asl consultant utterance target video index info dicts" >> beam.Map(
        lambda d_document_asl_consultant_utterance_video_index_info: beam.Row(
          # SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'TargetVideoFilename',
          #   'UtteranceSequence',
          #   'CameraPerspective'
          # ]
          DocumentID=int(d_document_asl_consultant_utterance_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[0]]),
          ASLConsultantID=int(d_document_asl_consultant_utterance_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[1]]),
          TargetVideoFilename=str(d_document_asl_consultant_utterance_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[2]]),
          UtteranceSequence=int(d_document_asl_consultant_utterance_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[3]]),
          CameraPerspective=int(d_document_asl_consultant_utterance_video_index_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS[4]])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_utterance_video_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_utterance_video_index_schemad_pcoll entry"))
  ) # document_asl_consultant_utterance_video_index_schemad_pcoll


def load_document_target_video_segment_index_csv(d_document_target_video_segment_index_info):
  return load_csv(
    d_document_target_video_segment_index_info['document_target_video_segment_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS
  )

def pl__1__read_document_target_video_segment_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document target video segment index csv" >> beam.Create(
        [{
          'document_target_video_segment_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_SEGMENT_DS_PATH]
        }]
      )
    # | "Beam PL: print path to document_target_video_segment_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("document_target_video_segment_index path"))
    | "Beam PL: read document target video segment index into pcoll" >> beam.FlatMap(load_document_target_video_segment_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS keys)
    | "Beam PL: apply schema to extracted document target video segment index info dicts" >> beam.Map(
        lambda d_document_target_video_segment_index_info: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'TargetVideoFilename',
          #   'SegmentSequence',
          #   'SegmentVideoFilename',
          #   'URL'
          # ]
          DocumentID=int(d_document_target_video_segment_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[0]]),
          ASLConsultantID=int(d_document_target_video_segment_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[1]]),
          CameraPerspective=int(d_document_target_video_segment_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[2]]),
          TargetVideoFilename=str(d_document_target_video_segment_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[3]]),
          SegmentSequence=int(d_document_target_video_segment_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[4]]),
          SegmentVideoFilename=str(d_document_target_video_segment_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[5]]),
          URL=str(d_document_target_video_segment_index_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS[6]])
        )
      )
    # debug
    # | "Beam PL: print document_target_video_segment_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_target_video_segment_index_schemad_pcoll entry"))
  ) # document_target_video_segment_index_schemad_pcoll


def load_vocabulary_index_csv(d_vocabulary_index_info):
  return load_csv(
    d_vocabulary_index_info['vocabulary_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS
  )

def pl__1__read_vocabulary_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the vocabulary index csv" >> beam.Create(
        [{
          'vocabulary_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_VOCABULARY_DS_PATH]
        }]
      )
    # | "Beam PL: print path to vocabulary_index" >> beam.ParDo(PipelinePcollPrinter("vocabulary_index path"))
    | "Beam PL: read vocabulary index into pcoll" >> beam.FlatMap(load_vocabulary_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS keys)
    | "Beam PL: apply schema to extracted vocabulary index info dicts" >> beam.Map(
        lambda d_vocabulary_index_info: beam.Row(
          # SCHEMA_COL_NAMES__VOCABULARY_DS = [
          #   'TokenID',
          #   'Token'
          # ]
          TokenID=int(d_vocabulary_index_info[fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS[0]]),
          Token=str(d_vocabulary_index_info[fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS[1]])
        )
      )
    # debug
    # | "Beam PL: print vocabulary_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded vocabulary_index_schemad_pcoll entry"))
  ) # vocabulary_index_schemad_pcoll 


def load_document_asl_consultant_utterance_token_index_csv(d_document_asl_consultant_utterance_token_info):
  return load_csv(
    d_document_asl_consultant_utterance_token_info['document_asl_consultant_utterance_token_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS
  )

def pl__1__read_document_asl_consultant_utterance_token_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant utterance token index csv" >> beam.Create(
        [{
          'document_asl_consultant_utterance_token_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_DS_PATH]
        }]
      )
    # | "Beam PL: print path to document_asl_consultant_utterance_token_index" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_utterance_token_index path"))
    | "Beam PL: read document asl consultant utterance token index into pcoll" >> beam.FlatMap(load_document_asl_consultant_utterance_token_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS keys)
    # | "Beam PL: print path to d_document_asl_consultant_utterance_token_info dicts" >> beam.ParDo(PipelinePcollPrinter("d_document_asl_consultant_utterance_token_info dict"))
    | "Beam PL: apply schema to extracted document asl consultant utterance token index info dicts" >> beam.Map(
        lambda d_document_asl_consultant_utterance_token_info: beam.Row(
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
          DocumentID=int(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[0]]),
          ASLConsultantID=int(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[1]]),
          UtteranceSequence=int(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[2]]),
          TokenSequence=int(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[3]]),

          StartTime=int(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[4]]),
          EndTime=int(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[5]]),
          TokenID=int(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[6]]),
          Field=str(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[7]]),
          FieldValue=str(d_document_asl_consultant_utterance_token_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS[8]]),
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_utterance_token_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_utterance_token_index_schemad_pcoll entry"))
  ) # document_asl_consultant_utterance_token_index_schemad_pcoll


def load_document_asl_consultant_target_video_frame_index_csv(d_document_asl_consultant_target_video_frame_info):
  return load_csv(
    d_document_asl_consultant_target_video_frame_info['document_asl_consultant_target_video_frame_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS
  )

def pl__1__read_document_asl_consultant_target_video_frame_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant target video frame index csv" >> beam.Create(
        [{
          'document_asl_consultant_target_video_frame_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_FRAME_DS_PATH]
        }]
      )
    # | "Beam PL: print path to document_asl_consultant_utterance_token_index" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_utterance_token_index path"))
    | "Beam PL: read document asl consultant target video frame index into pcoll" >> beam.FlatMap(load_document_asl_consultant_target_video_frame_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS keys)
    # | "Beam PL: print path to d_document_asl_consultant_target_video_frame_info dicts" >> beam.ParDo(PipelinePcollPrinter("d_document_asl_consultant_target_video_frame_info dict"))
    | "Beam PL: apply schema to extracted document asl consultant target video frame index info dicts" >> beam.Map(
        lambda d_document_asl_consultant_target_video_frame_info: beam.Row(
          DocumentID=int(d_document_asl_consultant_target_video_frame_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS[0]]),
          ASLConsultantID=int(d_document_asl_consultant_target_video_frame_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS[1]]),
          CameraPerspective=int(d_document_asl_consultant_target_video_frame_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS[2]]),
          TargetVideoFilename=str(d_document_asl_consultant_target_video_frame_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS[3]]),
          FrameSequence=int(d_document_asl_consultant_target_video_frame_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS[4]]),
          FramePath=str(d_document_asl_consultant_target_video_frame_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS[5]]),
          # UtteranceSequence=int(d_document_asl_consultant_target_video_frame_info[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS[6]])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_frame_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_target_video_frame_index_schemad_pcoll entry"))
  ) # document_asl_consultant_target_video_frame_index_schemad_pcoll


def load_document_asl_consultant_target_video_utterance_token_frame_index_csv(d_document_asl_consultant_target_video_utterance_token_frame_info):
  return load_csv(
    d_document_asl_consultant_target_video_utterance_token_frame_info['document_asl_consultant_target_video_utterance_token_frame_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS
  )

def pl__1__read_document_asl_consultant_target_video_utterance_token_frame_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the document asl consultant target video utterance token frame index csv" >> beam.Create(
        [{
          'document_asl_consultant_target_video_utterance_token_frame_index_csv_path': pl._options._all_options[fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_FRAME_DS_PATH]
        }]
      )
    # | "Beam PL: print path to document_asl_consultant_target_video_utterance_token_frame_index" >> beam.ParDo(PipelinePcollPrinter("document_asl_consultant_target_video_utterance_token_frame_index path"))
    | "Beam PL: read document asl consultant target video utterance token frame index into pcoll" >> beam.FlatMap(load_document_asl_consultant_target_video_utterance_token_frame_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS keys)
    # | "Beam PL: print path to d_document_asl_consultant_target_video_frame_info dicts" >> beam.ParDo(PipelinePcollPrinter("d_document_asl_consultant_target_video_frame_info dict"))
    | "Beam PL: apply schema to extracted document asl consultant target video utterance token frame index info dicts" >> beam.Map(
        lambda d_document_asl_consultant_target_video_utterance_token_frame_info: beam.Row(
          # SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'TargetVideoFilename',
          #   'UtteranceSequence',
          #   'TokenSequence',
          #   'FrameSequence',
          #   'TokenID'
          # ]
          DocumentID=int(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[0]]),
          ASLConsultantID=int(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[1]]),
          CameraPerspective=int(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[2]]),
          TargetVideoFilename=str(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[3]]),
          UtteranceSequence=int(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[4]]),
          TokenSequence=int(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[5]]),
          FrameSequence=int(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[6]]),
          TokenID=int(d_document_asl_consultant_target_video_utterance_token_frame_info[fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS[7]])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll entry"))
  ) # document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll


def load_train_frame_sequences__assoc_index_csv(d_train_frame_sequences__assoc_index):
  return load_csv(
    d_train_frame_sequences__assoc_index['train_frame_sequences__assoc_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX
  )

def pl__1__read_train_frame_sequences__assoc_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the train frame sequences (assoc) index csv" >> beam.Create(
        [{
          'train_frame_sequences__assoc_index_csv_path': fidscs_globals.TRAIN_ASSOC_DS_PATH # pl._options._all_options[fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_FRAME_DS_PATH]
        }]
      )
    | "Beam PL: read train frame sequences (assoc) index into pcoll" >> beam.FlatMap(load_train_frame_sequences__assoc_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX keys)
    | "Beam PL: apply schema to extracted train frame sequences (assoc) index info dicts" >> beam.Map(
          lambda d_train_frame_sequence_info: beam.Row(
              # SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX = [
              #     'TokenID',
              #     'CameraPerspective',
              #     'ASLConsultantID',
              #     'TargetVideoFilename',
              #     'UtteranceSequence',
              #     'TokenSequence',
              #     'FrameSequence'
              # ]
              TokenID=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[0]]),
              CameraPerspective=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[1]]),
              ASLConsultantID=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[2]]),
              TargetVideoFilename=str(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[3]]),
              UtteranceSequence=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[4]]),
              TokenSequence=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[5]]),
              FrameSequence=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[6]])
          )
      )
    # debug
    # | "Beam PL: print train_frame_sequences__assoc_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded train_frame_sequences__assoc_index_schemad_pcoll entry"))
  ) # train_frame_sequences__assoc_index_schemad_pcoll


def load_val_frame_sequences_index_csv(d_val_frame_sequences_index_info):
  return load_csv(
    d_val_frame_sequences_index_info['val_frame_sequences_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX
  )

def pl__1__read_val_frame_sequences__index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the val frame sequences index csv" >> beam.Create(
        [{
          'val_frame_sequences_index_csv_path': fidscs_globals.VAL_DS_PATH # pl._options._all_options[fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_FRAME_DS_PATH]
        }]
      )
    | "Beam PL: read val frame sequences index into pcoll" >> beam.FlatMap(load_val_frame_sequences_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX keys)
    | "Beam PL: apply schema to extracted val frame sequences index info dicts" >> beam.Map(
          lambda d_val_frame_sequence_info: beam.Row(
              # SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX = [
              #     'TokenID',
              #     'CameraPerspective',
              #     'ASLConsultantID',
              #     'TargetVideoFilename',
              #     'UtteranceSequence',
              #     'TokenSequence',
              #     'FrameSequence'
              # ]
              TokenID=int(d_val_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[0]]),
              CameraPerspective=int(d_val_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[1]]),
              ASLConsultantID=int(d_val_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[2]]),
              TargetVideoFilename=str(d_val_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[3]]),
              UtteranceSequence=int(d_val_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[4]]),
              TokenSequence=int(d_val_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[5]]),
              FrameSequence=int(d_val_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[6]])
          )
      )
    # debug
    # | "Beam PL: print val_frame_sequences_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded val_frame_sequences_index_schemad_pcoll entry"))
  ) # val_frame_sequences_index_schemad_pcoll


def load_train_frame_sequences_index_csv(d_train_frame_sequences_index_info):
  return load_csv(
    d_train_frame_sequences_index_info['train_frame_sequences_index_csv_path'], 
    rows_to_dicts=True, 
    dict_field_names=fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX
  )

def pl__1__read_train_frame_sequences_index_csv(pl):
  return (
    pl
    | "Beam PL: create initial pcoll containing info to load the train frame sequences index csv" >> beam.Create(
        [{
          'train_frame_sequences_index_csv_path': fidscs_globals.TRAIN_DS_PATH # pl._options._all_options[fidscs_globals.OPT_NAME_UTTERANCE_TOKEN_FRAME_DS_PATH]
        }]
      )
    | "Beam PL: read train frame sequences index into pcoll" >> beam.FlatMap(load_train_frame_sequences_index_csv) # outputs another pcoll but with each row as dict (with fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX keys)
    | "Beam PL: apply schema to extracted train frame sequences index info dicts" >> beam.Map(
          lambda d_train_frame_sequence_info: beam.Row(
              # SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX = [
              #     'TokenID',
              #     'CameraPerspective',
              #     'ASLConsultantID',
              #     'TargetVideoFilename',
              #     'UtteranceSequence',
              #     'TokenSequence',
              #     'FrameSequence'
              # ]
              TokenID=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[0]]),
              CameraPerspective=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[1]]),
              ASLConsultantID=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[2]]),
              TargetVideoFilename=str(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[3]]),
              UtteranceSequence=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[4]]),
              TokenSequence=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[5]]),
              FrameSequence=int(d_train_frame_sequence_info[fidscs_globals.SCHEMA_COL_NAMES__TRAIN_OR_VAL_INDEX[6]])
          )
      )
    # debug
    # | "Beam PL: print train_frame_sequences_index_schemad_pcoll" >> beam.ParDo(PipelinePcollPrinter("loaded train_frame_sequences_index_schemad_pcoll entry"))
  ) # train_frame_sequences_index_schemad_pcoll
