from __future__ import absolute_import

import base64
import io
import logging
import os
import random
import re
import sys
import time
import urllib
import zipfile

import apache_beam as beam
# import apache_beam.runners.interactive.interactive_beam as ib
import apache_beam.io.fileio
# from apache_beam.transforms.sql import SqlTransform
import apache_beam.transforms.sql
import cv2
import numpy as np
from apache_beam.io.filesystems import FileSystems, GCSFileSystem
from apache_beam.options.pipeline_options import PipelineOptions

from api import beam__common, data_extractor__common, fidscs_globals, fileio
from api.signstreamxmlparser_refactored.analysis import signstream as ss

# from tensorflow.keras.preprocessing.image import img_to_array, load_img



def prepare_output_str(str, label=""):
  return f"{label+': ' if len(label)>0 else ''}{str}"


def boostrap_signstream_corpus(d_corpus_info, d_pl_options, label=""):
  """
  d_corpus_info MUST be a dict as follows:
    {
      'tmp_dir': fidscs_globals.TMP_DIR,
      'corpus_archive': fidscs_globals.CORPUS_ARCHIVE
    }

  this function downloads d_corpus_info['corpus_archive'] from http://secrets.rutgers.edu/dai/xml
    and extracts it to fileio.path_join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])
    (assuming that has not already been done - i.e. if not os.path.isdir(fileio.path_join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])) 
      or len(os.listdir(fileio.path_join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'])))==0
    )

  if the datasets exist in the storage path then this function does nothing
    (other than printing out the paths to the datasets)
  """

  print(prepare_output_str(f"CORPUS-INDEX BOOTSTRAP INFO: {d_corpus_info}", label=label))

  # download archive
  """
  requires:
    d_corpus_info['corpus_archive']
    d_corpus_info['tmp_dir']
  """
  corpus_parent_dir = d_corpus_info['tmp_dir']
  corpus_dir = fileio.path_join(corpus_parent_dir, d_corpus_info['corpus_archive'].split('.')[0])
  remote_archive_path = 'http://secrets.rutgers.edu/dai/xml/'+d_corpus_info['corpus_archive']
  local_archive_parent_dir = d_corpus_info['tmp_dir']
  local_archive_path = fileio.path_join(local_archive_parent_dir, d_corpus_info['corpus_archive'])

  memfile = data_extractor__common.download_to_memfile(remote_archive_path, block_sz=8192, display=False)
  zip_ref = zipfile.ZipFile(memfile, 'r')
  print(f"unzipping {remote_archive_path} in-memory...")
  # zip_ref.printdir()
  
  if not fileio.path_exists(corpus_dir, d_pl_options)[0]:
    fileio.make_dirs(corpus_dir, d_pl_options)

  doc_file_path_suffixes = [
    'ncslgr-xml/ncslgr10f.xml',
    'ncslgr-xml/ncslgr10m.xml',
    'ncslgr-xml/ncslgr10a.xml',
    'ncslgr-xml/ncslgr10a.xml',
    'ncslgr-xml/accident.xml',
    'ncslgr-xml/ncslgr10r.xml',
    'ncslgr-xml/DSP Immigrants Story.xml',
    'ncslgr-xml/ncslgr10i.xml',
    'ncslgr-xml/DSP Dead Dog Story.xml',
    'ncslgr-xml/roadtrip2.xml',
    'ncslgr-xml/ncslgr10h.xml',
    'ncslgr-xml/football.xml',
    'ncslgr-xml/biker.xml',
    'ncslgr-xml/ncslgr10c.xml',
    'ncslgr-xml/three pigs.xml',
    'ncslgr-xml/DSP Ski Trip Story.xml',
    'ncslgr-xml/dorm prank.xml',
    'ncslgr-xml/ncslgr10p.xml',
    'ncslgr-xml/siblings.xml',
    'ncslgr-xml/lapd.xml',
    'ncslgr-xml/ali.xml',
    'ncslgr-xml/ncslgr10e.xml',
    'ncslgr-xml/ncslgr10t.xml',
    'ncslgr-xml/ncslgr10l.xml',
    'ncslgr-xml/scarystory.xml',
    'ncslgr-xml/ncslgr10k.xml',
    'ncslgr-xml/close call.xml',
    'ncslgr-xml/speeding.xml',
    'ncslgr-xml/ncslgr10d.xml',
    'ncslgr-xml/ncslgr10j.xml',
    'ncslgr-xml/ncslgr10b.xml',
    'ncslgr-xml/boston-la.xml',
    'ncslgr-xml/ncslgr10g.xml',
    'ncslgr-xml/DSP Intro to a Story.xml',
    'ncslgr-xml/ncslgr10n.xml',
    'ncslgr-xml/whitewater.xml',
    'ncslgr-xml/ncslgr10q.xml',
    'ncslgr-xml/ncslgr10s.xml',
    'ncslgr-xml/roadtrip1.xml'
  ]

  for doc_file_path_suffix in doc_file_path_suffixes:
    bytes_unzipped = zip_ref.read(doc_file_path_suffix)
    with fileio.open_file_write(corpus_parent_dir+'/'+doc_file_path_suffix) as f:
      f.write(bytes_unzipped)
      f.close()
  zip_ref.close()
  memfile.close()
  print(f"\tDONE")

  return [fileio.path_join(corpus_dir,"*")]

class SignstreamCorpusBootstrapper(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options, label=""):
    super(SignstreamCorpusBootstrapper, self).__init__(
      fn_pcoll_element_processor=boostrap_signstream_corpus,
      kargs={'d_pl_options':d_pl_options,'label':label},
      return_result=True
    )



def get_video_segment_download_info(vid_index_schemad_pcoll_row, d_pl_options):
  """
  vid_index_schemad_pcoll_row:
    beam.Row(
      target_video_filename=str(urllib.parse.quote(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[0]])),
      video_seq_id=int(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[1]]),                            
      perspective_cam_id=int(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[2]]),                  
      compressed_mov_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[3]]),
      uncompressed_avi_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[4]]),                     
      uncompressed_avi_mirror_1_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[5]]),   
      uncompressed_avi_mirror_2_url=str(x[fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX[6]])
    )

  return:
    listof(
      {
        'target_video_fname': target_video_fname, 
        'target_video_frames_dir': target_video_frames_dir, 
        'segment_url': str(url), 
        'segment_fname': str(url).split('/')[-1]
      }
    )
  """
  target_video_fname = vid_index_schemad_pcoll_row.target_video_filename
  target_video_frames_dir = fileio.path_join(d_pl_options[fidscs_globals.OPT_NAME_STITCHED_VIDEO_FRAMES_DIR], target_video_fname.split('.')[0])
  segment_urls = vid_index_schemad_pcoll_row.compressed_mov_url.split(';') # this can be a list, separated by ';'
  return [{'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]} for url in segment_urls]

class VideoSegmentInfoGatherer(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options):
    super(VideoSegmentInfoGatherer, self).__init__(
      fn_pcoll_element_processor=get_video_segment_download_info,
      kargs={'d_pl_options':d_pl_options},
      return_result=True
    )

def beam_download_target_video_segment(d_target_vid_seg_download_info, d_pl_options, max_fail=fidscs_globals.DOWNLOAD_MAX_FAIL_COUNT, label=""):
  """
  expects d_target_vid_seg_download_info: {'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': url, 'segment_fname': url.split('/')[-1]}
  """
  segment_url = d_target_vid_seg_download_info['segment_url']
  segment_fname = d_target_vid_seg_download_info['segment_fname']
  video_dir = d_pl_options[fidscs_globals.OPT_NAME_VIDEO_DIR]  
  if not fileio.path_exists(video_dir, d_pl_options)[0]:
    fileio.make_dirs(video_dir, d_pl_options)
  local_segment_path = fileio.path_join(video_dir, segment_fname)
  n_fail = 0
  if not fileio.path_exists(local_segment_path, d_pl_options)[0]:
    while n_fail < max_fail:
      try:
        memfile = data_extractor__common.download_to_memfile(segment_url, block_sz=fidscs_globals._1MB, display=False) # returns with memfile.seek(0)
        memfile.seek(0)
        with fileio.open_file_write(local_segment_path) as f:
          f.write(memfile.getbuffer())
          f.close()
        print(f"{label+': ' if len(label)>0 else ''}Downloaded {segment_url} to {local_segment_path}")
        memfile.close()
        break
      except Exception as e:
        n_fail += 1
        if n_fail < max_fail:
          print(f"{label+': ' if len(label)>0 else ''}*** {e} ***: fail count: {n_fail}, max fail: {max_fail} --> sleeping 1 second, then trying again...")
          time.sleep(fidscs_globals.DOWNLOAD_FAIL_SLEEP_TIME)
        else:
          print(f"{label+': ' if len(label)>0 else ''}*** {e} ***: fail count: {n_fail}, max fail: {max_fail} --> giving up!")
  else:
    print(f"{label+': ' if len(label)>0 else ''}Found target video ({d_target_vid_seg_download_info['target_video_fname']}) segment {local_segment_path} (downloaded from {segment_url})".format(local_segment_path, segment_url))
  return [d_target_vid_seg_download_info] # passthrough

class VideoSegmentDownloader(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options, label=""):
    super(VideoSegmentDownloader, self).__init__(
      fn_pcoll_element_processor=beam_download_target_video_segment,
      kargs={'d_pl_options':d_pl_options,'label':label},
      return_result=True
    )


def capture_segment_video(vid_segment_path, truly_local_vid_dir, d_pl_options, debug=False):
  video_fname = vid_segment_path.split('/')[-1]

  truly_local_target_video_frames_dir = None

  fs = FileSystems.get_filesystem(vid_segment_path)
  if type(fs) == GCSFileSystem:
    if debug: print(f"\n\n\tattempting to open video {vid_segment_path} for reading...")
    with fileio.open_file_read(vid_segment_path) as f:
      if debug: print(f"\t\tSUCCESS")

      # now read from local bytes and write to GCS
      buffer = f.read()
      truly_local_vid_segment_path = truly_local_vid_dir+'/'+video_fname
      if debug: print(f"\t\tattempting to write {truly_local_vid_segment_path} (truly) locally...")
      with fileio.open_file_write(truly_local_vid_segment_path) as f_local:
        f_local.write(buffer)
        f_local.close()
        if debug: print(f"\t\t\tSUCCESS")
      f.close()

      vid_segment_path = truly_local_vid_segment_path

      # (truly local) dir for saving frames
      truly_local_target_video_frames_dir = truly_local_vid_dir+'/'+fidscs_globals.STICHED_VIDEO_FRAMES_DIR_NAME+'/'+video_fname.split('.')[0]
      if debug: print(f"\t\t\tattempting to create directory {truly_local_target_video_frames_dir} (truly_local_target_video_frames_dir) for frames extracted from (truly local) video {truly_local_vid_segment_path}...")
      if not fileio.path_exists(truly_local_target_video_frames_dir, d_pl_options, is_dir=True)[0]:
        if debug: print(f"\t\t\t\tcreating {truly_local_target_video_frames_dir}...")
        fileio.make_dirs(truly_local_target_video_frames_dir, d_pl_options)
      truly_local_target_video_frames_dir_exists = fileio.path_exists(truly_local_target_video_frames_dir, d_pl_options, is_dir=True)[0]
      if debug: print(f"\t\t\t\t\t{truly_local_target_video_frames_dir} exists: {truly_local_target_video_frames_dir_exists}")
      if not truly_local_target_video_frames_dir_exists:
        raise Exception(f"required directory truly_local_target_video_frames_dir {truly_local_target_video_frames_dir_exists} does not exist")

  if debug: print(f"\t\t\tattempting to capture (cv2.VideoCapture) video {vid_segment_path})...")

  # finally, capture the video bytes
  return cv2.VideoCapture(vid_segment_path), truly_local_target_video_frames_dir


def write_frame_to_file(frame, index, target_video_frames_dir, truly_local_target_video_frames_dir=None, debug=False):
  local_frame_path = fileio.path_join(target_video_frames_dir, f"{index}.jpg") # this is the final frame path

  if truly_local_target_video_frames_dir is not None:
    # write truly local frame file
    truly_local_frame_path = truly_local_target_video_frames_dir+'/'+f"{index}.jpg"
    if debug: print(f"\t\t\t\t\t\tattempting to write {truly_local_frame_path} frame...")
    cv2.imwrite(truly_local_frame_path, frame)
    if debug: print(f"\t\t\t\t\t\t\tSUCCESS")
    if debug: print(f"\t\t\t\t\t\t\tattempting to open {truly_local_frame_path} for read...")
    with fileio.open_file_read(truly_local_frame_path) as f_truly_local_frame:
      buffer = f_truly_local_frame.read()
      if debug: print(f"\t\t\t\t\t\t\t\tSUCCESS")
      if debug: print(f"\t\t\t\t\t\t\t\t\tattempting to open {local_frame_path} for final write...")
      with fileio.open_file_write(local_frame_path) as f_frame_final:
        f_frame_final.write(buffer)
        f_frame_final.close()
        if debug: print(f"\t\t\t\t\t\t\t\t\t\tSUCCESS")
      buffer = None
      f_truly_local_frame.close()

  else:
    if debug: print(f"\t\t\t\t\t\t\t\t\tattempting to open {local_frame_path} for final write...")
    cv2.imwrite(local_frame_path, frame)
    if debug: print(f"\t\t\t\t\t\t\t\t\t\tSUCCESS")


def beam_extract_frames(tpl_target_video_extraction_info, d_pl_options, label="", debug=False):
  """
  expects tpl_target_video_extraction_info: (video_fname, list({'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}))
  """

  # # log_results = []
  target_video_fname = tpl_target_video_extraction_info[0]
  segment_dicts = sorted(tpl_target_video_extraction_info[1], key=lambda segment_dict: segment_dict['segment_fname'])
  target_video_frames_dir = segment_dicts[0]['target_video_frames_dir']

  target_stitched_vid_name = target_video_frames_dir.split(os.path.sep)[-1]
  if not fileio.path_exists(target_video_frames_dir, d_pl_options)[0]:
    fileio.make_dirs(target_video_frames_dir, d_pl_options)

  video_dir = d_pl_options[fidscs_globals.OPT_NAME_VIDEO_DIR]
  local_vid_segment_paths = [fileio.path_join(video_dir, segment_dict['segment_fname']) for segment_dict in segment_dicts]
  for segment_dict in segment_dicts:
    segment_dict['n_frames_extracted'] = 0


  truly_local_vid_dir = None
  truly_local_vid_dir_suffix = None
  truly_local_vid_dir_root = None
  fs = FileSystems.get_filesystem(video_dir)
  if type(fs) == GCSFileSystem:
    truly_local_vid_dir_suffix = '/'.join(video_dir.split('/')[1:])
    truly_local_vid_dir_root = '/tmp/'+truly_local_vid_dir_suffix.split('/')[1]
    truly_local_vid_dir = '/tmp'+truly_local_vid_dir_suffix
    print(f"\t\tGCS storage detected! Extracting frames to truly_local_vid_dir {truly_local_vid_dir} (and will then upload to GCS after that)...")
    if debug: print(f"\t\t{truly_local_vid_dir} exists: {fileio.path_exists(truly_local_vid_dir, d_pl_options, is_dir=True)}")

    if not fileio.path_exists(truly_local_vid_dir, d_pl_options, is_dir=True)[0]:
      if debug: print(f"\tcreating {truly_local_vid_dir}...")
      truly_local_vid_dir_path_segs = truly_local_vid_dir.split('/')
      if debug: print(f"\t\ttruly_local_vid_dir_path_segs: {truly_local_vid_dir_path_segs}")
      s_cum_path = ''
      for i, truly_local_vid_dir_path_seg in enumerate(truly_local_vid_dir_path_segs[1:]):
        s_cum_path += '/'+truly_local_vid_dir_path_seg
        fileio.make_dirs(s_cum_path, d_pl_options)
      if debug: print(f"\t\t{s_cum_path} exists: {fileio.path_exists(s_cum_path, d_pl_options, is_dir=True)}")

  vc_results = [capture_segment_video(local_vid_segment_path, truly_local_vid_dir, d_pl_options, debug=debug) for local_vid_segment_path in local_vid_segment_paths]
  vid_caps = [vc_result[0] for vc_result in vc_results]
  truly_local_target_video_frames_dirs = [vc_result[1] for vc_result in vc_results]

  for seg_vid_cap in vid_caps:
    seg_vid_cap.set(cv2.CAP_PROP_FPS, fidscs_globals.FPS)
  frame_counts = list(map(lambda vc: int(vc.get(cv2.CAP_PROP_FRAME_COUNT)), vid_caps))
  n_frames_expected = sum(frame_counts)

  failed_target_videos = []

  n_stitched_frames = 0
  if n_frames_expected > 0:
    # get count of existing stitched frames in target_stitched_vid_frames_dir
    n_stitched_frames = len(fileio.list_dir(target_video_frames_dir, d_pl_options))

    b_restitch = n_stitched_frames < n_frames_expected
    n_stitched_frames = 0 if b_restitch else n_stitched_frames

    for i, seg_vid_cap in enumerate(vid_caps):
      segment_dict = segment_dicts[i]
      _n_frames_expected = frame_counts[i]

      if b_restitch:
        success, frame = seg_vid_cap.read()
        n_frames = 0
        while success:
          write_frame_to_file(
            frame, 
            n_stitched_frames, 
            target_video_frames_dir, 
            truly_local_target_video_frames_dir=truly_local_target_video_frames_dirs[i], 
            debug=debug
          )

          n_frames += 1
          n_stitched_frames += 1
          success, frame = seg_vid_cap.read()

        seg_path = local_vid_segment_paths[i]
        seg_fname = seg_path.split(os.path.sep)[-1]
        if n_frames != _n_frames_expected:
          print(f"{label+': ' if len(label)>0 else ''}{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} Cannot stitch together target video {target_video_fname} since {_n_frames_expected} frames were expected from segment {seg_fname} ({seg_path}) but only {n_frames} were successfully extracted")
          failed_target_videos.append(target_video_fname)
          fail = True
          break
        else:
          print(f"{label+': ' if len(label)>0 else ''}Added {n_stitched_frames} frames from segment {seg_fname} for target video {target_video_fname} (stitched-frames dir {target_video_frames_dir})")

      else:
        n_frames = _n_frames_expected
        print(f"{label+': ' if len(label)>0 else ''}Found existing stiched-frames for {target_stitched_vid_name} ({n_stitched_frames} frames in {target_video_frames_dir})")

      segment_dict['n_frames_extracted'] = n_frames

  else:
    if fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
      print(f"\t{fidscs_globals.VALIDATION_WARNING_TEXT} Cannot stitch together target video {target_video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segments have zero frames")
    failed_target_videos.append(target_video_fname)
    fail = True

  for local_vid_segment_path in local_vid_segment_paths:
    fileio.delete_file(local_vid_segment_path, d_pl_options)
    if fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__DEBUG:
      print(f"PROCESSED(FRAME-EXTRACTION)/DELETED taget video ({target_video_fname}) segment: {local_vid_segment_path}")

  if truly_local_vid_dir is not None and fileio.path_exists(truly_local_vid_dir, d_pl_options, is_dir=True)[0]:
    print(f"\t\tdeleting intermediate frame extraction directory {truly_local_vid_dir_root} (used for upload to GCS)...")
    fileio.delete_file(truly_local_vid_dir_root, d_pl_options)

  return [(tpl_target_video_extraction_info[0], n_stitched_frames, segment_dicts)]

class SegmentFrameExtractor(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options, label="", debug=False):
    super(SegmentFrameExtractor, self).__init__(
      fn_pcoll_element_processor=beam_extract_frames,
      kargs={'d_pl_options':d_pl_options,'label':label,'debug':debug},
      return_result=True
    )


def process_corpus_document(corpus_readable_file, d_pl_options, label, ref_CorpusDocumentFileProcessor):
  xml_db_path = str(corpus_readable_file.metadata.path)
  xml_db_fname = xml_db_path.split(os.path.sep)[-1].strip()
  # f = beam.io.filesystems.FileSystems.open(xml_db_path)
  f = fileio.open_file_read(xml_db_path)
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
    DocumentID=int(ref_CorpusDocumentFileProcessor.next_doc_id),
    Filename=xml_db_fname,
    XML_B64=raw_xml_b64,
    LEN=len(raw_xml_b64)
  )
  ref_CorpusDocumentFileProcessor.next_doc_id += 1
  fileio.delete_file(xml_db_path, d_pl_options)
  # if fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.ERROR:
  print(f"PROCESSED/DELETED corpus document {xml_db_path}") # always show this
  return [row]

class CorpusDocumentFileProcessor(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options, label=""):
    super(CorpusDocumentFileProcessor, self).__init__(
      fn_pcoll_element_processor=process_corpus_document,
      kargs={'d_pl_options':d_pl_options,'label':label,'ref_CorpusDocumentFileProcessor':self},
      return_result=True
    )
    self.label = label
    self.next_doc_id = 0
  

# class RowIndexer(beam.DoFn):
#   def __init__(self, var_name_prefix):
#     self.var_name = var_name_prefix+"_next_id"

#   def process(self, element):
#     tpl = (fidscs_globals.D_IN_MEMORY_VARS.get(self.var_name, 0), element)
#     fidscs_globals.D_IN_MEMORY_VARS[self.var_name] = fidscs_globals.D_IN_MEMORY_VARS.get(self.var_name, 0)+1
#     return [tpl]


def decode_XML(corpus_index_schemad_pcoll_row):
  """
  corpus_index_schemad_pcoll_row:
    beam.Row(
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
  """
  raw_XML_b64_as_str = corpus_index_schemad_pcoll_row.XML_B64
  raw_XML_b64_as_str = str(raw_XML_b64_as_str[2:-1]) # strip
  raw_XML_b64_to_ascii = raw_XML_b64_as_str.encode('ascii')
  raw_XML_b64 = base64.b64decode(raw_XML_b64_to_ascii)
  raw_xml = raw_XML_b64.decode('ascii').strip()
  # print(raw_xml)
  return [
    {
      'DocumentID': corpus_index_schemad_pcoll_row.DocumentID, 
      'Filename': corpus_index_schemad_pcoll_row.Filename,
      'XML': raw_xml,
      'LEN': len(raw_xml)
    }
  ]

class SSXMLDecoder(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options, label=""):
    super(SSXMLDecoder, self).__init__(
      fn_pcoll_element_processor=decode_XML,
      kargs={'d_pl_options':d_pl_options,'label':label},
      return_result=True
    )


def assign_to_global__raw_xml_b64_max_len(max_xml_b64_len):
    fidscs_globals.MAX_RAW_XML_B64_LEN = max_xml_b64_len+4
    # debug
    # print(f"ASSIGNED fidscs_globals.MAX_RAW_XML_B64_LEN={fidscs_globals.MAX_RAW_XML_B64_LEN}")
    return [max_xml_b64_len]


def boostrap_target_video_index(d_vid_indexes_info, d_pl_options):
  if d_pl_options is None or not isinstance(d_pl_options, dict):
    raise ValueError(f"require d_pl_options as dict but got {type(d_pl_options)}")

  """
  d_vid_indexes_info MUST be a dict as follows:
    {
      'vid_indexes_dir': fidscs_globals.VIDEO_INDEXES_DIR, 
      'sel_vid_index_path': fidscs_globals.SELECTED_VIDEO_INDEX_PATH, 
      'video_indexes_archive': fidscs_globals.VIDEO_INDEXES_ARCHIVE, 
      'tmp_dir': fidscs_globals.TMP_DIR,
      'video_ds_path': fidscs_globals.VIDEO_DS_PATH
    }

  this function downloads d_vid_indexes_info['video_indexes_archive'] from http://www.bu.edu/asllrp/ncslgr-for-download
    and extracts it to FileSystems.join(d_vid_indexes_info['tmp_dir'], d_vid_indexes_info['video_indexes_archive'])
    (assuming that has not already been done - i.e. if not os.path.isdir(d_vid_indexes_info['vid_indexes_dir']) or not os.path.isfile(d_vid_indexes_info['sel_vid_index_path']))

  this function returns d_vid_indexes_info['sel_vid_index_path'] only after the above has been done
  """

  remote_archive_path = 'http://www.bu.edu/asllrp/ncslgr-for-download/'+d_vid_indexes_info['video_indexes_archive']
  local_archive_parent_dir = d_vid_indexes_info['tmp_dir']
  local_archive_path = fileio.path_join(local_archive_parent_dir, d_vid_indexes_info['video_indexes_archive'])
  video_ds_path = d_vid_indexes_info['video_ds_path']

  print(f"VIDEO-INDEX BOOTSTRAP INFO: {d_vid_indexes_info}")

  memfile = data_extractor__common.download_to_memfile(remote_archive_path, block_sz=8192, display=False)
  zip_ref = zipfile.ZipFile(memfile, 'r')
  print(f"unzipping {remote_archive_path} in-memory...")
  # zip_ref.printdir()
  sel_vid_index_path = d_vid_indexes_info['sel_vid_index_path']
  sel_vid_index_path_suffix = d_vid_indexes_info['video_indexes_archive'].split('.')[0]+'/'+sel_vid_index_path.split('/')[-1]
  sel_vid_index_fname = sel_vid_index_path_suffix.split('/')[-1]
  # print(f"we need to pull {sel_vid_index_path_suffix} out of in-memory extracted archive")
  bytes_unzipped = zip_ref.read(sel_vid_index_path_suffix)
  zip_ref.close()
  if not fileio.path_exists(d_vid_indexes_info['vid_indexes_dir'], d_pl_options=d_pl_options)[0]:
    print()
    fileio.make_dirs(d_vid_indexes_info['vid_indexes_dir'], d_pl_options=d_pl_options)
  with fileio.open_file_write(d_vid_indexes_info['vid_indexes_dir']+'/'+sel_vid_index_fname) as f:
    f.write(bytes_unzipped)
    f.close()
  memfile.close()
  print(f"\tDONE")

  return [d_vid_indexes_info['sel_vid_index_path']]

class TargetVideoIndexBootstrapper(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options):
    super(TargetVideoIndexBootstrapper, self).__init__(
      fn_pcoll_element_processor=boostrap_target_video_index,
      kargs={'d_pl_options':d_pl_options},
      return_result=True
    )


def pl__1__bootstrap_target_video_index(pl):
  # ******************** start the pipeline, bootstrap video index, read it, apply schema: BEGIN ********************
  return (
    pl
    | "Beam PL: create initial pcoll containing information for boostrap_target_video_index" >> beam.Create(
        [ # one row containing dict of:
            # 1. url of video indexes archive
            # 2. local destination (path) for the downloaded archive
            # 3. local destination (path) which will receive the extracted archive csv files (there are more than one)
            # 4. final path to the selected videx index csv
            #   (note that the dict is not laid out in the above order)
          {
            'vid_indexes_dir': pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_INDEXES_DIR], 
            'sel_vid_index_path': pl._options._all_options[fidscs_globals.OPT_NAME_SELECTED_VIDEO_INDEX_PATH], 
            'video_indexes_archive': fidscs_globals.VIDEO_INDEXES_ARCHIVE, 
            'tmp_dir': pl._options._all_options[fidscs_globals.OPT_NAME_TMP_DIR],
            'video_ds_path': pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_DS_PATH]
          }
        ]
      )
    # | "Beam PL: bootstrap target video index" >> beam.Map(boostrap_target_video_index) # boostrap_target_video_index outputs SELECTED_VIDEO_INDEX_PATH but beam.Map() wraps this in a pcoll and is fed to...
    | "Beam PL: bootstrap target video index" >> beam.ParDo(TargetVideoIndexBootstrapper(pl._options._all_options)) # boostrap_target_video_index outputs SELECTED_VIDEO_INDEX_PATH but beam.Map() wraps this in a pcoll and is fed to...
    | "Beam PL: read video index into pcoll" >> beam.FlatMap(beam__common.load_vid_index_csv) # outputs another pcoll but with each row as dict
    # note that we want rows as dicts since dicts help us apply a schema to the pcoll, which is what we want in the end

    # now we want to apply the schema so that we can ultimately use beam's beam.transforms.sql.SqlTransform (very similar to pandas sqldf) when necessary
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
    # debug
    # | "Beam PL: print schemad video index pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter())  # passthrough but comment out for production
  ) # full_target_vid_index_schemad_pcoll
  # ******************** start the pipeline, bootstrap video index, read it, apply schema: END ********************


def pl__2__write_target_vid_index_csv(full_target_vid_index_schemad_pcoll, d_pl_options):
  if d_pl_options is None or not isinstance(d_pl_options, dict):
    raise ValueError(f"require d_pl_options as dict but got {type(d_pl_options)}")

  sorted_full_target_vid_index_schemad_pcoll = beam__common.pl__X__sort_pcoll(full_target_vid_index_schemad_pcoll, pcoll_label="full_target_vid_index")
  sorted_corpus_index_csv_rows_pcoll = (
    sorted_full_target_vid_index_schemad_pcoll
    | "Beam PL: re-apply schema to sorted_full_target_vid_index" >> beam.Map(lambda sorted_full_target_vid_index_schemad_pcoll_row: beam.Row(
          target_video_filename=sorted_full_target_vid_index_schemad_pcoll_row.target_video_filename,
          video_seq_id=sorted_full_target_vid_index_schemad_pcoll_row.video_seq_id,                            
          perspective_cam_id=sorted_full_target_vid_index_schemad_pcoll_row.perspective_cam_id,                  
          compressed_mov_url=sorted_full_target_vid_index_schemad_pcoll_row.compressed_mov_url,            
          uncompressed_avi_url=sorted_full_target_vid_index_schemad_pcoll_row.uncompressed_avi_url,                     
          uncompressed_avi_mirror_1_url=sorted_full_target_vid_index_schemad_pcoll_row.uncompressed_avi_mirror_1_url,   
          uncompressed_avi_mirror_2_url=sorted_full_target_vid_index_schemad_pcoll_row.uncompressed_avi_mirror_2_url
        )
      )
    | beam.Map(lambda sorted_full_target_vid_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(sorted_full_target_vid_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_corpus_index_csv_rows_pcoll, 
    "TARGET-VIDEO-INDEX", 
    fidscs_globals.VIDEO_INDEXES_ARCHIVE, 
    fidscs_globals.SCHEMA_COL_NAMES__VIDEO_INDEX,
    d_pl_options
  )


def pl__2__filter_target_vid_index(full_target_vid_index_schemad_pcoll, d_pl_options):
  max_target_videos = d_pl_options[fidscs_globals.OPT_NAME_MAX_TARGET_VIDEOS]
  # ******************** filter schemad target video index pcoll as desired (if necessary) using beam.transforms.sql.SqlTransform(), for example limiting size of pcoll data items to fidscs_globals.MAX_TARGET_VIDEOS: BEGIN ********************
  if max_target_videos is not None and max_target_videos>0:
    return beam__common.pl__X__subset_pcoll(full_target_vid_index_schemad_pcoll, "full_target_vid_index_schemad_pcoll", max_target_videos)
  else:
    return full_target_vid_index_schemad_pcoll
  # ******************** filter schemad video index pcoll as desired (if necessary) using beam.transforms.sql.SqlTransform(), for example limiting size of pcoll data items to fidscs_globals.MAX_TARGET_VIDEOS: END ********************


def pl__1__bootstrap_corpus_index(pl):
  # ******************** bootstrap SignStream corpus: BEGIN ********************
  corpus_documents_dir_path_schemad_pcoll = (
    pl
    | "Beam PL: create initial pcoll containing information for boostrap_signstream_corpus" >> beam.Create(
        [
          {
            'tmp_dir': pl._options._all_options[fidscs_globals.OPT_NAME_TMP_DIR],
            'corpus_archive': fidscs_globals.CORPUS_ARCHIVE
          }
        ]
      )
    # | "Beam PL: bootstrap SignStream corpus" >> beam.FlatMap(boostrap_signstream_corpus) # boostrap_signstream_corpus outputs [fileio.path_join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'].split('.')[0])] if datasets do not yet exist, otherwise []
    | "Beam PL: bootstrap SignStream corpus" >> beam.ParDo(SignstreamCorpusBootstrapper(pl._options._all_options)) # boostrap_signstream_corpus outputs [fileio.path_join(d_corpus_info['tmp_dir'], d_corpus_info['corpus_archive'].split('.')[0])] if datasets do not yet exist, otherwise []
    | "Beam PL: apply schema to corpus document files path pcoll" >> beam.Map(lambda x: beam.Row(corpus_docs_dir=str(x)))
  )
  return corpus_documents_dir_path_schemad_pcoll
  # ******************** bootstrap SignStream corpus: END ********************


def pl__1__corpus_document_file_structure_to_corpus_index(pl):
  tmp_dir = pl._options._all_options[fidscs_globals.OPT_NAME_TMP_DIR]
  return (
    pl
    | "Beam PL: get corpus documents" >> beam.io.fileio.MatchFiles(fileio.path_join(fileio.path_join(tmp_dir, fidscs_globals.CORPUS_ARCHIVE.split('.')[0]), "*"))
    | "Beam PL: read corpus documents" >> beam.io.fileio.ReadMatches() # this results in a pcoll of fileio.ReadableFile objects
    | "Beam PL: create corpus index dataset" >> beam.ParDo(CorpusDocumentFileProcessor(pl._options._all_options))
  ) # corpus_index_schemad_pcoll


def pl__2__write_corpus_index_csv(corpus_index_schemad_pcoll, global_var_value_assigner__raw_xml_b64_max_len, d_pl_options):
  corpus_index_pcoll = (
    corpus_index_schemad_pcoll
    | "Beam PL: extract (<corpus doc id>, <corpus doc filename>, <xml (base-64)>, <length of xml (base-64)>)" >> beam.Map(
        lambda corpus_index_schemad_pcoll_row:
        # row = beam.Row(
        #   # SCHEMA_COL_NAMES__CORPUS_DS = [
        #   #   'DocumentID',
        #   #   'Filename',
        #   #   'XML_B64',
        #   #   'LEN'
        #   # ]
        #   DocumentID=int(self.next_doc_id),
        #   Filename=xml_db_fname,
        #   XML_B64=raw_xml_b64,
        #   LEN=len(raw_xml_b64)
        # )
        (
          corpus_index_schemad_pcoll_row.DocumentID,
          corpus_index_schemad_pcoll_row.Filename,
          corpus_index_schemad_pcoll_row.XML_B64,
          corpus_index_schemad_pcoll_row.LEN
        )
      )
  )
  sorted_corpus_index_pcoll = beam__common.pl__X__sort_pcoll(corpus_index_pcoll, pcoll_label="corpus_index")
  sorted_corpus_index_csv_rows_pcoll = (
    sorted_corpus_index_pcoll
    | "Beam PL: re-apply schema to sorted_corpus_index" >> beam.Map(
        lambda sorted_corpus_index_pcoll_row: beam.Row(
          DocumentID=sorted_corpus_index_pcoll_row[0],
          Filename=sorted_corpus_index_pcoll_row[1],
          XML_B64=sorted_corpus_index_pcoll_row[2],
          LEN=sorted_corpus_index_pcoll_row[3]
        )
      )
    | beam.Map(lambda corpus_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(corpus_index_schemad_pcoll_row))
  )
  corpus_index_csv_path = beam__common.pl__X__write_pcoll_to_csv(
    sorted_corpus_index_csv_rows_pcoll, 
    "CORPUS-INDEX", 
    fidscs_globals.CORPUS_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__CORPUS_DS,
    d_pl_options
  )

  max_xml_b64_len = (
    corpus_index_schemad_pcoll
    | "Beam PL: select LEN" >> beam.Map(lambda corpus_index_schemad_pcoll_row: corpus_index_schemad_pcoll_row.LEN)
    | beam.CombineGlobally(lambda corpus_index_b64_doc_length_rows: max(corpus_index_b64_doc_length_rows or [None]))
    # debug
    # | "Beam PL: print max (b64-encoded) length corpus doc" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="MAX (b64-encoded) DOC LENGTH"))
  )
  # corpus_index_csv_path_indexed = (
  #   corpus_index_csv_path
  #   | "Beam PL: apply RowIndex to corpus index csv path" >> beam.ParDo(RowIndexer(var_name_prefix="corpus_index_csv_path_id"))
  #   # debug
  #   # | "Beam PL: print indexed path to corpus index csv" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="INDEXED CORPUS INDEX CSV PATH"))
  # )
  corpus_index_csv_path_indexed = beam__common.pl__X__index_pcoll(corpus_index_csv_path, "corpus_index_csv_path")
  max_xml_b64_len_indexed = (
    max_xml_b64_len
    | "Beam PL: assign to global var (fidscs_globals.MAX_RAW_XML_B64_LEN)" >> beam.ParDo(global_var_value_assigner__raw_xml_b64_max_len) 
    # | "Beam PL: apply RowIndex to maxlen" >> beam.ParDo(RowIndexer(var_name_prefix="max_xml_b64_len_id"))
    # debug
    # | "Beam PL: print indexed max (b64-encoded) length corpus doc" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="INDEXED MAX (b64-encoded) DOC LENGTH"))
  )
  max_xml_b64_len_indexed = beam__common.pl__X__index_pcoll(max_xml_b64_len_indexed, "max_xml_b64_len_indexed")
  combined_corpus_index_csv_path_and_max_xml_b64_len_indexed = (
    ({
      'corpus_index_csv_path': corpus_index_csv_path_indexed,
      'max_len': max_xml_b64_len_indexed
    })
    | "Beam PL: merge corpus_index_csv_path and max_len" >> beam.CoGroupByKey()
    # debug
    # | "Beam PL: print combined results" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="READ CORPUS INDEX CSV TO PCOLL"))
  )

  return combined_corpus_index_csv_path_and_max_xml_b64_len_indexed


def pl__2__decode_XML(corpus_index_schemad_pcoll, d_pl_options):
  # each row is of the form {'DocumentID': '37', 'Filename': ' biker.xml', 'XML_B64', 'LEN'}
  """
  corpus_index_schemad_pcoll:
    beam.Row(
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
  """
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

              'TARGET_VIDEO_SEQUENCE': [
                {
                  'TARGET_VIDEO_FNAME': <target vid fname>,
                  'TARGET_VIDEO_CAMERA_PERSPECTIVE': <target vid camera perspective>
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
      for target_video in utterance.get_media():
        target_video_record = {}
        target_video_fname = str(urllib.parse.quote(target_video.get_filename().split(':')[-1]))
        target_video_record['TARGET_VIDEO_FNAME'] = target_video_fname
        media_camera_perspective = -1 # need to look this up!
        target_video_record['TARGET_VIDEO_CAMERA_PERSPECTIVE'] = media_camera_perspective
        media_url = "<need to look this up!>"
        target_video_record['MEDIA_URL'] = media_url
        media_sequence.append(target_video_record)
      utterance_record['TARGET_VIDEO_SEQUENCE'] = media_sequence

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
            'vid_index_path': fileio.path_join(fidscs_globals.DATA_ROOT_DIR, fidscs_globals.VIDEO_INDEXES_ARCHIVE.split('.')[0]+'.csv')
          }
        ]
      )
    # debug
    | "Beam PL: print saved vid index path" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="READ SAVED VID INDEX PATH"))
  )


def pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll, d_pl_options):
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
    particpant_info_tpl_list = list(tpl_participant_info_grouped_by_name[1])
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
        if len(multiple_ages) > 1 and fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
          print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} participant {participant_name} age is not unique: {multiple_ages}; assigning greatest value (most recent): {age}")
      else:
        if fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
          print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} participant {participant_name} age info does not exist; assigning default age (-1)")
        age = -1

      multiple_genders = set(multiple_genders)
      if len(multiple_genders) > 0 and (gender is None or len(gender)==0):
        for _gender in multiple_genders:
          if len(_gender)>0:
            gender = _gender
            if len(multiple_genders) > 1 and fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
              print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} participant {participant_name} gender is not unique: {multiple_genders}; current gender is {gender}; assigning first (non-empty) gender: {_gender}")              
            break

      return [(participant_name, age, gender)]
    else:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} participant {participant_name} does not have any associated info")
      return [tpl_participant_info_grouped_by_name] # passthrough
      

def pl__4__create_asl_consultant_index_schemad_pcoll(ss_parsed_xmldb_pcoll, d_pl_options):
  validated_mapping = (
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
    # | "Beam PL: print participant record for document" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="participant record"))
    | "Beam PL: group participants keyed by named" >> beam.GroupByKey()
    # the above produces tuples of the form:
    #   (<participant name>, [(<participant age (as string)>, participant_gender)])
    | "Beam PL: validate/preprocess participant_to_asl_consultant_id mapping" >> beam.FlatMap(validate_preprocess_participant_to_asl_consultant_id) # outputs (<participant name>, <participant age (most recent)>, <participant gender>)
  )
  indexed_validated_mapping = beam__common.pl__X__index_pcoll(validated_mapping, "validated_mapping")
  return (
    indexed_validated_mapping
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
    # | "Beam PL: print asl_consultant_index_schemad_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="asl_consultant_index_schemad_pcoll entry"))
  ) # asl_consultant_index_schemad_pcoll


def pl__5__write_asl_consultant_index_csv(asl_consultant_index_schemad_pcoll, d_pl_options):
  sorted_asl_consultant_index_schemad_pcoll = beam__common.pl__X__sort_pcoll(asl_consultant_index_schemad_pcoll, pcoll_label="asl_consultant_index")
  sorted_asl_consultant_index_csv_rows_pcoll = (
    sorted_asl_consultant_index_schemad_pcoll
    | "Beam PL: re-apply schema to sorted_asl_consultant_index" >> beam.Map(lambda sorted_asl_consultant_index_schemad_pcoll_row: beam.Row(
        ASLConsultantID=sorted_asl_consultant_index_schemad_pcoll_row.ASLConsultantID,
        Name=sorted_asl_consultant_index_schemad_pcoll_row.Name,
        Age=sorted_asl_consultant_index_schemad_pcoll_row.Age,                  
        Gender=sorted_asl_consultant_index_schemad_pcoll_row.Gender
      )
    )
    | beam.Map(lambda asl_consultant_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(asl_consultant_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_asl_consultant_index_csv_rows_pcoll, 
    "ASLCONSULTANT-INDEX", 
    fidscs_globals.ASL_CONSULTANT_DS_FNAME,
    fidscs_globals.SCHEMA_COL_NAMES__ASL_CONSULTANT_DS,
    d_pl_options
  ) # asl_consultant_index_csv_path


def pl__5__create_document_asl_consultant_index_schemad_pcoll(ss_parsed_xmldb_pcoll, corpus_index_schemad_pcoll, asl_consultant_index_schemad_pcoll, d_pl_options):
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
    # | "Beam PL: print document-participant record" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document-participant record"))
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
    # | "Beam PL: print document-participant records" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document-participant entry"))
  )

  document_id_pcoll = (
    corpus_index_schemad_pcoll
    | "Beam PL: extract (<document filename>, <document id>) from corpus_index_schemad_pcoll" >> beam.Map(
        lambda corpus_index_schemad_pcoll_row: (
          corpus_index_schemad_pcoll_row.Filename,
          corpus_index_schemad_pcoll_row.DocumentID
        )
      )
    # | "Beam PL: print document-id-to-filename records" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document-id-to-filename entry"))
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
    # | "Beam PL: print merged document_id_pcoll and document_participant_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="merged document_id_pcoll and document_participant_pcoll entry"))
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
    # | "Beam PL: print extracted (<participant name>, <asl consultant id>) from asl_consultant_index_schemad_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="extracted (<participant name>, <asl consultant id>) from asl_consultant_index_schemad_pcoll"))
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
    # | "Beam PL: print document_asl_consultant_mapping" >> beam.ParDo(beam__common.PipelinePcollPrinter("document_asl_consultant_mapping entry"))
  )

  return document_asl_consultant_index_schemad_pcoll


def pl__6__write_document_asl_consultant_index_csv(document_asl_consultant_index_schemad_pcoll, d_pl_options):
  distinct_document_asl_consultant_index_schemad_pcoll = (
    document_asl_consultant_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS columns from document_asl_consultant_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_index_schemad_pcoll_row: (
          document_asl_consultant_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_index_schemad_pcoll_row.ASLConsultantID,
          document_asl_consultant_index_schemad_pcoll_row.Filename,
          document_asl_consultant_index_schemad_pcoll_row.ParticipantName
        )
      )
    | "Beam PL: select distinct document_asl_consultant_index rows" >> beam.Distinct()
  )
  sorted_distinct_document_asl_consultant_index_schemad_pcoll= beam__common.pl__X__sort_pcoll(distinct_document_asl_consultant_index_schemad_pcoll, pcoll_label="distinct_document_asl_consultant_index")
  sorted_distinct_document_asl_consultant_index_csv_rows_pcoll = (
    sorted_distinct_document_asl_consultant_index_schemad_pcoll
    | "Beam PL: apply minimal schema to create final document_asl_consultant_index_schemad_pcoll of distinct rows" >> beam.Map(
        lambda sorted_distinct_document_asl_consultant_index_row: beam.Row(
          DocumentID=int(sorted_distinct_document_asl_consultant_index_row[0]),
          ASLConsultantID=int(sorted_distinct_document_asl_consultant_index_row[1]),
          Filename=str(sorted_distinct_document_asl_consultant_index_row[2]),
          ParticipantName=str(sorted_distinct_document_asl_consultant_index_row[3])
        )
      )
    | beam.Map(lambda distinct_document_asl_consultant_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(distinct_document_asl_consultant_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_distinct_document_asl_consultant_index_csv_rows_pcoll, 
    "DOCUMENT-ASLCONSULTANT-INDEX", 
    fidscs_globals.DOCUMENT_ASL_CONSULTANT_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__DOCUMENT_ASL_CONSULTANT_DS,
    d_pl_options
  ) # document_asl_consultant_index_csv_path


def pl__6__create_document_asl_consultant_utterance_index_schemad_pcoll(ss_parsed_xmldb_pcoll, document_asl_consultant_index_schemad_pcoll, d_pl_options):
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
    # | "Beam PL: print corpus_document_participant_utterance_mapping" >> beam.ParDo(beam__common.PipelinePcollPrinter("corpus_document_participant_utterance_mapping entry"))
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
    # | "Beam PL: print corpus_document_participant_doc_id_asl_consultant_id_mapping" >> beam.ParDo(beam__common.PipelinePcollPrinter("corpus_document_participant_doc_id_asl_consultant_id_mapping entry"))
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
    # | "Beam PL: print document_asl_consultant_utterance_index_schemad_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter("document_asl_consultant_utterance_index_schemad_pcoll entry"))
  )

  return document_asl_consultant_utterance_index_schemad_pcoll


def pl__7__write_document_asl_consultant_utterance_index_csv(document_asl_consultant_utterance_index_schemad_pcoll, d_pl_options):
  distinct_document_asl_consultant_utterance_index_schemad_pcoll = (
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
  )
  sorted_distinct_document_asl_consultant_utterance_index_schemad_pcoll= beam__common.pl__X__sort_pcoll(distinct_document_asl_consultant_utterance_index_schemad_pcoll, pcoll_label="distinct_document_asl_consultant_utterance_index")
  sorted_distinct_document_asl_consultant_utterance_index_csv_rows_pcoll = (
    sorted_distinct_document_asl_consultant_utterance_index_schemad_pcoll
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
    | beam.Map(lambda distinct_document_asl_consultant_utterance_index_row: beam__common.beam_row_to_csv_string(distinct_document_asl_consultant_utterance_index_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_distinct_document_asl_consultant_utterance_index_csv_rows_pcoll, 
    "DOCUMENT-ASLCONSULTANT-UTTERANCE-INDEX", 
    fidscs_globals.UTTERANCE_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_DS, 
    d_pl_options
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
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid doc_fname {doc_fname}!")
    return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
  participant_name = merged_doc_participant_utterance_token[0][1]
  if len(participant_name)==0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid participant_name {participant_name}!")
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
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} (<corpus doc id>, <asl consultant id>) association is not unique! It occurs has the following (<corpus doc id>, <asl consultant id>) associations: {multiple_associations}")
      return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
    else:
      doc_id = multiple_docs[0]
      asl_consultant_id = multiple_asl_consultants[0]
  else:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} does not have a (<corpus doc id>, <asl consultant id>) association!")
    return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
  
  if len(utterance_token_mapping) > 0:
    for utterance_token_mapping_instance in utterance_token_mapping:
      _utterance_seq_id = utterance_token_mapping_instance[0]
      if not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _utterance_seq_id {_utterance_seq_id} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_ling_text = utterance_token_mapping_instance[1]
      if len(_token_ling_text)==0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_ling_text {_token_ling_text} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_new_seq_id = utterance_token_mapping_instance[2]
      if not isinstance(_token_new_seq_id, int) or _token_new_seq_id<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_new_seq_id {_token_new_seq_id} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_start_time = utterance_token_mapping_instance[3]
      if not isinstance(_token_start_time, int) or _token_start_time<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_start_time {_token_start_time} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      _token_end_time = utterance_token_mapping_instance[4]
      if not isinstance(_token_end_time, int) or _token_end_time<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} contains invalid _token_end_time {_token_end_time} in utterance_token_mapping!")
        return merged_doc_participant_utterance_token # this will throw an exception since other validation rows be differently shaped
      
      validated_results.append(
        (
          (doc_id, asl_consultant_id, _utterance_seq_id, _token_new_seq_id), 
          (doc_fname, participant_name, _token_ling_text, _token_start_time, _token_end_time)
        )
      )
  else:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} merged_doc_participant_utterance_token key {merged_doc_participant_utterance_token[0]} is not associated with an utterance_token_mapping!")
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
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl key is invalid: {token_ling_text}!")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped

  vocabulary_token_id_map = document_asl_consultant_utterance_token_tpl[1]['vocabulary_token_id_map']
  if len(vocabulary_token_id_map) == 0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) does not have a <vocab token id> association!")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped

  doc_participant_utterance_token_info_map = document_asl_consultant_utterance_token_tpl[1]['doc_participant_utterance_token_info_map']
  if len(doc_participant_utterance_token_info_map) == 0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) does not have a doc_participant_utterance_token_info_map!")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped

  validated_results = []

  multiple_token_ids = []

  vocab_token_id = None
  for vocabulary_token_id_map_instance in vocabulary_token_id_map:
    _vocab_token_id = vocabulary_token_id_map_instance
    if isinstance(_vocab_token_id, int) and _vocab_token_id>-1 and _vocab_token_id not in multiple_token_ids:
      multiple_token_ids.append(_vocab_token_id)
  if len(multiple_token_ids) > 1:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) <vocab token id> association is not unique! It occurs has the following <vocab token id> associations: {multiple_token_ids}")
    return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
  else:
    vocab_token_id = multiple_token_ids[0]

  for doc_participant_utterance_token_info_map_instance in doc_participant_utterance_token_info_map:
    _corpus_doc_id = doc_participant_utterance_token_info_map_instance[0]
    if not isinstance(_corpus_doc_id, int) or _corpus_doc_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _corpus_doc_id {_corpus_doc_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _doc_fname = doc_participant_utterance_token_info_map_instance[1]
    if len(_doc_fname)==0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _doc_fname {_doc_fname} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _asl_consultant_id = doc_participant_utterance_token_info_map_instance[2]
    if not isinstance(_asl_consultant_id, int) or _asl_consultant_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _asl_consultant_id {_asl_consultant_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _participant_name = doc_participant_utterance_token_info_map_instance[3]
    if len(_participant_name)==0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _participant_name {_participant_name} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _utterance_seq_id = doc_participant_utterance_token_info_map_instance[4]
    if not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _utterance_seq_id {_utterance_seq_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _token_new_seq_id = doc_participant_utterance_token_info_map_instance[5]
    if not isinstance(_token_new_seq_id, int) or _token_new_seq_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _token_new_seq_id {_token_new_seq_id} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _token_start_time = doc_participant_utterance_token_info_map_instance[6]
    if not isinstance(_token_start_time, int) or _token_start_time<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _token_start_time {_token_start_time} in doc_participant_utterance_token_info_map!")
      return document_asl_consultant_utterance_token_tpl # this will throw an exception since other validation rows be differently shaped
    _token_end_time = doc_participant_utterance_token_info_map_instance[7]
    if not isinstance(_token_end_time, int) or _token_end_time<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant_utterance_token_tpl (key {token_ling_text}) contains invalid _token_end_time {_token_end_time} in doc_participant_utterance_token_info_map!")
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


def pl__6__create_document_asl_consultant_utterance_token_index_schemad_pcoll(ss_parsed_xmldb_pcoll, document_asl_consultant_index_schemad_pcoll, d_pl_options):
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
    # | "Beam PL: print doc_participant_utterance_token_mapping" >> beam.ParDo(beam__common.PipelinePcollPrinter("doc_participant_utterance_token_mapping entry"))
  )

  # now extract distinct token linguistic text from doc_participant_utterance_token_mapping to build the final vocabulary index
  ling_text_pcoll = (
    doc_participant_utterance_token_mapping
    # ((<corpus doc filename>, <participant name>, <utterance seq id>, <token linguistic text>), (<token (new) seq id>, <token start time>, <token end time>))
    | "Beam PL: extract token linguistic text" >> beam.Map(lambda doc_participant_utterance_token_mapping_row_tpl: doc_participant_utterance_token_mapping_row_tpl[0][3])
    | "Beam PL: select distinct token linguistic text" >> beam.Distinct()
  )
  indexed_ling_text = beam__common.pl__X__index_pcoll(ling_text_pcoll, "ling_text_pcoll")
  # the above produces tuples of the form:
      # (<vocab token id>, <vocab token linguistic text>)
  vocabulary_index_pcoll = (
    indexed_ling_text
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
    # | "Beam PL: print vocabulary" >> beam.ParDo(beam__common.PipelinePcollPrinter("vocabulary token"))
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
    # | "Beam PL: print validated merged_doc_participant_utterance_token" >> beam.ParDo(beam__common.PipelinePcollPrinter("merged_doc_participant_utterance_token (validated) entry"))
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
    # | "Beam PL: print document_asl_consultant_utterance_token_index_schemad_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter("document_asl_consultant_utterance_token_index_schemad_pcoll entry"))
  )

  return vocabulary_index_pcoll, document_asl_consultant_utterance_token_index_schemad_pcoll


def pl__7__write_vocabulary_index_csv(vocabulary_index_pcoll, d_pl_options):
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
  sorted_vocabulary_index_pcoll = beam__common.pl__X__sort_pcoll(vocabulary_index_pcoll, pcoll_label="vocabulary_index")
  sorted_vocabulary_index_csv_rows_pcoll = (
    sorted_vocabulary_index_pcoll
    | "Beam PL: re-apply schema to sorted_vocabulary_index_pcoll rows" >> beam.Map(
        lambda sorted_vocabulary_index_pcoll_row: beam.Row(
          # SCHEMA_COL_NAMES__VOCABULARY_DS = [
          #   'TokenID',
          #   'Token'
          # ]
          TokenID=sorted_vocabulary_index_pcoll_row.TokenID,
          Token=sorted_vocabulary_index_pcoll_row.Token
        )
      )
    | beam.Map(lambda vocabulary_index_pcoll_row: beam__common.beam_row_to_csv_string(vocabulary_index_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_vocabulary_index_csv_rows_pcoll, 
    "VOCABULARY-INDEX", 
    fidscs_globals.VOCABULARY_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__VOCABULARY_DS, 
    d_pl_options
  ) # vocabulary_index_csv_path


def pl__7__write_document_asl_consultant_utterance_token_index_csv(document_asl_consultant_utterance_token_index_schemad_pcoll, d_pl_options):
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
  distinct_document_asl_consultant_utterance_token_index_pcoll = (
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
  )
  sorted_document_asl_consultant_utterance_token_index_schemad_pcoll = beam__common.pl__X__sort_pcoll(
    distinct_document_asl_consultant_utterance_token_index_pcoll, 
    pcoll_label="document_asl_consultant_utterance_token_index"
  )
  sorted_document_asl_consultant_utterance_token_index_csv_rows_pcoll = (
    sorted_document_asl_consultant_utterance_token_index_schemad_pcoll
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
    | beam.Map(lambda distinct_document_asl_consultant_utterance_token_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(distinct_document_asl_consultant_utterance_token_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_document_asl_consultant_utterance_token_index_csv_rows_pcoll, 
    "DOCUMENT-ASLCONSULTANT-UTTERANCE-TOKEN-INDEX", 
    fidscs_globals.UTTERANCE_TOKEN_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_DS, 
    d_pl_options
  ) # document_asl_consultant_utterance_token_index_csv_path


def validate_preprocess_document_asl_consultant__to__target_video_utterance_token_map_tpl(document_asl_consultant__to__target_video_utterance_token_map_tpl, d_pl_options):
  # document_asl_consultant__to__target_video_utterance_token_map_tpl:
    # (
    #   (<corpus doc id>, <asl consultant id>), # key
    #   {
    #     'target_video_map': [(<target video fname>, <camera perspective>)], # there may be up to three (corresponding to camera perspective)
    #     'utterance_token_map': [(<utterance seq id>, <token seq id>, <token id>, <token start time>, <token end time>)] # there will be many (corresponding to each utterance)
    #   }
    # )

  validated_results = []

  key = document_asl_consultant__to__target_video_utterance_token_map_tpl[0]

  corpus_doc_id = key[0]
  if corpus_doc_id is None or not isinstance(corpus_doc_id, int) or corpus_doc_id<0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} corpus doc id is invalid: {corpus_doc_id}")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  asl_consultant_id = key[1]
  if asl_consultant_id is None or not isinstance(asl_consultant_id, int) or asl_consultant_id<0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} asl consultant id is invalid: {asl_consultant_id}")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  target_video_map = document_asl_consultant__to__target_video_utterance_token_map_tpl[1]['target_video_map']
  if len(target_video_map)==0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} {key} target_video_map is empty!")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  utterance_token_map = document_asl_consultant__to__target_video_utterance_token_map_tpl[1]['utterance_token_map']
  if len(utterance_token_map)==0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} {key} utterance_token_map is empty!")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  for i, utterance_token_map_instance in enumerate(utterance_token_map):
    # (<utterance seq id>, <token seq id>, <token id>, <token start time>, <token end time>)
    _utterance_seq_id = utterance_token_map_instance[0]
    if _utterance_seq_id is None or not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] utterance_seq_id is invalid: {_utterance_seq_id}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_seq_id = utterance_token_map_instance[1]
    if _token_seq_id is None or not isinstance(_token_seq_id, int) or _token_seq_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_seq_id is invalid: {_token_seq_id}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_id = utterance_token_map_instance[2]
    if _token_id is None or not isinstance(_token_id, int) or _token_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_id is invalid: {_token_id}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_start_time = utterance_token_map_instance[3]
    if _token_start_time is None or not isinstance(_token_start_time, int) or _token_start_time<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_start_time is invalid: {_token_start_time}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_end_time = utterance_token_map_instance[4]
    if _token_end_time is None or not isinstance(_token_end_time, int) or _token_end_time<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_end_time is invalid: {_token_end_time}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl

    for j, target_video_map_instance in enumerate(target_video_map):
      # (<target video fname>, <camera perspective>)
      _target_video_fname = target_video_map_instance[0]
      if _target_video_fname is None or len(_target_video_fname)==0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname is invalid: {_target_video_fname}")
        return document_asl_consultant__to__target_video_utterance_token_map_tpl

      target_video_frames_dir = fileio.path_join(fidscs_globals.STICHED_VIDEO_FRAMES_DIR, _target_video_fname.split('.')[0])
      _n_existing_frame_images = -1 if not fileio.path_exists(target_video_frames_dir, d_pl_options)[0] else len(fileio.list_dir(target_video_frames_dir, d_pl_options))
      # if _n_existing_frame_images == -1:
      #   print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant__to__target_video_utterance_token_map_tpl utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} frames dir ({target_video_frames_dir}) does not exist!")
      _token_end_frame = _token_start_frame = -1
      frame_seq_paths = []
      if _n_existing_frame_images > 0:
        _token_start_frame = int(round(_token_start_time/1000.0*fidscs_globals.FPS))
        _token_end_frame = int(round(_token_end_time/1000.0*fidscs_globals.FPS))+1
        n_frames = _token_end_frame-_token_start_frame
        last_frame_idx = (_n_existing_frame_images-1)

        if _token_start_frame > last_frame_idx:
          # comment out for now
          # print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} _token_start_frame ({_token_start_frame}) > _n_existing_frame_images ({_n_existing_frame_images}): reconciling bounds of {n_frames} frames (from last frame index {last_frame_idx}) to {(last_frame_idx-(n_frames-1), last_frame_idx)}")
          # return document_asl_consultant__to__target_video_utterance_token_map_tpl
          # readjust bounds from last_frame_idx going backwards
          _token_start_frame = last_frame_idx-(n_frames-1)
          _token_end_frame = last_frame_idx
        else:
          if _token_end_frame > last_frame_idx:
            # print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} _token_end_frame ({_token_end_frame}) > _n_existing_frame_images ({_n_existing_frame_images}): reconciling _token_end_frame to {last_frame_idx}")
            # return document_asl_consultant__to__target_video_utterance_token_map_tpl
            # take all that is available to the end
            _token_end_frame = last_frame_idx

        if _token_start_frame <= last_frame_idx and _token_end_frame <= last_frame_idx:
          for frame_idx in range(_token_start_frame, _token_end_frame+1):
            frame_path = fileio.path_join(target_video_frames_dir, f"{frame_idx}.jpg")
            if fileio.path_exists(frame_path)[0]:
              frame_seq_paths.append(frame_path)
            else:
              print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname}: failed to reconcile invalid requested frame bounds {(_token_start_frame, _token_end_frame)} (valid bounds are: {(0, last_frame_idx)})")
              # return document_asl_consultant__to__target_video_utterance_token_map_tpl
        else:
          if fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
            print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} _token_end_frame ({_token_end_frame}) > _n_existing_frame_images ({_n_existing_frame_images}): reconciling _token_end_frame to {last_frame_idx}")

      _camera_perspective = target_video_map_instance[1]
      if _camera_perspective is None or not isinstance(_camera_perspective, int) or _camera_perspective<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] camera_perspective is invalid: {_camera_perspective}")
        return document_asl_consultant__to__target_video_utterance_token_map_tpl

      validated_results.append(
        (
          corpus_doc_id,
          asl_consultant_id,
          _target_video_fname,
          _camera_perspective,
          _utterance_seq_id,
          _token_seq_id,
          _token_id,
          _token_start_time,
          _token_end_time,
          _token_start_frame,
          _token_end_frame,
          frame_seq_paths,
          _n_existing_frame_images
        )
      )

  return validated_results


def get_target_video_frame_paths(document_asl_consultant_target_video_index_schemad_pcoll_row, d_pl_options):
  """
  document_asl_consultant_target_video_index_schemad_pcoll_row:
    beam.Row(
      DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
      DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
      ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
      ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
      CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][3]),                  
      TargetVideoFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][2])
    )
  """
  stitched_video_frames_dir = d_pl_options[fidscs_globals.OPT_NAME_STITCHED_VIDEO_FRAMES_DIR]
  target_video_frames_dir = fileio.path_join(stitched_video_frames_dir, document_asl_consultant_target_video_index_schemad_pcoll_row.TargetVideoFilename.split('.')[0])
  target_video_frame_paths = []
  if fileio.path_exists(target_video_frames_dir, d_pl_options)[0]:
    fs = FileSystems.get_filesystem(target_video_frames_dir)
    if type(fs) == GCSFileSystem:
      target_video_frame_paths = list(filter(lambda candidate_path: candidate_path.endswith(".jpg"), fileio.list_dir(target_video_frames_dir, d_pl_options, exclude_subdir=True)))
    else:
      target_video_frame_paths = [fileio.path_join(target_video_frames_dir, frame_fname) for frame_fname in fileio.list_dir(target_video_frames_dir, d_pl_options, exclude_subdir=True)]
    target_video_frame_paths = sorted(
      target_video_frame_paths, 
      key=lambda target_video_frame_path: int(target_video_frame_path.split(os.path.sep)[-1].split('.')[0])
    )
    
    # # target_video_frames = [(target_video_frame_path) for target_video_frame_path in target_video_frame_paths]
    # for target_video_frame_path in target_video_frame_paths:
    #   img = load_img(target_video_frame_path, target_size=fidscs_globals.FRAME_IMG_INPUT_SHAPE)  # this is a PIL image
    #   # img = load_img(target_video_frame_path)  # this is a PIL image
    #   img_array = img_to_array(img)                         
    #   # x = img_array.reshape((1,) + img_array.shape)
    #   # # Rescale by 1/255
    #   # x /= 255.0
    #   target_video_frames.append((target_video_frame_path, img_array))

  return [(
    (
      document_asl_consultant_target_video_index_schemad_pcoll_row.DocumentID,
      document_asl_consultant_target_video_index_schemad_pcoll_row.ASLConsultantID,
      document_asl_consultant_target_video_index_schemad_pcoll_row.CameraPerspective
    ),
    (
      document_asl_consultant_target_video_index_schemad_pcoll_row.TargetVideoFilename,
      target_video_frame_paths
    )
  )]

class TargetVideoFramePathGetter(beam__common.PipelinePcollElementProcessor):
  def __init__(self, d_pl_options):
    super(TargetVideoFramePathGetter, self).__init__(
      fn_pcoll_element_processor=get_target_video_frame_paths,
      kargs={'d_pl_options':d_pl_options},
      return_result=True
    )


def target_video_frame_image_to_bytes(document_asl_consultant_target_video_index_schemad_pcoll_row_tpl):
  """
  document_asl_consultant_target_video_index_schemad_pcoll_row_tpl:
    ((<corpus doc id>, <asl consultant id>, <camera perspective>), (<target video filename>, <target video frame seq id>, <target video frame path>))
  """
  corpus_doc_id = document_asl_consultant_target_video_index_schemad_pcoll_row_tpl[0][0]
  asl_consultant_id = document_asl_consultant_target_video_index_schemad_pcoll_row_tpl[0][1]
  camera_perspective = document_asl_consultant_target_video_index_schemad_pcoll_row_tpl[0][2]
  target_video_fname = document_asl_consultant_target_video_index_schemad_pcoll_row_tpl[1][0]
  frame_seq_id = document_asl_consultant_target_video_index_schemad_pcoll_row_tpl[1][1]
  frame_path = document_asl_consultant_target_video_index_schemad_pcoll_row_tpl[1][2]

  # # frame_tensor = img_to_array(load_img(frame_path, target_size=fidscs_globals.FRAME_IMG_INPUT_SHAPE))
  # img = load_img(frame_path, target_size=fidscs_globals.FRAME_IMG_INPUT_SHAPE)
  # bytesio = io.BytesIO()
  # img.save(bytesio, format='JPEG')
  # jpeg_bytes = bytesio.getvalue()

  # # now delete corresponding image file
  # fileio.delete_file(frame_path)
  # if fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__DEBUG:
  #   print(f"PROCESSED/DELETED target video frame: {frame_path}")

  return (
    (
      corpus_doc_id,
      asl_consultant_id,
      camera_perspective
    ),
    (
      target_video_fname,
      frame_seq_id,
      frame_path
      # , jpeg_bytes
    )
  )

def pl__7__create_document_asl_consultant_target_video_frame_index_schemad_pcoll(document_asl_consultant_target_video_index_schemad_pcoll, d_pl_options):
  """
  document_asl_consultant_target_video_index_schemad_pcoll:
    beam.Row(
      # SCHEMA_COL_NAMES__VIDEO_DS = [
      #   'DocumentID',
      #   'ASLConsultantID',
      #   'CameraPerspective',
      #   'TargetVideoFilename'
      # ]
      DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
      DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
      ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
      ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
      CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][3]),                  
      TargetVideoFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][2])
    )

  return schemad pcoll using:
    SCHEMA_COL_NAMES__VIDEO_FRAME_DS = [
      'DocumentID',
      'ASLConsultantID',
      'CameraPerspective',
      # 'TargetVideoFilename',
      'FrameSequence',
      'ImageTensor'
    ]
  """
  document_asl_consultant_target_video_frame_index_pcoll = (
    document_asl_consultant_target_video_index_schemad_pcoll
    | "Beam PL: extract ((<corpus doc id>, <asl consultant id>, <camera perspective>), (<target video filename>, listof(<target video frame path>))) "
      # "from document_asl_consultant_target_video_index_schemad_pcoll" >> beam.Map(get_target_video_frame_paths)
      "from document_asl_consultant_target_video_index_schemad_pcoll" >> beam.ParDo(TargetVideoFramePathGetter(d_pl_options))
    | "Beam PL: filter out ((<corpus doc id>, <asl consultant id>, <camera perspective>), (<target video filename>, listof(<target video frame path>))) with empty listof(<target video frame path>)" >> beam.Filter(
        lambda document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl: len(document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl[1])>0
      )
    | "Beam PL: 'explode' to ((<corpus doc id>, <asl consultant id>, <camera perspective>), (<target video filename>, <target video frame seq id>, <target video frame path>))" >> beam.FlatMap(
        lambda document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl: [
          (
            (
              document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl[0][0],   # <corpus doc id>
              document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl[0][1],   # <asl consultant id>
              document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl[0][2]    # <camera perspective>
            ),
            (
              document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl[1][0],
              target_video_frame_seq_id,
              target_video_frame_path
            )
          ) for target_video_frame_seq_id, target_video_frame_path in enumerate(document_asl_consultant_target_video_frame_index_schemad_pcoll_row_tpl[1][1])
        ]
      )
    | "Beam PL: select distict ((<corpus doc id>, <asl consultant id>, <camera perspective>), (<target video filename>, <target video frame seq id>, <target video frame path>))" >> beam.Distinct()
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_frame_index_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document_asl_consultant_target_video_frame_index_pcoll entry"))
  )
  sorted_document_asl_consultant_target_video_frame_index_pcoll = beam__common.pl__X__sort_pcoll(
    document_asl_consultant_target_video_frame_index_pcoll, 
    pcoll_label="document_asl_consultant_target_video_frame_index_pcoll"
  )
  document_asl_consultant_target_video_frame_index_schemad_pcoll = (
    sorted_document_asl_consultant_target_video_frame_index_pcoll # rows of ((<corpus doc id>, <asl consultant id>, <camera perspective>), (<target video filename>, <target video frame seq id>, <target video frame path>))
      | "Beam PL: read target video frame to tensor" >> beam.Map(target_video_frame_image_to_bytes)
      # the above outputs tuples of the form:
      # ((<corpus doc id>, <asl consultant id>, <camera perspective>), (<target video filename>, <target video frame seq id>, <target video frame path>, <target video frame jpeg bytes>))
      | "Beam PL: apply schema to create final document_asl_consultant_target_video_frame_index_schemad_pcoll" >> beam.Map(
        lambda sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_FRAME_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'TargetVideoFilename',
          #   'FrameSequence',
          #   'JPEGBytes'
          # ]
          DocumentID=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][0]),
          ASLConsultantID=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][1]),
          CameraPerspective=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][2]),
          TargetVideoFilename=str(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][0]),
          FrameSequence=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][1]),
          FramePath=str(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][2])
          # , JPEGBytes=sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][3]
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_frame_index_schemad_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document_asl_consultant_target_video_frame_index_schemad_pcoll entry"))
  )

  return document_asl_consultant_target_video_frame_index_schemad_pcoll


def pl__8__write_document_asl_consultant_target_video_frame_index_schemad_pcoll(document_asl_consultant_target_video_frame_index_schemad_pcoll, d_pl_options):
  document_asl_consultant_target_video_frame_index_csv_rows = (
    document_asl_consultant_target_video_frame_index_schemad_pcoll
    # | "Beam PL: extract SCHEMA_COL_NAMES__VIDEO_FRAME_DS only from document_asl_consultant_target_video_frame_index_schemad_pcoll" >> beam.Map(
    #     lambda document_asl_consultant_target_video_frame_index_schemad_pcoll_row: beam.Row(
    #       DocumentID=document_asl_consultant_target_video_frame_index_schemad_pcoll_row.DocumentID,
    #       ASLConsultantID=document_asl_consultant_target_video_frame_index_schemad_pcoll_row.ASLConsultantID,
    #       CameraPerspective=document_asl_consultant_target_video_frame_index_schemad_pcoll_row.CameraPerspective,
    #       TargetVideoFilename=document_asl_consultant_target_video_frame_index_schemad_pcoll_row.TargetVideoFilename,
    #       FrameSequence=document_asl_consultant_target_video_frame_index_schemad_pcoll_row.FrameSequence
    #       # , JPEGBytes=document_asl_consultant_target_video_frame_index_schemad_pcoll_row.JPEGBytes
    #     )
    #   )
    | beam.Map(lambda document_asl_consultant_target_video_frame_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(document_asl_consultant_target_video_frame_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    document_asl_consultant_target_video_frame_index_csv_rows, 
    "DOCUMENT-ASLCONSULTANT-TARGETVIDEO-FRAME-INDEX", 
    fidscs_globals.VIDEO_FRAME_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__VIDEO_FRAME_DS, 
    d_pl_options
  ) # target_video_frame_index_csv_path


def validate_preprocess_document_asl_consultant__to__target_video_utterance_token_map_tpl(document_asl_consultant__to__target_video_utterance_token_map_tpl):
  # document_asl_consultant__to__target_video_utterance_token_map_tpl:
    # (
    #   (<corpus doc id>, <asl consultant id>), # key
    #   {
    #     'target_video_map': [(<target video fname>, <camera perspective>)], # there may be up to three (corresponding to camera perspective)
    #     'utterance_token_map': [(<utterance seq id>, <token seq id>, <token id>, <token start time>, <token end time>)] # there will be many (corresponding to each utterance)
    #   }
    # )

  validated_results = []

  key = document_asl_consultant__to__target_video_utterance_token_map_tpl[0]

  corpus_doc_id = key[0]
  if corpus_doc_id is None or not isinstance(corpus_doc_id, int) or corpus_doc_id<0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} corpus doc id is invalid: {corpus_doc_id}")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  asl_consultant_id = key[1]
  if asl_consultant_id is None or not isinstance(asl_consultant_id, int) or asl_consultant_id<0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} asl consultant id is invalid: {asl_consultant_id}")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  target_video_map = document_asl_consultant__to__target_video_utterance_token_map_tpl[1]['target_video_map']
  if len(target_video_map)==0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} {key} target_video_map is empty!")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  utterance_token_map = document_asl_consultant__to__target_video_utterance_token_map_tpl[1]['utterance_token_map']
  if len(utterance_token_map)==0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} {key} utterance_token_map is empty!")
    return document_asl_consultant__to__target_video_utterance_token_map_tpl

  for i, utterance_token_map_instance in enumerate(utterance_token_map):
    # (<utterance seq id>, <token seq id>, <token id>, <token start time>, <token end time>)
    _utterance_seq_id = utterance_token_map_instance[0]
    if _utterance_seq_id is None or not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] utterance_seq_id is invalid: {_utterance_seq_id}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_seq_id = utterance_token_map_instance[1]
    if _token_seq_id is None or not isinstance(_token_seq_id, int) or _token_seq_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_seq_id is invalid: {_token_seq_id}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_id = utterance_token_map_instance[2]
    if _token_id is None or not isinstance(_token_id, int) or _token_id<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_id is invalid: {_token_id}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_start_time = utterance_token_map_instance[3]
    if _token_start_time is None or not isinstance(_token_start_time, int) or _token_start_time<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_start_time is invalid: {_token_start_time}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl
    _token_end_time = utterance_token_map_instance[4]
    if _token_end_time is None or not isinstance(_token_end_time, int) or _token_end_time<0:
      print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] token_end_time is invalid: {_token_end_time}")
      return document_asl_consultant__to__target_video_utterance_token_map_tpl

    for j, target_video_map_instance in enumerate(target_video_map):
      # (<target video fname>, <camera perspective>)
      _target_video_fname = target_video_map_instance[0]
      if _target_video_fname is None or len(_target_video_fname)==0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname is invalid: {_target_video_fname}")
        return document_asl_consultant__to__target_video_utterance_token_map_tpl

      target_video_frames_dir = fileio.path_join(fidscs_globals.STICHED_VIDEO_FRAMES_DIR, _target_video_fname.split('.')[0])
      _n_existing_frame_images = -1 if not fileio.path_exists(target_video_frames_dir)[0] else len(fileio.list_dir(target_video_frames_dir))
      # if _n_existing_frame_images == -1:
      #   print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document_asl_consultant__to__target_video_utterance_token_map_tpl utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} frames dir ({target_video_frames_dir}) does not exist!")
      _token_end_frame = _token_start_frame = -1
      frame_seq_paths = []
      if _n_existing_frame_images > 0:
        _token_start_frame = int(round(_token_start_time/1000.0*fidscs_globals.FPS))
        _token_end_frame = int(round(_token_end_time/1000.0*fidscs_globals.FPS))+1
        n_frames = _token_end_frame-_token_start_frame
        last_frame_idx = (_n_existing_frame_images-1)

        if _token_start_frame > last_frame_idx:
          # comment out for now
          # print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} _token_start_frame ({_token_start_frame}) > _n_existing_frame_images ({_n_existing_frame_images}): reconciling bounds of {n_frames} frames (from last frame index {last_frame_idx}) to {(last_frame_idx-(n_frames-1), last_frame_idx)}")
          # return document_asl_consultant__to__target_video_utterance_token_map_tpl
          # readjust bounds from last_frame_idx going backwards
          _token_start_frame = last_frame_idx-(n_frames-1)
          _token_end_frame = last_frame_idx
        else:
          if _token_end_frame > last_frame_idx:
            # print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} _token_end_frame ({_token_end_frame}) > _n_existing_frame_images ({_n_existing_frame_images}): reconciling _token_end_frame to {last_frame_idx}")
            # return document_asl_consultant__to__target_video_utterance_token_map_tpl
            # take all that is available to the end
            _token_end_frame = last_frame_idx

        if _token_start_frame <= last_frame_idx and _token_end_frame <= last_frame_idx:
          for frame_idx in range(_token_start_frame, _token_end_frame+1):
            frame_path = fileio.path_join(target_video_frames_dir, f"{frame_idx}.jpg")
            if fileio.path_exists(frame_path)[0]:
              frame_seq_paths.append(frame_path)
            else:
              print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname}: failed to reconcile invalid requested frame bounds {(_token_start_frame, _token_end_frame)} (valid bounds are: {(0, last_frame_idx)})")
              # return document_asl_consultant__to__target_video_utterance_token_map_tpl
        else:
          if fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
            print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] target_video_fname {_target_video_fname} _token_end_frame ({_token_end_frame}) > _n_existing_frame_images ({_n_existing_frame_images}): reconciling _token_end_frame to {last_frame_idx}")

      _camera_perspective = target_video_map_instance[1]
      if _camera_perspective is None or not isinstance(_camera_perspective, int) or _camera_perspective<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} utterance_token_map[{i}] target_video_map_instance[{j}] camera_perspective is invalid: {_camera_perspective}")
        return document_asl_consultant__to__target_video_utterance_token_map_tpl

      validated_results.append(
        (
          corpus_doc_id,
          asl_consultant_id,
          _target_video_fname,
          _camera_perspective,
          _utterance_seq_id,
          _token_seq_id,
          _token_id,
          _token_start_time,
          _token_end_time,
          _token_start_frame,
          _token_end_frame,
          frame_seq_paths,
          _n_existing_frame_images
        )
      )

  return validated_results


def pl__8__create_document_asl_consultant_utterance_token_frame_index_schemad_pcoll(
  document_asl_consultant_utterance_token_index_schemad_pcoll,
  document_asl_consultant_target_video_index_schemad_pcoll,
  document_asl_consultant_target_video_frame_index_schemad_pcoll,
  d_pl_options
):
  """
  document_asl_consultant_utterance_token_index_schemad_pcoll:
    beam.Row(
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

  document_asl_consultant_target_video_index_schemad_pcoll:
    beam.Row(
      # SCHEMA_COL_NAMES__VIDEO_DS = [
      #   'DocumentID',
      #   'ASLConsultantID',
      #   'CameraPerspective',
      #   'TargetVideoFilename'
      # ]
      DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
      DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
      ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
      ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
      UtteranceSequence=int(document_asl_consultant_video_index_pcoll_row_tpl[1][2]),
      CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][4]),                  
      TargetVideoFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][3])
    )

  document_asl_consultant_target_video_frame_index_schemad_pcoll
    beam.Row(
      DocumentID=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][0]),
      ASLConsultantID=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][1]),
      CameraPerspective=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][2]),
      TargetVideoFilename=str(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][0]),
      FrameSequence=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][1]),
      FramePath=str(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][2]),
      UtteranceSequence
    )

  return schemad pcoll using:
    SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS = [
      'DocumentID',
      'ASLConsultantID',
      'CameraPerspective',
      'TargetVideoFilename',
      'UtteranceSequence',
      'TokenSequence',
      'FrameSequence',
      'TokenID'
    ]
  """
  doc_consultant_utterance__to__target_video_map = (
    document_asl_consultant_target_video_index_schemad_pcoll
    | "Beam PL: extract ((<doc id>, <asl consultant id>, <utterance seq id>), (<target vid fname>, <camera perspective>))" >> beam.Map(
        lambda document_asl_consultant_target_video_index_schemad_pcoll_row: (
          (
            document_asl_consultant_target_video_index_schemad_pcoll_row.DocumentID,
            document_asl_consultant_target_video_index_schemad_pcoll_row.ASLConsultantID,
            document_asl_consultant_target_video_index_schemad_pcoll_row.UtteranceSequence
          ),
          (
            document_asl_consultant_target_video_index_schemad_pcoll_row.TargetVideoFilename,
            document_asl_consultant_target_video_index_schemad_pcoll_row.CameraPerspective
          )
        )
      )
    | "Beam PL: select distinct ((<doc id>, <asl consultant id>, <utterance seq id>), (<target vid fname>, <camera perspective>))" >> beam.Distinct()
  )

  doc_consultant_utterance__to__token_map = (
    document_asl_consultant_utterance_token_index_schemad_pcoll
    | "Beam PL: extract ((<doc id>, <asl consultant id>, <utterance seq id>), (<tok vocab id>, <tok seq id>, <tok start time>, <tok end time>, listof(<frame seq id>))) from document_asl_consultant_utterance_token_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_utterance_token_index_schemad_pcoll_row: (
          (
            document_asl_consultant_utterance_token_index_schemad_pcoll_row.DocumentID,
            document_asl_consultant_utterance_token_index_schemad_pcoll_row.ASLConsultantID,
            document_asl_consultant_utterance_token_index_schemad_pcoll_row.UtteranceSequence
          ),
          (
            document_asl_consultant_utterance_token_index_schemad_pcoll_row.TokenID,
            document_asl_consultant_utterance_token_index_schemad_pcoll_row.TokenSequence,
            document_asl_consultant_utterance_token_index_schemad_pcoll_row.StartTime,
            document_asl_consultant_utterance_token_index_schemad_pcoll_row.EndTime,
            [frame_seq_id for frame_seq_id in range(
                int(round(document_asl_consultant_utterance_token_index_schemad_pcoll_row.StartTime/1000.0*fidscs_globals.FPS)), # float(float(document_asl_consultant_utterance_token_index_schemad_pcoll_row.StartTime)/1000.0)*fidscs_globals.FPS
                int(round(document_asl_consultant_utterance_token_index_schemad_pcoll_row.EndTime/1000.0*fidscs_globals.FPS))    # float(float(document_asl_consultant_utterance_token_index_schemad_pcoll_row.EndTime)/1000.0)*fidscs_globals.FPS
              )
            ]
          )
        )
      )
    # debug
    # | "Beam PL: print doc_consultant__to__utterance_token_map" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="doc_consultant__to__utterance_token_map entry"))
  )

  doc_consultant_utterance__to__target_video_token_map = (
    ({
      'target_video_map': doc_consultant_utterance__to__target_video_map,
      'token_map': doc_consultant_utterance__to__token_map
    })
    | "Beam PL: merge target_video_map and token_map" >> beam.CoGroupByKey()
    # the above produces tuples of the form:
      # (
      #   (<doc id>, <asl consultant id>, <utterance seq id>), # key
      #   {
      #     'target_video_map': [(<target vid fname>, <camera perspective>)], # up to three, corresponding to camera perspective
      #     'token_map': [(<tok vocab id>, <tok seq id>, <tok start time>, <tok end time>, listof(<frame seq id>))] # there will be many, corresponding to token sequence id for this utterance
      #   }
      # )
    | "Beam PL: sort doc_consultant_utterance__to__target_video_token_map token_map based on <tok seq id>" >> beam.Map(
        lambda doc_consultant_utterance__to__target_video_token_map_tpl: (
          (
            doc_consultant_utterance__to__target_video_token_map_tpl[0][0], # <doc id>
            doc_consultant_utterance__to__target_video_token_map_tpl[0][1], # <asl consultant id>
            doc_consultant_utterance__to__target_video_token_map_tpl[0][2]  # <utterance seq id>
          ),
          {
            'target_video_map': sorted(
              doc_consultant_utterance__to__target_video_token_map_tpl[1]['target_video_map'],
              key=lambda target_vid_info_tpl: target_vid_info_tpl[1] # <camera perspective>
            ),
            'token_map': sorted(
              doc_consultant_utterance__to__target_video_token_map_tpl[1]['token_map'], 
              key=lambda tok_info_tpl: tok_info_tpl[1] # <tok seq id>
            )
          }
        )
      )
    # debug
    # | "Beam PL: print doc_consultant_utterance__to__target_video_token_map" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="doc_consultant_utterance__to__target_video_token_map entry"))
  )

  doc_consultant_utterance__to__target_video_token_map__target_video_map_lte_MAX_CAMERA_PERSPECTIVES = (
    doc_consultant_utterance__to__target_video_token_map
    | f"filter doc_consultant_utterance__to__target_video_token_map_tpls with len(target_video_map)<={fidscs_globals.MAX_CAMERA_PERSPECTIVES}" >> beam.Filter(
        lambda doc_consultant_utterance__to__target_video_token_map_tpl: len(doc_consultant_utterance__to__target_video_token_map_tpl[1]['target_video_map'])<=fidscs_globals.MAX_CAMERA_PERSPECTIVES
      )
    # debug
    # | "Beam PL: print doc_consultant_utterance__to__target_video_token_map__target_video_map_lte_3" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="doc_consultant_utterance__to__target_video_token_map__target_video_map_lte_3 entry"))
  )

  # debug
  #   we have a problem if doc_consultant_utterance__to__target_video_token_map__target_video_map_gt_MAX_CAMERA_PERSPECTIVES is non-empty!
  doc_consultant_utterance__to__target_video_token_map__target_video_map_gt_MAX_CAMERA_PERSPECTIVES = (
    doc_consultant_utterance__to__target_video_token_map
    | f"filter doc_consultant_utterance__to__target_video_token_map_tpls with len(target_video_map)>{fidscs_globals.MAX_CAMERA_PERSPECTIVES}" >> beam.Filter(
        lambda doc_consultant_utterance__to__target_video_token_map_tpl: len(doc_consultant_utterance__to__target_video_token_map_tpl[1]['target_video_map'])>fidscs_globals.MAX_CAMERA_PERSPECTIVES
      )
    # debug
    # | "Beam PL: print doc_consultant_utterance__to__target_video_token_map__target_video_map_gt_MAX_CAMERA_PERSPECTIVES" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg=f"{fidscs_globals.VALIDATION_WARNING_TEXT} len(target_video_map)>{fidscs_globals.MAX_CAMERA_PERSPECTIVES} entry"))
  )

  # now transform doc_consultant_utterance__to__target_video_token_map tuples
    # from:
      # (
      #   (<doc id>, <asl consultant id>, <utterance seq id>), # key
      #   {
      #     'target_video_map': [(<target vid fname>, <camera perspective>)], # up to three, corresponding to camera perspective
      #     'token_map': [(<tok vocab id>, <tok seq id>, <tok start time>, <tok end time>, listof(<frame seq id>))] # there will be many, corresponding to token sequence id for this utterance
      #   }
      # )
    # to:
      # (
      #   (<doc id>, <asl consultant id>, <utterance seq id>, <tok seq id>), # key
      #   (
      #     <tok vocab id>,
      #     <tok start time>,
      #     <tok end time>,
      #     listof(<frame seq id>),
      #     listof((<target vid fname>, <camera perspective>))
      #   )
      # )
    
  document_asl_consultant_utterance_token_frame_index_tpl_pcoll = (
    doc_consultant_utterance__to__target_video_token_map__target_video_map_lte_MAX_CAMERA_PERSPECTIVES
    | "Beam PL: transform doc_consultant_utterance__to__target_video_token_map__target_video_map_lte_MAX_CAMERA_PERSPECTIVES" >> beam.Map(
        lambda doc_consultant_utterance__to__target_video_token_map_tpl: [
          (
            (
              doc_consultant_utterance__to__target_video_token_map_tpl[0][0],                   # <doc id>
              doc_consultant_utterance__to__target_video_token_map_tpl[0][1],                   # <asl consultant id>
              doc_consultant_utterance__to__target_video_token_map_tpl[0][2],                   # <utterance seq id>
              token_info_tpl[1]                                                                 # <tok seq id>
            ),
            (
              token_info_tpl[0],                                                                # <tok vocab id>
              token_info_tpl[2],                                                                # <tok start time>
              token_info_tpl[3],                                                                # <tok end time>
              token_info_tpl[4],                                                                # listof(<frame seq id>)
              doc_consultant_utterance__to__target_video_token_map_tpl[1]['target_video_map'],  # listof((<target vid fname>, <camera perspective>))
            )
          ) for token_info_tpl in doc_consultant_utterance__to__target_video_token_map_tpl[1]['token_map']
        ]
      )
    | "Beam PL: 'explode' list of doc_consultant_utterance__to__target_video_token_map_tpl" >> beam.FlatMap(
        lambda list_doc_consultant_utterance__to__target_video_token_map_tpl: list_doc_consultant_utterance__to__target_video_token_map_tpl
      )
    # debug
    # | "Beam PL: print document_asl_consultant_utterance_token_frame_index_tpl_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document_asl_consultant_utterance_token_frame_index_tpl_pcoll entry"))
  )
  document_asl_consultant_utterance_token_frame_index_tpl_pcoll = beam__common.pl__X__sort_pcoll(
    document_asl_consultant_utterance_token_frame_index_tpl_pcoll, 
    pcoll_label="document_asl_consultant_utterance_token_frame_index_tpl_pcoll"
  )

  doc_consultant_target_video_frame__to__utterance_token_map = (
    document_asl_consultant_utterance_token_frame_index_tpl_pcoll
    | "Beam PL: 'explode' to list of ((<doc id>,<asl consultant id>,<camera perspective>,<target vid fname>,<utterance seq id>,<tok seq id>)),(listof(<frame seq id>),<tok vocab id>)) from document_asl_consultant_utterance_token_frame_index_tpl_pcoll" >> beam.Map(
        lambda document_asl_consultant_utterance_token_frame_index_tpl: [
          (
            (
              document_asl_consultant_utterance_token_frame_index_tpl[0][0],  # <doc id>
              document_asl_consultant_utterance_token_frame_index_tpl[0][1],  # <asl consultant id>
              target_vid_cam_persp[1],                                        # <camera perspective>
              target_vid_cam_persp[0],                                        # <target vid fname>
              document_asl_consultant_utterance_token_frame_index_tpl[0][2],  # <utterance seq id>
              document_asl_consultant_utterance_token_frame_index_tpl[0][3]   # <tok seq id>
            ),
            (
              document_asl_consultant_utterance_token_frame_index_tpl[1][3],   # listof(<frame seq id>)
              document_asl_consultant_utterance_token_frame_index_tpl[1][0]   # <tok vocab id>
            )
          # ) for target_vid_frame_seq_tpl in zip(document_asl_consultant_utterance_token_frame_index_tpl[1][4], document_asl_consultant_utterance_token_frame_index_tpl[1][3])
          ) for target_vid_cam_persp in document_asl_consultant_utterance_token_frame_index_tpl[1][4] # listof((<target vid fname>, <camera perspective>)) 
        ] 
      )
    | "Beam PL: 'explode' list of ((<doc id>,<asl consultant id>,<camera perspective>,<target vid fname>,<utterance seq id>,<tok seq id>)),(listof(<frame seq id>),<tok vocab id>)) tuples" >> beam.FlatMap(
        lambda list_doc_consultant_target_video_utterance_token__to__list_of_frame_seq_map_tpl: list_doc_consultant_target_video_utterance_token__to__list_of_frame_seq_map_tpl
      )
    | "Beam PL: 'explode' to list of (<doc id>,<asl consultant id>,<camera perspective>,<target vid fname>,<utterance seq id>,<tok seq id>,<frame seq id>,<tok vocab id>) from list_doc_consultant_target_video_utterance_token__to__list_of_frame_seq_map_tpl" >> beam.Map(
        lambda tpl: [
          (
            (
              tpl[0][0],      # <doc id>
              tpl[0][1],      # <asl consultant id>
              tpl[0][2],      # <camera perspective>
              tpl[0][3],      # <target vid fname>
              frame_seq_id    # <frame seq id>
            ),
            (
              tpl[0][4],      # <utterance seq id>
              tpl[0][5],      # <tok seq id>
              tpl[1][1]       # <tok vocab id>
            )
          ) for frame_seq_id in tpl[1][0]
        ]
      )
    | "Beam PL: 'explode' list of tuples" >> beam.FlatMap(lambda list_tpl: list_tpl)
    # debug
    # | "Beam PL: print doc_consultant_target_video_frame__to__utterance_token_map" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="doc_consultant_target_video_frame__to__utterance_token_map entry"))
  )

  # document_asl_consultant_target_video_frame_index_schemad_pcoll
    # beam.Row(
    #   DocumentID=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][0]),
    #   ASLConsultantID=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][1]),
    #   CameraPerspective=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[0][2]),
    #   TargetVideoFilename=str(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][0]),
    #   FrameSequence=int(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][1]),
    #   FramePath=str(sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][2])
    #   # , JPEGBytes=sorted_document_asl_consultant_target_video_frame_index_pcoll_row_tpl[1][3]
    # )
  existing_target_video_frames_pcoll = (
    document_asl_consultant_target_video_frame_index_schemad_pcoll
    | "Beam PL: extract ((<doc id>, <asl consultant id>, <camera perspective>, <target vid fname>, <frame seq id>), '<TARGET VIDEO FRAME EXISTS>') from document_asl_consultant_target_video_frame_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_target_video_frame_index_schemad_pcoll_row: (
          (
            document_asl_consultant_target_video_frame_index_schemad_pcoll_row.DocumentID,
            document_asl_consultant_target_video_frame_index_schemad_pcoll_row.ASLConsultantID,
            document_asl_consultant_target_video_frame_index_schemad_pcoll_row.CameraPerspective,
            document_asl_consultant_target_video_frame_index_schemad_pcoll_row.TargetVideoFilename,
            document_asl_consultant_target_video_frame_index_schemad_pcoll_row.FrameSequence
          ),
          '<TARGET VIDEO FRAME EXISTS>'
        )
      )
    # debug
    # | "Beam PL: print existing_target_video_frames_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="existing_target_video_frames_pcoll entry"))
  )

  doc_consultant_target_video_frame__to__target_video_token_map = (
    ({
      'target_video_frame_exists': existing_target_video_frames_pcoll,
      'existing_utterance_token_map': doc_consultant_target_video_frame__to__utterance_token_map
    })
    | "Beam PL: merge existing_target_video_frames_map and utterance_token_map" >> beam.CoGroupByKey()
    # the above produces tuples of the form:
      # (
      #   (<doc id>, <asl consultant id>, <camera perspective>, <target vid fname>, <frame seq id>), # key
      #   {
      #     'target_video_frame_exists': ['<TARGET VIDEO FRAME EXISTS>'] | [], # if empty, then the frame does not exist in document_asl_consultant_target_video_frame_index_schemad_pcoll
      #     'utterance_token_map': [(<utterance seq id>, <tok seq id>, <tok vocab id>)]
      #   }
      # )
    # debug
    # | "Beam PL: print doc_consultant_target_video_frame__to__target_video_token_map" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="doc_consultant_target_video_frame__to__target_video_token_map entry"))
  )

  doc_consultant_target_video_frame__to__target_video_token_map__non_existing_frame = (
    doc_consultant_target_video_frame__to__target_video_token_map
    | "Beam PL: filter non-existing frames (in document_asl_consultant_target_video_frame_index_schemad_pcoll)" >> beam.Filter(
        lambda doc_consultant_target_video_frame__to__target_video_token_map_tpl: len(doc_consultant_target_video_frame__to__target_video_token_map_tpl[1]['target_video_frame_exists'])==0
      )
  )
  doc_consultant_target_video_frame__to__target_video_token_map__existing_frame = (
    doc_consultant_target_video_frame__to__target_video_token_map
    | "Beam PL: filter existing frames (in document_asl_consultant_target_video_frame_index_schemad_pcoll)" >> beam.Filter(
        lambda doc_consultant_target_video_frame__to__target_video_token_map_tpl: len(doc_consultant_target_video_frame__to__target_video_token_map_tpl[1]['target_video_frame_exists'])>0
      )
    # debug
    # | "Beam PL: print doc_consultant_target_video_frame__to__target_video_token_map__existing_frame" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="doc_consultant_target_video_frame__to__target_video_token_map__existing_frame entry"))
  )

  doc_consultant_target_video_frame__to__target_video_token_map__existing_frame__non_existing_utterance_token = (
    doc_consultant_target_video_frame__to__target_video_token_map__existing_frame
    | "Beam PL: filter non-existing record keys (in doc_consultant_target_video_frame__to__utterance_token_map)" >> beam.Filter(
        lambda doc_consultant_target_video_frame__to__target_video_token_map__existing_frame_tpl: len(doc_consultant_target_video_frame__to__target_video_token_map__existing_frame_tpl[1]['existing_utterance_token_map'])==0
      )
  )
  document_asl_consultant_target_video_utterance_token_frame_index_pcoll = (
    doc_consultant_target_video_frame__to__target_video_token_map__existing_frame
    | "Beam PL: filter existing record keys (in doc_consultant_target_video_frame__to__utterance_token_map)" >> beam.Filter(
        lambda doc_consultant_target_video_frame__to__target_video_token_map__existing_frame_tpl: len(doc_consultant_target_video_frame__to__target_video_token_map__existing_frame_tpl[1]['existing_utterance_token_map'])>0
      )
    | "Beam PL: extract ((<doc id>, <asl consultant id>, <camera perspective>, <target vid fname>, <utterance seq id>, <token seq id>), (<token vocab id>, <frame seq id>)) from doc_consultant_target_video_frame__to__target_video_token_map" >> beam.Map(
        # (
        #   (<doc id>, <asl consultant id>, <camera perspective>, <target vid fname>, <frame seq id>), # key
        #   {
        #     'target_video_frame_exists': ['<TARGET VIDEO FRAME EXISTS>'] | [], # if empty, then the frame does not exist in document_asl_consultant_target_video_frame_index_schemad_pcoll
        #     'utterance_token_map': [(<utterance seq id>, <tok seq id>, <tok vocab id>)]
        #   }
        # )
        lambda doc_consultant_target_video_frame__to__target_video_token_map_tpl: [
          (
            (
              doc_consultant_target_video_frame__to__target_video_token_map_tpl[0][0],  # <doc id>
              doc_consultant_target_video_frame__to__target_video_token_map_tpl[0][1],  # <asl consultant id>
              doc_consultant_target_video_frame__to__target_video_token_map_tpl[0][2],  # <camera perspective>
              doc_consultant_target_video_frame__to__target_video_token_map_tpl[0][3],  # <target vid fname>
              utterance_token_tpl[0],                                                   # <utterance seq id>
              utterance_token_tpl[1],                                                   # <token seq id>
              doc_consultant_target_video_frame__to__target_video_token_map_tpl[0][4],  # <frame seq id>
            ), # key
            utterance_token_tpl[2]                                                      # <tok vocab id>
          ) for utterance_token_tpl in doc_consultant_target_video_frame__to__target_video_token_map_tpl[1]['existing_utterance_token_map']
        ]
      )
    | "Beam PL: 'explode' list of document_asl_consultant_target_video_utterance_token_frame_index tuples" >> beam.FlatMap(
        lambda list_document_asl_consultant_target_video_utterance_token_frame_index_tpl: list_document_asl_consultant_target_video_utterance_token_frame_index_tpl
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_utterance_token_frame_index_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document_asl_consultant_target_video_utterance_token_frame_index_pcoll entry"))
  )

  sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll = beam__common.pl__X__sort_pcoll(
    document_asl_consultant_target_video_utterance_token_frame_index_pcoll, 
    pcoll_label="document_asl_consultant_target_video_utterance_token_frame_index_pcoll"
  )
  document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll = (
    sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll
    | "Beam PL: apply schema to create final document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll" >> beam.Map(
        lambda sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll: beam.Row(
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
          DocumentID=int(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[0][0]),
          ASLConsultantID=int(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[0][1]),
          CameraPerspective=int(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[0][2]),
          TargetVideoFilename=str(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[0][3]),
          UtteranceSequence=int(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[0][4]),
          TokenSequence=int(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[0][5]),
          FrameSequence=int(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[0][6]),
          TokenID=int(sorted_document_asl_consultant_target_video_utterance_token_frame_index_pcoll[1])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll entry"))
  )

  return document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll


def pl__9__write_document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll(document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll, d_pl_options):
  document_asl_consultant_target_video_utterance_token_frame_index_csv_rows = (
    document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll
    | beam.Map(lambda document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    document_asl_consultant_target_video_utterance_token_frame_index_csv_rows, 
    "DOCUMENT-ASLCONSULTANT-TARGETVIDEO-UTTERANCE-TOKEN-FRAME-INDEX", 
    fidscs_globals.UTTERANCE_TOKEN_FRAME_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_TOKEN_FRAME_DS, 
    d_pl_options
  ) # document_asl_consultant_target_video_utterance_token_frame_index_csv_path


def validate_preprocess_doc_participant_to_utterance_video_cameraperspective_mapping(tvftmdputftvimt):
  """
  tvftmdputftvimt (abbreviation for target_video_fname_to_merged_doc_participant_utterance_target_full_target_vid_index_mapping_tpl):
    (
      <media fname>, # key
      {
        'target_video_doc_participant_utterance_mapping': [
          (
            <corpus doc filename>, 
            <participant_name>, 
            <utterance seq id>
          )
        ], 
        'target_full_target_vid_index_mapping': [
          beam.Row(
            target_video_filename=<target_video_filename>, 
            video_seq_id=<video_seq_id>, 
            perspective_cam_id=<perspective_cam_id>, 
            compressed_mov_url=<compressed_mov_url>, 
            compressed_mov_url=<compressed_mov_url>, 
            uncompressed_avi_url=<uncompressed_avi_url>, 
            uncompressed_avi_mirror_1_url=<uncompressed_avi_mirror_1_url>, 
            <uncompressed_avi_mirror_2_url>
          )
        ]
      }
    )

  return:
    listof(
      ((<document fname>, <participant name>), (<utterance seq id>, <media fname>, <camera perspective>))
    )
  """
  target_video_fname = tvftmdputftvimt[0]
  target_video_doc_participant_utterance_mapping = tvftmdputftvimt[1]['target_video_doc_participant_utterance_mapping']
  target_full_target_vid_index_mapping = tvftmdputftvimt[1]['target_full_target_vid_index_mapping']

  validated_results = []

  # there should always only be ONE camera perspective per video_fname file
  camera_perspective = None
  if len(target_full_target_vid_index_mapping) > 0:
    not_unique = False
    for full_target_vid_index_pcoll_row in target_full_target_vid_index_mapping:
      # _camera_perspective = full_target_vid_index_pcoll_row['CameraPerspective']
      _camera_perspective = full_target_vid_index_pcoll_row.perspective_cam_id
      if camera_perspective is None:
        camera_perspective = _camera_perspective
      else:
        if _camera_perspective != camera_perspective:
          not_unique = True
          print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} target video {target_video_fname} camera perspective not unique! It has camera perspectives: {camera_perspective} and {_camera_perspective}")
          break
  else:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} target video {target_video_fname} has no associated camera perspective!")
  
  doc_fname = None
  participant_name = None
  utterance_seq_id = None
  if len(target_video_doc_participant_utterance_mapping) > 0:
    multiple_docs = []
    multiple_participants = []
    multiple_utterances = []
    for target_video_doc_participant_utterance_mapping_instance in target_video_doc_participant_utterance_mapping:
      _doc_fname = target_video_doc_participant_utterance_mapping_instance[0]
      _participant_name = target_video_doc_participant_utterance_mapping_instance[1]
      _utterance_seq_id = target_video_doc_participant_utterance_mapping_instance[2]

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
          (_utterance_seq_id, target_video_fname, camera_perspective)
        )
      )
    multiple_docs = set(multiple_docs)
    if len(multiple_docs) > 1 and fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
      print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} target video {target_video_fname} document occurrence is not unique! It occurs in documents: {multiple_docs}")

    multiple_participants = set(multiple_participants)
    if len(multiple_participants) > 1 and fidscs_globals.OUTPUT_INFO_LEVEL <= fidscs_globals.OUTPUT_INFO_LEVEL__WARNING:
      print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} target video {target_video_fname} participant occurrence is not unique! It has participants: {multiple_participants}")
    # if len(multiple_utterances) > 1: # this is actually expected
    #   print(f"{fidscs_globals.VALIDATION_WARNING_TEXT} target video {target_video_fname} utterance seq id occurrence is not unique! It has utterance seq ids: {multiple_utterances}")

  else:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} target video {target_video_fname} is not associated with a corpus document!")
    validated_results.append(
      (None, participant_name),
      (utterance_seq_id, target_video_fname, camera_perspective)
    )

  return validated_results


def validate_preprocess_video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl(vdpucpwiprt):
  """
  vdpucpwiprt (video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl):
    (
      (<doc filename>, <participant name>),           # key
      {
        'document_participant_with_ids_mapping': [
          (<doc id>, <asl consultant id>)
        ], 
        'doc_participant_to_utterance_video_cameraperspective_mapping': [
          (<utterance seq id>, <video fname>, <camera perspective>)
        ]
      }
    )

  return:
    listof(
      ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <utterance seq id>, <media filename>, <camera perspective>))
    )
  """
  doc_fname = vdpucpwiprt[0][0]
  if doc_fname is None or len(doc_fname)==0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl {vdpucpwiprt} has no associated corpus document filename")
    return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows

  participant_name = vdpucpwiprt[0][1]
  if participant_name is None or len(participant_name)==0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl {vdpucpwiprt} has no associated participant name")
    return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows

  document_participant_with_ids_mapping = vdpucpwiprt[1]['document_participant_with_ids_mapping']
  doc_participant_to_utterance_video_cameraperspective_mapping = vdpucpwiprt[1]['doc_participant_to_utterance_video_cameraperspective_mapping']

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
          print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} doc_id is not unique! It has doc ids: {doc_id} and {_doc_id}")
          return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows

      if asl_consultant_id is None:
        asl_consultant_id = _asl_consultant_id
      else:
        if _asl_consultant_id != asl_consultant_id:
          not_unique = True
          print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} asl_consultant_id is not unique! It has doc ids: {asl_consultant_id} and {_asl_consultant_id}")
          return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows
  else:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has no document_participant_with_ids_mapping!")
    return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows
  
  if len(doc_participant_to_utterance_video_cameraperspective_mapping) > 0:
    not_unique = False
    for doc_participant_to_utterance_video_cameraperspective_mapping_instance in doc_participant_to_utterance_video_cameraperspective_mapping:
      # (<utterance seq id>, <video fname>, <camera perspective>)
      _utterance_seq_id = doc_participant_to_utterance_video_cameraperspective_mapping_instance[0]
      if _utterance_seq_id is None or not isinstance(_utterance_seq_id, int) or _utterance_seq_id<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has an invalid utterance seq id: {_utterance_seq_id}")
        return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows

      _media_fname = doc_participant_to_utterance_video_cameraperspective_mapping_instance[1]
      if _media_fname is None or len(_media_fname)==0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has an empty (or None) media filename")
        return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows

      _camera_perspective = doc_participant_to_utterance_video_cameraperspective_mapping_instance[2]
      if _camera_perspective is None or not isinstance(_camera_perspective, int) or _camera_perspective<0:
        print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has an invalid camera perspective: {_camera_perspective}")
        return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows

      # ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <utterance seq id>, <media filename>, <camera perspective>))
      validated_results.append(
        (
          (doc_id, asl_consultant_id), 
          (doc_fname, participant_name, _utterance_seq_id, _media_fname, _camera_perspective)
        )
      )
  else:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} document {doc_fname} has no doc_participant_to_utterance_video_cameraperspective_mapping entries")
    return vdpucpwiprt # note that this will cause an exception in beam since the shape will not match other validated rows

  return validated_results


def validate_preprocess_target_video_to_segment_mapping(target_video_to_segment_mapping_tpl):
  """
  target_video_to_segment_mapping_tpl:
    # (
    #   <target video fname>,
    #   {
    #     'doc_consultant_camera_perspective_mapping': [ 
    #       (
    #         <doc id>,               # <target video fname> can occur in more than one document
    #         <asl consultant id>,    # must map 1-to-1 to <target video fname>
    #         <camera perspective>    # must map 1-to-1 to <target video fname>
    #       ) # there can be more than one
    #     ],
    #     'video_to_segment_url_list_as_str': [<target video segment url list (as string)>] should be only one string
    #   }
    # )

  return:
    listof(
      (
        (<corpus doc id>, <asl consultant id>, <camera perspective>, <seg seq id>), # key
        (<target video fname>, <seg filename>, <seg url>)) # data
      )
    )
  """
  validated_results = []

  target_video_fname = target_video_to_segment_mapping_tpl[0]

  doc_consultant_camera_perspective_mapping = list(set(target_video_to_segment_mapping_tpl[1]['doc_consultant_camera_perspective_mapping']))
  if len(doc_consultant_camera_perspective_mapping) == 0:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} target video {target_video_fname} doc_consultant_camera_perspective_mapping is empty")
    return target_video_to_segment_mapping_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  video_to_segment_url_list_as_str = list(set(target_video_to_segment_mapping_tpl[1]['video_to_segment_url_list_as_str']))
  if len(video_to_segment_url_list_as_str) != 1:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} target video {target_video_fname} segment list is either empty or not unique: {video_to_segment_url_list_as_str}")
    return target_video_to_segment_mapping_tpl # note that this will cause an exception in beam since the shape will not match other validated rows
  video_to_segment_url_list_as_str = video_to_segment_url_list_as_str[0]

  segments = []
  asl_consultant_id = None
  multiple_asl_consultants = []
  camera_perspective = None
  multiple_camera_perspectives = []

  for seg_seg_id, seg_url in enumerate(video_to_segment_url_list_as_str.split(';')):
    seg_fname = str(seg_url).split('/')[-1]

    for doc_consultant_camera_perspective_mapping_instance in doc_consultant_camera_perspective_mapping:
      _doc_id = doc_consultant_camera_perspective_mapping_instance[0]
      _asl_consultant_id = doc_consultant_camera_perspective_mapping_instance[1]
      if _asl_consultant_id not in multiple_asl_consultants:
        multiple_asl_consultants.append(_asl_consultant_id)
      _camera_perspective = doc_consultant_camera_perspective_mapping_instance[2]
      if _camera_perspective not in multiple_camera_perspectives:
        multiple_camera_perspectives.append(_camera_perspective)

      validated_results.append(
        (
          (
            _doc_id,
            _asl_consultant_id,
            _camera_perspective,
            seg_seg_id
          ),
          (
            target_video_fname,
            seg_fname,
            seg_url
          )
        )
      )

  if len(multiple_asl_consultants) != 1:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} target video {target_video_fname} asl consultant is either either empty or not unique: {multiple_asl_consultants}")
    return target_video_to_segment_mapping_tpl # note that this will cause an exception in beam since the shape will not match other validated rows
  if len(multiple_camera_perspectives) != 1:
    print(f"{fidscs_globals.VALIDATION_FATAL_ERROR_TEXT} target video {target_video_fname} camera perspective is either either empty or not unique: {multiple_camera_perspectives}")
    return target_video_to_segment_mapping_tpl # note that this will cause an exception in beam since the shape will not match other validated rows

  return validated_results


def pl__6__create_document_asl_consultant_target_video_index_pcolls(
  ss_parsed_xmldb_pcoll, 
  document_asl_consultant_index_schemad_pcoll, 
  full_target_vid_index_schemad_pcoll, 
  d_pl_options
):
  # get list of target video infos
  target_video_doc_participant_utterance_mapping = (
    ss_parsed_xmldb_pcoll
    | "Beam PL: get doc filename, participant name, utterance seq id associated with this target video, keyed by target video filename" >> beam.Map(
        lambda ss_parsed_xmldb_pcoll_row_dict: [
          ( # key
            doc_participant_utterance_video_tpl[3], # <target video filename>
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
                str(urllib.parse.quote(d_target_video['TARGET_VIDEO_FNAME'])) # there may be spaces!
              ) for d_participant in ss_parsed_xmldb_pcoll_row_dict['PARTICIPANT_SEQUENCE']
                  for utterance_seq_id, d_utterance in enumerate(d_participant['UTTERANCE_SEQUENCE']) 
                    for d_target_video in d_utterance['TARGET_VIDEO_SEQUENCE']
            ] 
        ]
      ) # outputs pcoll with each row list of (<video filename>, (<corpus doc filename>, <participant name>, <utterance seq id>))
    | "Beam PL: 'explode' list of target_video_doc_participant_utterance_mapping tuples" >> beam.FlatMap(lambda list_target_video_doc_participant_utterance_mapping_mapping_tpl: list_target_video_doc_participant_utterance_mapping_mapping_tpl)
    | "Beam PL: select distinct list_target_video_doc_participant_utterance_mapping_tpl tuples" >> beam.Distinct()
    # debug
    # | "Beam PL: print pl__6__create_document_asl_consultant_target_video_index_pcolls result" >> beam.ParDo(beam__common.PipelinePcollPrinter("pl__6__create_document_asl_consultant_target_video_index_pcolls result"))
  )
  # now extract distinct target video fnames from media_doc_participant_mapping
  target_video_list_pcoll = (
    target_video_doc_participant_utterance_mapping
    | "Beam PL: extract media fname" >> beam.Map(lambda target_video_doc_participant_utterance_mapping_row_tpl: target_video_doc_participant_utterance_mapping_row_tpl[0])
    | "Beam PL: select distinct target video filenames" >> beam.Distinct()
    # debug
    # | "Beam PL: print media associated with this ss_parsed_xmldb" >> beam.ParDo(beam__common.PipelinePcollPrinter("\tmedia"))
  )
  target_full_target_vid_index_mapping = (
    full_target_vid_index_schemad_pcoll
    | "Beam PL: filter matching rows from vid index" >> beam.Filter(
        lambda vid_index_entry, matching_target_video_fnames: vid_index_entry.target_video_filename in matching_target_video_fnames,
        matching_target_video_fnames=beam.pvalue.AsIter(target_video_list_pcoll),
      )
    | "Beam PL: key matching vid index entries by target vid fname" >> beam.Map(
        lambda matching_target_vid_index_entry: (matching_target_vid_index_entry.target_video_filename, matching_target_vid_index_entry)
      )
    # debug
    # | "Beam PL: print vid index entries matching media associated with this ss_parsed_xmldb" >> beam.ParDo(beam__common.PipelinePcollPrinter("\tmatching vid index media"))
  )
  # merge doc, participant, and camera perspective keyed by media filename
  #   this pcoll will be used to produce final pcolls:
  #     document_asl_consultant_target_video_index_schemad_pcoll
  target_video_fname_to_merged_doc_participant_utterance_target_full_target_vid_index_mapping = (
    ({
      'target_video_doc_participant_utterance_mapping': target_video_doc_participant_utterance_mapping,
      'target_full_target_vid_index_mapping': target_full_target_vid_index_mapping
    })
    | "Beam PL: merge target_video_doc_participant_utterance_mapping and target_full_target_vid_index_mapping" >> beam.CoGroupByKey()
    # the above produces tuples in the form:
      # (
      #   <target video fname>, # key
      #   {
      #     'target_video_doc_participant_utterance_mapping': [
      #       (
      #         <corpus doc filename>, 
      #         <participant_name>, 
      #         <utterance seq id>
      #       )
      #     ], 
      #     'target_video_camera_perspective_mapping': [
      #       beam.Row(
      #         target_video_filename=<target_video_filename>, 
      #         video_seq_id=<video_seq_id>, 
      #         perspective_cam_id=<perspective_cam_id>, 
      #         compressed_mov_url=<compressed_mov_url>, 
      #         compressed_mov_url=<compressed_mov_url>, 
      #         uncompressed_avi_url=<uncompressed_avi_url>, 
      #         uncompressed_avi_mirror_1_url=<uncompressed_avi_mirror_1_url>, 
      #         <uncompressed_avi_mirror_2_url>
      #       )
      #     ]
      #   }
      # )
    # debug
    # | "Beam PL: print target_video_fname_to_merged_doc_participant_utterance_target_full_target_vid_index_mapping" >> beam.ParDo(beam__common.PipelinePcollPrinter("target_video_fname_to_merged_doc_participant_utterance_target_full_target_vid_index_mapping entry"))
  )

  doc_participant_to_utterance_video_cameraperspective_mapping = (
    target_video_fname_to_merged_doc_participant_utterance_target_full_target_vid_index_mapping
    | "Beam PL: validate/preprocess doc_participant_to_utterance_video_cameraperspective_mapping" >> beam.FlatMap(validate_preprocess_doc_participant_to_utterance_video_cameraperspective_mapping)
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
  document_asl_consultant_target_video_utterance_index_schemad_pcoll = (
    ({
      'document_participant_with_ids_mapping': document_participant_with_ids_mapping,
      'doc_participant_to_utterance_video_cameraperspective_mapping': doc_participant_to_utterance_video_cameraperspective_mapping
    })
    | "Beam PL: merge document_participant_with_ids_mapping and doc_participant_to_utterance_video_cameraperspective_mapping" >> beam.CoGroupByKey()
    # the above produces tuples in the form:
      # (
      #   (<doc filename>, <participant name>),     # key
      #   {
      #     'document_participant_with_ids_mapping': [(<doc id>, <asl consultant id>)], 
      #     'doc_participant_to_utterance_video_cameraperspective_mapping': [(<utterance seq id>, <video fname>, <camera perspective>)]
      #   }
      # )
    # | "Beam PL: print merged document_participant_with_ids_mapping and doc_participant_to_utterance_video_cameraperspective_mapping" >> beam.ParDo(beam__common.PipelinePcollPrinter("\tmerged document_participant_with_ids_mapping and doc_participant_to_utterance_video_cameraperspective_mapping entry"))
    | "Beam PL: validate/preprocess video_doc_participant_utterance_camera_perspective_with_ids_pcoll" >> beam.FlatMap(validate_preprocess_video_doc_participant_utterance_camera_perspective_with_ids_pcoll_row_tpl)
    # the above produces tuples in the form:
    #   ((<doc id>, <asl consultant id>), (<doc filename>, <participant name>, <utterance seq id>, <media filename>, <camera perspective>))
    | "Beam PL: apply schema to create final document_asl_consultant_target_video_utterance_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_video_index_pcoll_row_tpl: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'TargetVideoFilename'
          # ]
          DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
          DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
          ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
          ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
          UtteranceSequence=int(document_asl_consultant_video_index_pcoll_row_tpl[1][2]),
          CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][4]),                  
          TargetVideoFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][3])
        )
      )
    # debug
    # | "Beam PL: print document_asl_consultant_target_video_utterance_index_schemad_pcoll" >> beam.ParDo(beam__common.PipelinePcollPrinter("document_asl_consultant_target_video_utterance_index_schemad_pcoll entry"))
  )
  
  target_video_fname_to_doc_consultant_camera_perspective_mapping = (
    document_asl_consultant_target_video_utterance_index_schemad_pcoll
    | "Beam PL: extract (<target video filename>, (<corpus doc id>, <asl consultant id>, <camera perspective>)) from document_asl_consultant_target_video_utterance_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_target_video_index_schemad_pcoll_row: (
          document_asl_consultant_target_video_index_schemad_pcoll_row.TargetVideoFilename, # key
          (
            document_asl_consultant_target_video_index_schemad_pcoll_row.DocumentID,
            document_asl_consultant_target_video_index_schemad_pcoll_row.ASLConsultantID,
            document_asl_consultant_target_video_index_schemad_pcoll_row.CameraPerspective
          )
        )
      )
  )

  target_video_to_segment_url_list = (
    target_full_target_vid_index_mapping # (matching_target_vid_index_entry.target_video_filename, matching_target_vid_index_entry)
    | "Beam PL: extract (<target video filename>, <segment video url list as string>) from full_target_vid_index_mapping" >> beam.Map(
        lambda target_full_target_vid_index_mapping_row_tpl: (
          target_full_target_vid_index_mapping_row_tpl[0],
          target_full_target_vid_index_mapping_row_tpl[1].compressed_mov_url
        )
      )
  )

  document_target_video_segment_index_schemad_pcoll = (
    ({
      'doc_consultant_camera_perspective_mapping': target_video_fname_to_doc_consultant_camera_perspective_mapping,
      'video_to_segment_url_list_as_str': target_video_to_segment_url_list
    })
    | "Beam PL: merge doc_consultant_camera_perspective_mapping and video_to_segment_url_list_as_str" >> beam.CoGroupByKey()
    # the above produces tuples of the form:
      # (
      #   <target video fname>,
      #   {
      #     'doc_consultant_camera_perspective_mapping': listof( (<doc id>, <asl consultant id>, <camera perspective>) ),
      #     'video_to_segment_url_list_as_str': listof( (<target video segment url list (as string)>) )
      #   }
      # )
    | "Beam PL: validate/preprocess target_video_to_segment_mapping_pcoll" >> beam.FlatMap(validate_preprocess_target_video_to_segment_mapping)
    # the above produces tuples of the form:
      # (
      #   (<corpus doc id>, <asl consultant id>, <camera perspective>, <seg seq id>), # key
      #   (<target video fname>, <seg filename>, <seg url>)) # data
      # )
    | "Beam PL: apply schema to create final target_video_segment_index_pcoll" >> beam.Map(
        lambda target_video_to_segment_mapping_pcoll_row_tpl: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'SegmentSequence',
          #   'SegmentVideoFilename',
          #   'URL'
          # ]
          DocumentID=int(target_video_to_segment_mapping_pcoll_row_tpl[0][0]),
          ASLConsultantID=int(target_video_to_segment_mapping_pcoll_row_tpl[0][1]),
          CameraPerspective=int(target_video_to_segment_mapping_pcoll_row_tpl[0][2]),                  
          TargetVideoFilename=str(target_video_to_segment_mapping_pcoll_row_tpl[1][0]),
          SegmentSequence=int(target_video_to_segment_mapping_pcoll_row_tpl[0][3]),
          SegmentVideoFilename=str(target_video_to_segment_mapping_pcoll_row_tpl[1][1]),
          URL=str(target_video_to_segment_mapping_pcoll_row_tpl[1][2])
        )
      )
    # debug
    # | "Beam PL: print target_video_to_segment_mapping" >> beam.ParDo(beam__common.PipelinePcollPrinter("target_video_to_segment_mapping entry"))
  )

  return document_asl_consultant_target_video_utterance_index_schemad_pcoll, document_target_video_segment_index_schemad_pcoll


def pl__7__write_document_asl_consultant_target_video_index_csv(document_asl_consultant_target_video_utterance_index_schemad_pcoll, d_pl_options):
  """
  document_asl_consultant_target_video_utterance_index_schemad_pcoll:
    beam.Row(
      # SCHEMA_COL_NAMES__VIDEO_DS = [
      #   'DocumentID',
      #   'ASLConsultantID',
      #   'CameraPerspective',
      #   'TargetVideoFilename'
      # ]
      DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
      DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
      ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
      ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
      UtteranceSequence=int(document_asl_consultant_video_index_pcoll_row_tpl[1][2]),
      CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][4]),                  
      TargetVideoFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][3])
    )
  """
  distinct_document_asl_consultant_video_index_pcoll = (
    document_asl_consultant_target_video_utterance_index_schemad_pcoll
    | "Beam PL: extract SCHEMA_COL_NAMES__VIDEO_DS columns from document_asl_consultant_target_video_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_target_video_index_schemad_pcoll_row: (
          document_asl_consultant_target_video_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_target_video_index_schemad_pcoll_row.ASLConsultantID,
          document_asl_consultant_target_video_index_schemad_pcoll_row.CameraPerspective,
          document_asl_consultant_target_video_index_schemad_pcoll_row.TargetVideoFilename,
          # document_asl_consultant_target_video_index_schemad_pcoll_row.FrameSequence,
          # document_asl_consultant_target_video_index_schemad_pcoll_row.FramePath,
          document_asl_consultant_target_video_index_schemad_pcoll_row.UtteranceSequence
        )
      )
    | "Beam PL: select distinct document_asl_consultant_video_index rows" >> beam.Distinct()
  )
  sorted_distinct_document_asl_consultant_video_index_pcoll = beam__common.pl__X__sort_pcoll(
    distinct_document_asl_consultant_video_index_pcoll, 
    pcoll_label="distinct_document_asl_consultant_video_index"
  )
  sorted_distinct_document_asl_consultant_video_index_csv_rows_pcoll = (
    sorted_distinct_document_asl_consultant_video_index_pcoll
    | "Beam PL: apply minimal schema to create final document_asl_consultant_target_video_index_schemad_pcoll of distinct rows" >> beam.Map(
        lambda distinct_document_asl_consultant_video_index_row: beam.Row(
          DocumentID=int(distinct_document_asl_consultant_video_index_row[0]),
          ASLConsultantID=int(distinct_document_asl_consultant_video_index_row[1]),
          CameraPerspective=int(distinct_document_asl_consultant_video_index_row[2]),
          Filename=str(distinct_document_asl_consultant_video_index_row[3])
        )
      )
    | beam.Map(lambda distinct_document_asl_consultant_target_video_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(distinct_document_asl_consultant_target_video_index_schemad_pcoll_row))
  )

  distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll = (
    document_asl_consultant_target_video_utterance_index_schemad_pcoll
    | "Beam PL: extract columns for distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_target_video_utterance_index_schemad_pcoll_row: (
          document_asl_consultant_target_video_utterance_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_target_video_utterance_index_schemad_pcoll_row.ASLConsultantID,
          document_asl_consultant_target_video_utterance_index_schemad_pcoll_row.CameraPerspective,
          document_asl_consultant_target_video_utterance_index_schemad_pcoll_row.TargetVideoFilename,
          # document_asl_consultant_target_video_utterance_index_schemad_pcoll_row.FrameSequence,
          # document_asl_consultant_target_video_utterance_index_schemad_pcoll_row.FramePath,
          document_asl_consultant_target_video_utterance_index_schemad_pcoll_row.UtteranceSequence
        )
      )
    | "Beam PL: select distinct document_asl_consultant_target_video_utterance_index rows" >> beam.Distinct()
  )
  sorted_distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll = beam__common.pl__X__sort_pcoll(
    distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll, 
    pcoll_label="distinct_document_asl_consultant_target_video_utterance_index"
  )
  sorted_distinct_document_asl_consultant_target_video_utterance_index_csv_rows_pcoll = (
    sorted_distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll
    | "Beam PL: apply minimal schema to create final distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll of distinct rows" >> beam.Map(
        lambda distinct_document_asl_consultant_target_video_utterance_index_row: beam.Row(
          DocumentID=int(distinct_document_asl_consultant_target_video_utterance_index_row[0]),
          ASLConsultantID=int(distinct_document_asl_consultant_target_video_utterance_index_row[1]),
          CameraPerspective=int(distinct_document_asl_consultant_target_video_utterance_index_row[2]),
          TargetVideoFilename=str(distinct_document_asl_consultant_target_video_utterance_index_row[3]),
          # FrameSequence=int(distinct_document_asl_consultant_target_video_utterance_index_row[4]),
          # FramePath=str(distinct_document_asl_consultant_target_video_utterance_index_row[4]),
          UtteranceSequence=int(distinct_document_asl_consultant_target_video_utterance_index_row[4])
        )
      )
    | beam.Map(lambda distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(distinct_document_asl_consultant_target_video_utterance_index_schemad_pcoll_row))
  )

  return beam__common.pl__X__write_pcoll_to_csv(  # document_asl_consultant_video_index_csv_path
    sorted_distinct_document_asl_consultant_video_index_csv_rows_pcoll, 
    "DOCUMENT-ASLCONSULTANT-VIDEO-INDEX", 
    fidscs_globals.VIDEO_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__VIDEO_DS, 
    d_pl_options
  ), beam__common.pl__X__write_pcoll_to_csv(
    sorted_distinct_document_asl_consultant_target_video_utterance_index_csv_rows_pcoll, 
    "DOCUMENT-ASLCONSULTANT-VIDEO-UTTERANCE-INDEX", 
    fidscs_globals.VIDEO_UTTERANCE_DS_FNAME, 
    ['DocumentID', 'ASLConsultantID', 'CameraPerspective', 'TargetVideoFilename', 'UtteranceSequence'], 
    d_pl_options
  )


def pl__7__write_document_asl_consultant_utterance_video_index_csv(document_asl_consultant_target_video_index_schemad_pcoll, d_pl_options):
  distinct_document_asl_consultant_utterance_video_index_pcoll = (
    document_asl_consultant_target_video_index_schemad_pcoll
    # beam.Row(
    #   # SCHEMA_COL_NAMES__VIDEO_DS = [
    #   #   'DocumentID',
    #   #   'ASLConsultantID',
    #   #   'CameraPerspective',
    #   #   'TargetVideoFilename'
    #   # ]
    #   DocumentID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][0]),
    #   DocumentFileName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][0]),
    #   ASLConsultantID=int(document_asl_consultant_video_index_pcoll_row_tpl[0][1]),
    #   ParticipantName=str(document_asl_consultant_video_index_pcoll_row_tpl[1][1]),
    #   UtteranceSequence=int(document_asl_consultant_video_index_pcoll_row_tpl[1][2]),
    #   CameraPerspective=int(document_asl_consultant_video_index_pcoll_row_tpl[1][4]),                  
    #   TargetVideoFilename=str(document_asl_consultant_video_index_pcoll_row_tpl[1][3])
    # )
    | "Beam PL: extract (DocumentID, ASLConsultantID, UtteranceSequence, CameraPerspective) from document_asl_consultant_target_video_index_schemad_pcoll" >> beam.Map(
        lambda document_asl_consultant_target_video_index_schemad_pcoll_row: (
          document_asl_consultant_target_video_index_schemad_pcoll_row.DocumentID,
          document_asl_consultant_target_video_index_schemad_pcoll_row.ASLConsultantID,
          document_asl_consultant_target_video_index_schemad_pcoll_row.TargetVideoFilename, # added
          document_asl_consultant_target_video_index_schemad_pcoll_row.UtteranceSequence,
          document_asl_consultant_target_video_index_schemad_pcoll_row.CameraPerspective
        )
      )
    | "Beam PL: select distinct (DocumentID, ASLConsultantID, UtteranceSequence, CameraPerspective) extracted from document_asl_consultant_target_video_index_schemad_pcoll" >> beam.Distinct()
  )
  sorted_distinct_document_asl_consultant_utterance_video_index_pcoll = beam__common.pl__X__sort_pcoll(
    distinct_document_asl_consultant_utterance_video_index_pcoll, 
    pcoll_label="distinct_document_asl_consultant_utterance_video_index"
  )
  sorted_distinct_document_asl_consultant_utterance_video_index_csv_rows_pcoll = (
    sorted_distinct_document_asl_consultant_utterance_video_index_pcoll
    | "Beam PL: apply schema to create final document_asl_consultant_utterance_video_index_schemad_pcoll" >> beam.Map(
      lambda distinct_document_asl_consultant_video_index_row: beam.Row(
        # SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS = [
        #   'DocumentID',
        #   'ASLConsultantID',
        #   'TargetVideoFilename',
        #   'UtteranceSequence',
        #   'CameraPerspective'
        # ]
        DocumentID=int(distinct_document_asl_consultant_video_index_row[0]),
        ASLConsultantID=int(distinct_document_asl_consultant_video_index_row[1]),
        TargetVideoFilename=str(distinct_document_asl_consultant_video_index_row[2]),
        UtteranceSequence=int(distinct_document_asl_consultant_video_index_row[3]),
        CameraPerspective=int(distinct_document_asl_consultant_video_index_row[4])
      )
    )
    | beam.Map(lambda distinct_document_asl_consultant_utterance_video_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(distinct_document_asl_consultant_utterance_video_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_distinct_document_asl_consultant_utterance_video_index_csv_rows_pcoll, 
    "DOCUMENT-ASLCONSULTANT-UTTERANCE-TARGETVIDEO-INDEX", 
    fidscs_globals.UTTERANCE_VIDEO_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__UTTERANCE_VIDEO_DS, 
    d_pl_options
  ) # document_asl_consultant_utterance_video_index_csv_path


def pl__7__write_document_target_video_segment_index_csv(document_target_video_segment_index_schemad_pcoll, d_pl_options):
  distinct_target_video_segment_index_pcoll = (
    document_target_video_segment_index_schemad_pcoll
    | "Beam PL: extract (DocumentID, ASLConsultantID, CameraPerspective, TargetVideoFilename, SegmentSequence, SegmentVideoFilename, URL) from document_target_video_segment_index_schemad_pcoll" >> beam.Map(
        lambda document_target_video_segment_index_schemad_pcoll_row: (
          document_target_video_segment_index_schemad_pcoll_row.DocumentID,
          document_target_video_segment_index_schemad_pcoll_row.ASLConsultantID,
          document_target_video_segment_index_schemad_pcoll_row.CameraPerspective,
          document_target_video_segment_index_schemad_pcoll_row.TargetVideoFilename,
          document_target_video_segment_index_schemad_pcoll_row.SegmentSequence,
          document_target_video_segment_index_schemad_pcoll_row.SegmentVideoFilename,
          document_target_video_segment_index_schemad_pcoll_row.URL,
        )
      )
    | "Beam PL: select distinct (DocumentID, ASLConsultantID, CameraPerspective, TargetVideoFilename, SegmentSequence, SegmentVideoFilename, URL) extracted from target_video_segment_index_pcoll" >> beam.Distinct()
  )
  sorted_distinct_target_video_segment_index_pcoll = beam__common.pl__X__sort_pcoll(
    distinct_target_video_segment_index_pcoll, 
    pcoll_label="target_video_segment_index"
  )
  sorted_target_video_segment_index_schemad_pcoll = (
    sorted_distinct_target_video_segment_index_pcoll
    | "Beam PL: re-apply schema to create final sorted_target_video_segment_index_schemad_pcoll" >> beam.Map(
        lambda sorted_distinct_target_video_segment_index_tpl: beam.Row(
          # SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS = [
          #   'DocumentID',
          #   'ASLConsultantID',
          #   'CameraPerspective',
          #   'TargetVideoFilename',
          #   'SegmentSequence',
          #   'SegmentVideoFilename',
          #   'URL'
          # ]
          DocumentID=sorted_distinct_target_video_segment_index_tpl[0],
          ASLConsultantID=sorted_distinct_target_video_segment_index_tpl[1],
          CameraPerspective=sorted_distinct_target_video_segment_index_tpl[2],                  
          TargetVideoFilename=sorted_distinct_target_video_segment_index_tpl[3],
          SegmentSequence=sorted_distinct_target_video_segment_index_tpl[4],
          SegmentVideoFilename=sorted_distinct_target_video_segment_index_tpl[5],
          URL=sorted_distinct_target_video_segment_index_tpl[6]
        )
      )
    | beam.Map(lambda sorted_target_video_segment_index_schemad_pcoll_row: beam__common.beam_row_to_csv_string(sorted_target_video_segment_index_schemad_pcoll_row))
  )
  return beam__common.pl__X__write_pcoll_to_csv(
    sorted_target_video_segment_index_schemad_pcoll, 
    "DOCUMENT-ASLCONSULTANT-TARGETVIDEO-SEGMENT-INDEX", 
    fidscs_globals.VIDEO_SEGMENT_DS_FNAME, 
    fidscs_globals.SCHEMA_COL_NAMES__VIDEO_SEGMENT_DS, 
    d_pl_options
  ) # target_video_segment_index_csv_path


def pl__4__debug_print_signstream_db(ss_parsed_xmldb_pcoll):
  return (
    ss_parsed_xmldb_pcoll
    | "Beam PL: debug print parsed signstream xmldb" >> beam.Map(debug_print_signstream_db)
  )


def pl__3__parallel_download_videos(vid_index_schemad_pcoll, d_pl_options, n_partitions=8):
  # ******************** DOWNLOAD VIDEOS IN PARALLEL: BEGIN ********************
  # this is just for debugging - comment out for production
  # (
  #   vid_index_schemad_pcoll
  #   | 'Count videos queued for download' >> beam.combiners.Count.Globally()
  #   | 'Print result' >> beam.Map(lambda count_pcol_element: print(f"Videos queued for download: {count_pcol_element}"))
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
      | f"Beam PL: {p_label} gather download info for video segments" >> beam.ParDo(VideoSegmentInfoGatherer(d_pl_options)) # get_video_segment_download_info(schemad_pcoll_element)
      | f"Beam PL: {p_label} download video segments" >> beam.ParDo(VideoSegmentDownloader(d_pl_options, f"{p_label_indented}")) # outputs a pcoll with each row as [{'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}]
    )
    partition_download_results[i] = p_dl_results

    # # note that this depends on the DAG - i.e. will not occur until p_dl_results are ready which, of course, does not occur until all videos have been downloaded
    # (
    #   p_dl_results
    #   | f"Beam PL: {p_label} count videos downloaded" >> beam.combiners.Count.Globally() 
    #   | f"Beam PL: {p_label} print videos downloaded count" >> beam.ParDo(beam__common.PipelinePcollPrinter(label=p_label_indented, msg="videos downloaded/found"))
    # )

  # now merge all download results
  merged_download_results = (
    (p_dl_r for p_dl_r in partition_download_results) 
    | f"Beam PL: merge download results" >> beam.Flatten() 
  )

  return merged_download_results
  # ******************** DOWNLOAD VIDEOS IN PARALLEL: END ********************


def pl__4__parallel_extract_target_video_frames(merged_download_results, d_pl_options, n_partitions=8):
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
  target_vid_seg_frame_extraction_partitions = (
    merged_download_results # pcoll with each row as {'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}
    | f"Beam PL: group extraction info for video segments by target video" >> beam.GroupBy(lambda d: d['target_video_fname']) # yields pcoll of rows as (target_video_fname, list({'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1]}))
    | f"Beam PL: partition target video segment info for extraction parallelization" >> beam.Partition(
        lambda vid_index_row, num_partitions: random.randint(0,num_partitions-1), 
        # lambda vid_index_row, num_partitions: np.random.uniform(0,num_partitions), # not working yet
        n_partitions
      )
  )

  partition_extraction_results = [None for i in range(n_partitions)]
  for i, p in enumerate(target_vid_seg_frame_extraction_partitions):
    p_label = f"p{i+1}"
    p_label_indented = f"\t{p_label}"

    p_extraction_results = (
      p
      | f"Beam PL: {p_label} extract frames of each segment per target video" >> beam.ParDo(SegmentFrameExtractor(d_pl_options, f"{p_label_indented}", debug=False)) # passthrough: pcoll of rows as (target_video_fname, n_stitched_frames, list({'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1], 'n_frames_extracted': n_frames_extracted}))
    )
    partition_extraction_results[i] = p_extraction_results

    (
      p_extraction_results
      | f"Beam PL: {p_label} count target videos processed" >> beam.combiners.Count.Globally() 
      | f"Beam PL: {p_label} print target videos processed count" >> beam.ParDo(beam__common.PipelinePcollPrinter(label=p_label_indented, msg="target videos processed"))
    )
  
  merged_extraction_results = (
    (p_extraction_results for p_extraction_results in partition_extraction_results) 
    | f"Beam PL: merge extraction results" >> beam.Flatten() # outputs pcoll of rows as tpl_target_video_extraction_info: (target_video_fname, n_stitched_frames, list({'target_video_fname': target_video_fname, 'target_video_frames_dir': target_video_frames_dir, 'segment_url': str(url), 'segment_fname': str(url).split('/')[-1], 'n_frames_extracted': n_frames_extracted}))
    # | f"Beam PL: print merged extraction results" >> beam.ParDo(beam__common.PipelinePcollPrinter(label="\t"))
  )
  _ = (
    merged_extraction_results
    | "Beam PL: apply schema to merged extraction results pcoll" >> beam.Map(lambda x: beam.Row(
          video_fname=str(x[0]),
          n_stitched_frames=int(x[1])
        ))
    # | "Beam PL: count total frames extracted" >> beam.transforms.sql.SqlTransform(f"SELECT SUM(n_stitched_frames) AS total_frames_extracted FROM PCOLLECTION") # this is VERY, VERY SLOW
    | "Beam PL: select n_stitched_frames" >> beam.Map(lambda extraction_results_row: extraction_results_row.n_stitched_frames) # on DirectRunner, this is literally about 100 times faster!
    | "Beam PL: count total frames extracted" >> beam.CombineGlobally(sum)
    | f"Beam PL: print total frames extracted" >> beam.ParDo(beam__common.PipelinePcollPrinter(msg="TOTAL FRAMES EXTRACTED"))
  )

  return merged_extraction_results
  # ******************** EXTRACT SEGMENT-FRAMES IN PARALLEL: END ********************


class FIDSCapstonePipelineOptions(PipelineOptions):
  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument(
      '--fidscs-capstone-max-target-videos',
      default=None
    )
    parser.add_argument(
      '--fidscs-capstone-work-dir',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-data-dir',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-tmp-dir',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-videos-dir',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-stitched-video-frames-dir',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-corpus-dir',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-corpus-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-document-asl-cconsultant-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-asl-consultant-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-video-indexes-dir',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-selected-video-index-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-video-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-video-segment-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-video-frame-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-utterance-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-utterance-video-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-utterance-token-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-utterance-token-frame-ds-path',
      default=None,
    )
    parser.add_argument(
      '--fidscs-capstone-vocabulary-ds-path',
      default=None,
    )


def run(
  max_target_videos,
  work_dir,
  beam_runner='DirectRunner', 
  beam_gcp_project=None,
  beam_gcp_region=None,
  beam_gcp_dataflow_job_name=None,
  beam_gcs_staging_bucket=None,
  beam_gcs_temp_location=None,
  beam_gcp_dataflow_setup_file=None
):
  options = None

  if beam_runner != 'DirectRunner':
    if beam_gcp_dataflow_setup_file != './setup.py':
      print(f"*** FATAL ERROR!!! ***  beam_gcp_setup_file=={beam_gcp_dataflow_setup_file} but it should be ./setup.py")
      return

    logging.getLogger().setLevel(logging.INFO) # enable logging only for DataflowRunner

    options = {
      'runner': beam_runner,
      'streaming': False, # set to True if data source is unbounded (e.g. GCP PubSub),
      'max_num_workers': 8,
      'autoscaling_algorithm': 'THROUGHPUT_BASED',
      'num_workers': 4,
      'save_main_session': True,
      'enable_streaming_engine': False,

      # GCP options
      'project': beam_gcp_project,
      'region': beam_gcp_region,
      'worker_region': beam_gcp_region,
      'service_account_email': 'fids-capstone-beam-pl-gcs@sc-fids-capstone.iam.gserviceaccount.com',
      'staging_location': beam_gcs_staging_bucket,
      'temp_location': beam_gcs_temp_location,
      'setup_file': beam_gcp_dataflow_setup_file,
      'job_name': beam_gcp_dataflow_job_name,
    }
  else:
    options = {
      'runner': 'DirectRunner',
      'environment_type': 'DOCKER',
      'direct_num_workers': 0, # 0 is use all available cores
      'direct_running_mode': 'multi_processing', # ['in_memory', 'multi_threading', 'multi_processing'] # 'multi_processing' doesn't seem to work for DirectRunner?
      'streaming': False # set to True if data source is unbounded (e.g. GCP PubSub),
    }

  options.update(beam__common.make_fids_options_dict(work_dir, max_target_videos=max_target_videos, beam_gcp_project=beam_gcp_project))

  n_partitions = 8 # hardcoded for now but we need to retrieve this from beam to be the number of workers

  job_suffix = 'boostrap-vid-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    full_target_vid_index_schemad_pcoll = pl__1__bootstrap_target_video_index(pl)
    pl__2__write_target_vid_index_csv(full_target_vid_index_schemad_pcoll, pl._options._all_options)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'download-videos-extract-frames'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
    filtered_target_vid_index_schemad_pcoll = pl__2__filter_target_vid_index(full_target_vid_index_schemad_pcoll, pl._options._all_options)
    merged_download_results = pl__3__parallel_download_videos(filtered_target_vid_index_schemad_pcoll, pl._options._all_options, n_partitions)
    merged_extraction_results = pl__4__parallel_extract_target_video_frames(merged_download_results, pl._options._all_options, n_partitions)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'bootstrap-corpus-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    pl__1__bootstrap_corpus_index(pl)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")
  # writing the corpus index needs to be in a separate pipeline, which will execute sequentially after the download completes
  #   note that if we don't do it this way, it is HIGHLY probable that file structure will not be ready
  #   for reading yet
  job_suffix = 'transform-corpus-documents-to-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    corpus_index_schemad_pcoll = pl__1__corpus_document_file_structure_to_corpus_index(pl)
    pl__2__write_corpus_index_csv(
      corpus_index_schemad_pcoll, 
      beam__common.GlobalVarValueAssigner(fn_assign_to_global=assign_to_global__raw_xml_b64_max_len),
      pl._options._all_options
    )
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'transform-ss-xml-to-asl-consultant-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
    corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl)
    corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll, pl._options._all_options)
    ss_parsed_xmldb_pcoll = pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll, pl._options._all_options)
    asl_consultant_index_schemad_pcoll = pl__4__create_asl_consultant_index_schemad_pcoll(ss_parsed_xmldb_pcoll, pl._options._all_options)
    pl__5__write_asl_consultant_index_csv(asl_consultant_index_schemad_pcoll, pl._options._all_options)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'transform-ss-xml-to-document-asl-consultant-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
    corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl)
    corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll, pl._options._all_options)
    ss_parsed_xmldb_pcoll = pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll, pl._options._all_options)
    asl_consultant_index_schemad_pcoll = beam__common.pl__1__read_asl_consultant_index_csv(pl)
    document_asl_consultant_index_schemad_pcoll = pl__5__create_document_asl_consultant_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      corpus_index_schemad_pcoll, 
      asl_consultant_index_schemad_pcoll,
      pl._options._all_options
    )
    pl__6__write_document_asl_consultant_index_csv(document_asl_consultant_index_schemad_pcoll, pl._options._all_options)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'transform-ss-xml-to-document-asl-consultant-utterance-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
    corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl)
    corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll, pl._options._all_options)
    ss_parsed_xmldb_pcoll = pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll, pl._options._all_options)
    document_asl_consultant_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_index_csv(pl)
    document_asl_consultant_utterance_index_schemad_pcoll = pl__6__create_document_asl_consultant_utterance_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      document_asl_consultant_index_schemad_pcoll,
      pl._options._all_options
    )
    pl__7__write_document_asl_consultant_utterance_index_csv(document_asl_consultant_utterance_index_schemad_pcoll, pl._options._all_options)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'transform-ss-xml-to-document-asl-consultant-target-video-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
    corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl)
    corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll, pl._options._all_options)
    ss_parsed_xmldb_pcoll = pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll, pl._options._all_options)
    document_asl_consultant_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_index_csv(pl)
    document_asl_consultant_target_video_index_schemad_pcoll, document_target_video_segment_index_schemad_pcoll = pl__6__create_document_asl_consultant_target_video_index_pcolls(
      ss_parsed_xmldb_pcoll, 
      document_asl_consultant_index_schemad_pcoll, 
      full_target_vid_index_schemad_pcoll, 
      pl._options._all_options
    )
    pl__7__write_document_asl_consultant_target_video_index_csv(document_asl_consultant_target_video_index_schemad_pcoll, pl._options._all_options)
    pl__7__write_document_asl_consultant_utterance_video_index_csv(document_asl_consultant_target_video_index_schemad_pcoll, pl._options._all_options)
    pl__7__write_document_target_video_segment_index_csv(document_target_video_segment_index_schemad_pcoll, pl._options._all_options)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'transform-ss-xml-to-vocabulary-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    full_target_vid_index_schemad_pcoll = beam__common.pl__1__read_target_vid_index_csv(pl)
    corpus_index_schemad_pcoll = beam__common.pl__1__read_corpus_index_csv(pl)
    corpus_index_decoded_XML_pcoll = pl__2__decode_XML(corpus_index_schemad_pcoll, pl._options._all_options)
    ss_parsed_xmldb_pcoll = pl__3__parse_signstream_database(corpus_index_decoded_XML_pcoll, pl._options._all_options)
    document_asl_consultant_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_index_csv(pl)
    vocabulary_index_pcoll, document_asl_consultant_utterance_token_index_schemad_pcoll = pl__6__create_document_asl_consultant_utterance_token_index_schemad_pcoll(
      ss_parsed_xmldb_pcoll, 
      document_asl_consultant_index_schemad_pcoll, 
      pl._options._all_options
    )
    pl__7__write_vocabulary_index_csv(vocabulary_index_pcoll, pl._options._all_options)
    pl__7__write_document_asl_consultant_utterance_token_index_csv(document_asl_consultant_utterance_token_index_schemad_pcoll, pl._options._all_options)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'transform-ss-xml-to-document-asl-consultant-target-video-frame-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    document_asl_consultant_target_video_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_index_csv(pl)
    document_asl_consultant_target_video_frame_index_schemad_pcoll = pl__7__create_document_asl_consultant_target_video_frame_index_schemad_pcoll(
      document_asl_consultant_target_video_index_schemad_pcoll, 
      pl._options._all_options
    )
    pl__8__write_document_asl_consultant_target_video_frame_index_schemad_pcoll(document_asl_consultant_target_video_frame_index_schemad_pcoll, pl._options._all_options)
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'transform-ss-xml-to-document-asl-consultant-target-video-utterance-token-frame-index'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    document_asl_consultant_utterance_token_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_utterance_token_index_csv(pl)
    document_asl_consultant_target_video_utterance_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_utterance_index_csv(pl)
    document_asl_consultant_target_video_frame_index_schemad_pcoll = beam__common.pl__1__read_document_asl_consultant_target_video_frame_index_csv(pl)
    document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll = pl__8__create_document_asl_consultant_utterance_token_frame_index_schemad_pcoll(
      document_asl_consultant_utterance_token_index_schemad_pcoll,
      document_asl_consultant_target_video_utterance_index_schemad_pcoll,
      document_asl_consultant_target_video_frame_index_schemad_pcoll, 
      pl._options._all_options
    )
    pl__9__write_document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll(
      document_asl_consultant_target_video_utterance_token_frame_index_schemad_pcoll, 
      pl._options._all_options
    )
  print(f"****************************** Finished pipeline job: {job_name} ******************************")


  job_suffix = 'remove-intermediate-files'
  job_name = f"{beam_gcp_dataflow_job_name}--{job_suffix}"
  print(f"\n\n****************************** Starting pipeline job: {job_name} ******************************")
  options.update({
    'job_name': job_name
  })
  pipeline_options = PipelineOptions(flags=[], **options) # easier to pass in options from command-line this way
  print(f"PipelineOptions:\n{pipeline_options.get_all_options()}\n")
  with beam.Pipeline(options=pipeline_options) as pl:
    tmp_dir_path_pcoll = (
      pl
      | f"Beam PL: create {pl._options._all_options[fidscs_globals.OPT_NAME_TMP_DIR]} pcoll for cleanup" >> beam.Create([pl._options._all_options[fidscs_globals.OPT_NAME_TMP_DIR]])
    )
    beam__common.pl__X__rmdir(
      tmp_dir_path_pcoll, 
      pl._options._all_options[fidscs_globals.OPT_NAME_TMP_DIR],
      pl._options._all_options
    )
    videos_dir_path_pcoll = (
      pl
      | f"Beam PL: create {pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_DIR]} pcoll for cleanup" >> beam.Create([pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_DIR]])
    )
    beam__common.pl__X__rmdir(
      videos_dir_path_pcoll, 
      pl._options._all_options[fidscs_globals.OPT_NAME_VIDEO_DIR],
      pl._options._all_options
    )
  print(f"****************************** Finished pipeline job: {job_name} ******************************")
  

  print(f"Beam PL: ALL DONE!")
