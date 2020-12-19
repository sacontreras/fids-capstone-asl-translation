import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
import os
import sys
import csv
import io
from apache_beam.transforms.sql import SqlTransform
# import apache_beam.runners.interactive.interactive_beam as ib
import random
import typing
from preprocessr__common import *
import utils
import cv2
import urllib
import globals

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
  return vid_index_csv_rows(sel_vid_index_csv_path, rows_to_dicts=True, dict_field_names=globals.SCHEMA_COL_NAMES__VIDEO_INDEX)


class PipelinePcollPrinter(beam.DoFn):
  """
  prints each element of the pcoll
  should generally only be used for debugging
  """
  def __init__(self, label="", msg=""):
    self.label = label
    self.msg = msg

  def process(self, pcoll_element):
    print(f"{self.label+': ' if len(self.label)>0 else ''}{self.msg+': ' if len(self.msg)>0 else ''}{pcoll_element}")
    return [pcoll_element] # passthrough


class VideoIndexPandasDataframeFromSchemadPcoll(beam.DoFn):
  """
  creates an underlying pandas DataFrame
  appends pcoll dict element to this dataframe
  """
  def __init__(self):
    self.df_video_index = pd.DataFrame(columns=globals.SCHEMA_COL_NAMES__VIDEO_INDEX)
    # debug
    self.rows = 0

  def process(self, pcoll_dict_element):
    self.df_video_index = self.df_video_index.append([pcoll_dict_element])
    return [pcoll_dict_element] # passthrough


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




def run():
  # vid_index_df_converter = VideoIndexPandasDataframeFromSchemadPcoll()

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
              'vid_indexes_dir': globals.VIDEO_INDEXES_DIR, 
              'sel_vid_index_path': globals.SELECTED_VIDEO_INDEX_PATH, 
              'video_indexes_archive': globals.VIDEO_INDEXES_ARCHIVE, 
              'tmp_dir': globals.TMP_DIR
            }
          ]
        )
      | "Beam PL: bootstrap video index" >> beam.Map(boostrap_video_index) # boostrap_video_index outputs SELECTED_VIDEO_INDEX_PATH but beam.Map() wraps this in a pcoll and is fed to...

      # | "Beam PL: read csv rows" >> beam.FlatMap(vid_index_csv_rows) # but rows of this PColl are lists and the first one is the header row (column names), which we do not want...
      | "Beam PL: read video index into pcoll" >> beam.FlatMap(vid_index_csv_rows_to_dicts) # outputs another pcoll but with each row as dict
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

      # filter schemad pcoll as desired (if necessary) using SqlTransform(), for example limiting size of pcoll data items to max_data_files
      | SqlTransform(f"SELECT * FROM PCOLLECTION {'LIMIT '+str(globals.MAX_DATA_FILES) if globals.MAX_DATA_FILES is not None and globals.MAX_DATA_FILES>0 else ''}")
    )

    # (
    #   vid_index_schemad_pcoll
    #   | 'Count videos queued for download' >> beam.combiners.Count.Globally()
    #   | 'Print result' >> beam.Map(lambda count_pcol_element: print(f"Videos queued for download: {count_pcol_element}"))
    # )

    # ******************** DOWNLOAD VIDEOS IN PARALLEL: BEGIN ********************
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
    n_partitions = 8 # hardcoded for now 
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
    # ******************** DOWNLOAD VIDEOS IN PARALLEL: END ********************


    # ******************** EXTRACT SEGMENT-FRAMES IN PARALLEL: BEGIN ********************
    #   but first we need to merge all download results
    merged_download_results = (
      (p_dl_r for p_dl_r in partition_download_results) 
      | f"Beam PL: merge download results" >> beam.Flatten() 
    )

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
      | "Beam PL: select n_stitched_frames" >> beam.Map(lambda extraction_results_row: extraction_results_row.n_stitched_frames)
      | "Beam PL: count total frames extracted" >> beam.CombineGlobally(sum)

      | f"Beam PL: print total frames extracted" >> beam.ParDo(PipelinePcollPrinter(msg="TOTAL FRAMES EXTRACTED"))
    )
    # ******************** EXTRACT SEGMENT-FRAMES IN PARALLEL: END ********************

  print(f"Beam PL: ALL DONE!")
  # df_video_index = vid_index_df_converter.df_video_index # this doesn't work since it's not thread-safe!
  df_video_index = None