from __future__ import absolute_import

import logging
import urllib
import urllib.request
import zipfile
from io import BytesIO

from tqdm.auto import tqdm, trange

from api import beam__common, fileio


def download(url, local_fname, block_sz=8192, display=True, nested_tqdm_pb=None):
  http_file = urllib.request.urlopen(url)
  if http_file.getcode() != 200:
      raise ValueError(f"ERROR {http_file.getcode()} while opening {url}.")

  meta = http_file.info()
  file_size = int(meta['Content-Length'])
  if nested_tqdm_pb is None and display:
    print(f"Downloading {url} (filesize: {file_size} bytes) to {local_fname}...")

  f = None
  memfile = BytesIO()
  try:
    file_size_dl = 0
    fblocks = range(0, file_size, block_sz)
    if nested_tqdm_pb is None:
      tqdm_pb = tqdm(fblocks, disable=not display)
    else:
      tqdm_pb = nested_tqdm_pb
    tqdm_pb.leave = True
    tqdm_pb.reset(total=file_size)
    tqdm_pb.refresh(nolock=False)
    for fblock in fblocks: 
        buffer = http_file.read(block_sz)
        if not buffer:
            break
        n_bytes = len(buffer)
        file_size_dl += n_bytes
        memfile.write(buffer)
        tqdm_pb.update(n_bytes)

    memfile.seek(0)
    with fileio.open_file_write(local_fname) as f:
      f.write(memfile.getbuffer())
  finally:
    if f is not None:
      f.close()
    memfile.close()

  file_size_local = fileio.get_file_size(local_fname)
  if file_size_local != file_size:
      raise ValueError(f"URL file {url} is {file_size} bytes but we only downloaded {file_size_local} bytes to local file {local_fname}.")
  else:
    if display:
      print(f"Successfully downloaded {file_size_local}/{file_size} bytes from URL file {url} to local file {local_fname}!")

  return nested_tqdm_pb


def download_to_memfile(url, block_sz=8192, display=False):
  http_file = urllib.request.urlopen(url)
  if http_file.getcode() != 200:
      raise ValueError(f"ERROR {http_file.getcode()} while opening {url}.")

  meta = http_file.info()
  file_size = int(meta['Content-Length'])
  # if nested_tqdm_pb is None and display:
  if display:
    print(f"Downloading {url} (filesize: {file_size} bytes)...")
    
  memfile = BytesIO()

  file_size_dl = 0
  fblocks = range(0, file_size, block_sz)
  for fblock in fblocks: 
      buffer = http_file.read(block_sz)
      if not buffer:
          break
      n_bytes = len(buffer)
      file_size_dl += n_bytes
      memfile.write(buffer)

  memfile.seek(0)

  if file_size_dl != file_size:
      raise ValueError(f"URL file {url} is {file_size} bytes but we only downloaded {file_size_dl} bytes.")
  else:
    if display:
      print(f"Successfully downloaded {file_size_dl}/{file_size} bytes from URL file {url}!")
  
  return memfile


def boostrap_target_video_index(d_vid_indexes_info):
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

  memfile = download_to_memfile(remote_archive_path, block_sz=8192, display=False)
  zip_ref = zipfile.ZipFile(memfile, 'r')
  print(f"unzipping {remote_archive_path} in-memory...")
  # zip_ref.printdir()
  sel_vid_index_path = d_vid_indexes_info['sel_vid_index_path']
  sel_vid_index_path_suffix = d_vid_indexes_info['video_indexes_archive'].split('.')[0]+'/'+sel_vid_index_path.split('/')[-1]
  sel_vid_index_fname = sel_vid_index_path_suffix.split('/')[-1]
  # print(f"we need to pull {sel_vid_index_path_suffix} out of in-memory extracted archive")
  bytes_unzipped = zip_ref.read(sel_vid_index_path_suffix)
  zip_ref.close()
  if not fileio.path_exists(d_vid_indexes_info['vid_indexes_dir'])[0]:
    fileio.make_dirs(d_vid_indexes_info['vid_indexes_dir'])
  with fileio.open_file_write(d_vid_indexes_info['vid_indexes_dir']+'/'+sel_vid_index_fname) as f:
    f.write(bytes_unzipped)
    f.close()
  memfile.close()

  print(f"\tDONE")
  
  return d_vid_indexes_info['sel_vid_index_path']
