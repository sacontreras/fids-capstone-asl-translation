from __future__ import absolute_import

import urllib
import urllib.request
import zipfile
from io import BytesIO

from tqdm.auto import tqdm, trange

from api import beam__common, fidscs_globals


def download(url, local_fname, block_sz=8192, display=True, nested_tqdm_pb=None):
  http_file = urllib.request.urlopen(url)
  if http_file.getcode() != 200:
      raise ValueError(f"ERROR {http_file.getcode()} while opening {url}.")

  meta = http_file.info()
  file_size = int(meta['Content-Length'])
  if nested_tqdm_pb is None and display:
    print(f"Downloading {url} (filesize: {file_size} bytes) to {local_fname}...")
    print()

  f = None
  try:
    memfile = BytesIO()

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
    with beam__common.open_file_write(local_fname) as f:
      f.write(memfile.getvalue())
  finally:
    if f is not None:
      f.close()

  # file_size_local = os.path.getsize(f.name)
  file_size_local = beam__common.get_file_size(local_fname)
  if file_size_local != file_size:
      raise ValueError(f"URL file {url} is {file_size} bytes but we only downloaded {file_size_local} bytes to local file {local_fname}.")
  else:
    if display:
      print(f"Successfully downloaded {file_size_local}/{file_size} bytes from URL file {url} to local file {local_fname}!")

  return nested_tqdm_pb


# def download_to_memfile(url, block_sz=8192, display=False, nested_tqdm_pb=None):
def download_to_memfile(url, block_sz=8192, display=False):
  http_file = urllib.request.urlopen(url)
  if http_file.getcode() != 200:
      raise ValueError(f"ERROR {http_file.getcode()} while opening {url}.")

  meta = http_file.info()
  file_size = int(meta['Content-Length'])
  # if nested_tqdm_pb is None and display:
  if display:
    print(f"Downloading {url} (filesize: {file_size} bytes)...")
    print()
    
  memfile = BytesIO()

  file_size_dl = 0
  fblocks = range(0, file_size, block_sz)
  # if nested_tqdm_pb is None:
  #   tqdm_pb = tqdm(fblocks, disable=not display)
  # else:
  #   tqdm_pb = nested_tqdm_pb
  #   tqdm_pb.leave = True
  # tqdm_pb.reset(total=file_size)
  # tqdm_pb.refresh(nolock=False)
  for fblock in fblocks: 
      buffer = http_file.read(block_sz)
      if not buffer:
          break
      n_bytes = len(buffer)
      file_size_dl += n_bytes
      memfile.write(buffer)
      # tqdm_pb.update(n_bytes)

  memfile.seek(0)

  if file_size_dl != file_size:
      raise ValueError(f"URL file {url} is {file_size} bytes but we only downloaded {file_size_dl} bytes.")
  else:
    if display:
      print(f"Successfully downloaded {file_size_dl}/{file_size} bytes from URL file {url}!")
  
  # return memfile, nested_tqdm_pb
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
  local_archive_path = beam__common.path_join(local_archive_parent_dir, d_vid_indexes_info['video_indexes_archive'])
  video_ds_path = d_vid_indexes_info['video_ds_path']

  print(f"VIDEO-INDEX BOOTSTRAP INFO: {d_vid_indexes_info}")
  download(
      remote_archive_path, 
      local_archive_path, 
      block_sz=fidscs_globals._1MB
  )
  zip_ref = zipfile.ZipFile(local_archive_path, 'r')
  print(f"unzipping {local_archive_path} to {d_vid_indexes_info['vid_indexes_dir']}...")
  zip_ref.extractall(local_archive_parent_dir)
  zip_ref.close()
  print(f"\tDONE")
  print(f"deleting {local_archive_path}...")
  # os.remove(local_archive_path)
  beam__common.delete_file(local_archive_path)
  print(f"\tDONE")
  
  return d_vid_indexes_info['sel_vid_index_path']
