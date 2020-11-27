import urllib
import urllib.request
from tqdm.auto import tqdm, trange
import os
from io import BytesIO
import tensorflow as tf

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
    with tf.io.gfile.GFile(name=local_fname, mode='w') as f:
      f.write(memfile.getvalue())
  finally:
    if f is not None:
      f.close()

  file_size_local = os.path.getsize(f.name)
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