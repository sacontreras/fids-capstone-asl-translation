from api import fidscs_globals
from apache_beam.io.filesystems import FileSystems
from tensorflow.python.lib.io.file_io import FileIO
import tensorflow as tf

def dir_path_to_pattern(path):
  if path[-2:] != '/*':
    if path[-1] == '/':
      path += '*'
    else:
      path += '/*'
  return path

def gcs_correct_dir_path_form(dir_path, strip_prefix=False):
  if not dir_path.endswith('/'):
    dir_path += '/'
  if strip_prefix and dir_path[0:len(fidscs_globals.WORK_DIR)] == fidscs_globals.WORK_DIR:
    dir_path = dir_path[len(fidscs_globals.WORK_DIR)+1:]
  return dir_path


def path_join(path, endpoint):
  return FileSystems.join(path, endpoint)

def path_exists(path):
  return FileSystems.exists(gcs_correct_dir_path_form(path) if fidscs_globals.GCS_CLIENT is not None else path)


def list_dir(dir_path, exclude_subdir=False):
  if fidscs_globals.GCS_CLIENT is not None:
    gcs_form_dir_path = gcs_correct_dir_path_form(dir_path, strip_prefix=True)
    blobs_in_dir = list(fidscs_globals.GCS_CLIENT.list_blobs(fidscs_globals.GCS_BUCKET.name, prefix=gcs_form_dir_path))
    if len(blobs_in_dir)==1 and blobs_in_dir[0].name==gcs_form_dir_path:
      blobs_in_dir = []
    else:
      blobs_in_dir = [blob_in_dir.name for blob_in_dir in blobs_in_dir]
    if gcs_form_dir_path in blobs_in_dir:
      del blobs_in_dir[blobs_in_dir.index(gcs_form_dir_path)]
    blob_paths_in_dir = [blob_path_in_dir[len(gcs_form_dir_path):] for blob_path_in_dir in blobs_in_dir]
    if exclude_subdir:
      blob_paths_in_dir = list(filter(lambda blob_path_in_dir: not blob_path_in_dir.endswith('/'), blob_paths_in_dir))
    return blob_paths_in_dir
  else:
    return tf.io.gfile.listdir(dir_path)


def make_dirs(path):
  if fidscs_globals.GCS_CLIENT:
    gcs_form_path = gcs_correct_dir_path_form(path, strip_prefix=True)
    blob_path = fidscs_globals.GCS_BUCKET.blob(gcs_form_path)
    blob_path_create_result = blob_path.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')
    # print(f"\n{path}: {blob_path}, exists: {blob_path.exists(fidscs_globals.GCS_CLIENT)}")
    return blob_path_create_result
  else:
    return FileSystems.mkdirs(path)


def open_file_read(fpath):
  if fidscs_globals.GCS_CLIENT:
    gcs_form_path = gcs_correct_dir_path_form(fpath, strip_prefix=True)
    return fidscs_globals.GCS_IO.open(gcs_form_path)
  else:
    return FileSystems.open(fpath)


def open_file_write(fpath):
  if fidscs_globals.GCS_CLIENT:
    # gcs_form_path = gcs_correct_dir_path_form(fpath, strip_prefix=True)
    return fidscs_globals.GCS_IO.open(fpath, mode='w')
  else:
    return FileSystems.create(fpath)


def delete_paths(paths):
  return FileSystems.delete(paths)


def delete_file(path, recursive=False):
  if fidscs_globals.GCS_CLIENT:
    # print(f"delete_file (debug): path: {path}, recursive: {recursive}")
    if recursive:
      child_paths = list_dir(path, exclude_subdir=False)
      for child_path in child_paths:
        child_path = gcs_correct_dir_path_form(path, strip_prefix=True)+child_path
        delete_file(child_path, recursive=True)
    blob_path = fidscs_globals.GCS_BUCKET.blob(path)
    exists = blob_path.exists(fidscs_globals.GCS_CLIENT)
    if exists:
      # print(f"\n{path}: {blob_path}, exists: True (before delete attempt)")
      blob_path_delete_result = blob_path.delete(fidscs_globals.GCS_CLIENT)
      # print(f"\n{path}: {blob_path}, exists: {blob_path.exists(fidscs_globals.GCS_CLIENT)} (before delete attempt)\n")
      return blob_path_delete_result
    else:
      gcs_form_path = gcs_correct_dir_path_form(path, strip_prefix=True)
      blob_path = fidscs_globals.GCS_BUCKET.blob(gcs_form_path)
      exists = blob_path.exists(fidscs_globals.GCS_CLIENT)
      if exists:
        blob_path_delete_result = blob_path.delete(fidscs_globals.GCS_CLIENT)
        return blob_path_delete_result
      else:
        return True
  else:
    return delete_paths([path])


def get_file_size(fpath):
  if fidscs_globals.GCS_CLIENT:
    return fidscs_globals.GCS_IO.size(fpath)
  else:
    return FileIO(fpath, "rb").size()