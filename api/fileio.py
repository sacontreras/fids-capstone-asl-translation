import logging
import os

import google.cloud.storage as gcs
import tensorflow as tf
from apache_beam.io.filesystems import FileSystems, GCSFileSystem
from apache_beam.io.gcp import gcsio
from tensorflow.python.lib.io.file_io import FileIO

from api import fidscs_globals


def get_gcs_client():
  return gcs.Client()

def get_gcs_bucket(d_pl_options):
  path_segments = d_pl_options[fidscs_globals.OPT_NAME_WORK_DIR][5:].split('/')
  gcs_bucket = path_segments[0]
  beam_gcp_project = d_pl_options[fidscs_globals.OPT_NAME_PROJECT]
  return gcs.Bucket(
    get_gcs_client(), 
    name=gcs_bucket, 
    user_project=beam_gcp_project
  )

def dir_path_to_pattern(path):
  if path[-2:] != '/*':
    if path[-1] == '/':
      path += '*'
    else:
      path += '/*'
  return path

def gcs_path_strip_prefix(path, d_pl_options):
  work_dir = d_pl_options[fidscs_globals.OPT_NAME_WORK_DIR]
  if path[0:len(work_dir)] == work_dir:
    path = path[len(work_dir)+1:]
  return path

def gcs_correct_dir_path_form(dir_path, d_pl_options, strip_prefix=False):
  if not dir_path.endswith('/'):
    dir_path += '/'
  if strip_prefix:
    dir_path = gcs_path_strip_prefix(dir_path, d_pl_options)
  return dir_path


def path_join(path, endpoint):
  return FileSystems.join(path, endpoint)

def path_exists(path, d_pl_options, is_dir=True):
  dir_path = path
  fs = FileSystems.get_filesystem(dir_path)
  if type(fs) == GCSFileSystem:
    dir_path = gcs_correct_dir_path_form(dir_path, d_pl_options, strip_prefix=False) if is_dir else path
  return FileSystems.exists(dir_path), dir_path


def list_dir(dir_path, d_pl_options, exclude_subdir=False):
  fs = FileSystems.get_filesystem(dir_path)
  if type(fs) == GCSFileSystem:
    return gcsio.GcsIO().list_prefix(gcs_correct_dir_path_form(dir_path, d_pl_options, strip_prefix=False))
  else:
    return tf.io.gfile.listdir(dir_path)


def make_dirs(path, d_pl_options):
  fs = FileSystems.get_filesystem(path)
  if type(fs) == GCSFileSystem:
    gcs_form_path = gcs_correct_dir_path_form(path, d_pl_options, strip_prefix=True)
    blob_path = get_gcs_bucket(d_pl_options).blob(gcs_form_path)
    blob_path_create_result = blob_path.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')
    return blob_path_create_result
  else:
    dir_creation_result = None
    try:
      dir_creation_result = FileSystems.mkdirs(path)
    except Exception as e:
      # if e is not None:
      #   print(e)
      pass
    return dir_creation_result


def open_file_read(fpath):
  fs = FileSystems.get_filesystem(fpath)
  if type(fs) == GCSFileSystem:
    return gcsio.GcsIO().open(fpath)
  else:
    return FileSystems.open(fpath)


def open_file_write(fpath):
  fs = FileSystems.get_filesystem(fpath)
  if type(fs) == GCSFileSystem:
    return gcsio.GcsIO().open(fpath, mode='w')
  else:
    return FileSystems.create(fpath)


def delete_paths(paths):
  return FileSystems.delete(paths)


def delete_file(path, d_pl_options, recursive=False, r_level=0, debug=False):
  fs = FileSystems.get_filesystem(path)
  if type(fs) == GCSFileSystem:
    gcs_client = get_gcs_client()
    if debug: print(f"{'-'*(r_level)} delete_file (debug): path: {path}, recursive: {recursive}")
    if recursive:
      child_paths = list_dir(path, d_pl_options, exclude_subdir=False)
      for child_path in child_paths:
        child_path_gcs_corrected = gcs_correct_dir_path_form(path, d_pl_options, strip_prefix=False)+child_path
        if debug: print(f"{'-'*(r_level+1)} delete_file (debug): path {path} has child: {child_path} (gcs-corrected: {child_path_gcs_corrected})")
        delete_file(child_path_gcs_corrected, d_pl_options, recursive=True, r_level=r_level+1)

    # not stripped, not corrrected case
    blob_path = get_gcs_bucket(d_pl_options).blob(path)
    path_not_stripped_not_gcs_corrected_exists = blob_path.exists(gcs_client)
    if debug: print(f"{'-'*(r_level+1)} {path} (not stripped, not gcs corrected): {blob_path}, exists: {path_not_stripped_not_gcs_corrected_exists}")
    if path_not_stripped_not_gcs_corrected_exists:
      if debug: print(f"{'-'*(r_level+1)} {path} (not stripped, not gcs corrected): {blob_path}, exists: True (before delete attempt)")
      blob_path_delete_result = blob_path.delete(gcs_client)
      if debug: print(f"{'-'*(r_level+1)} {path} (not stripped, not gcs corrected): {blob_path}, exists: {blob_path.exists(gcs_client)} (after delete attempt)")
      return blob_path_delete_result
    else:
      # not stripped, gcs corrected case
      path_not_stripped_gcs_corrected = gcs_correct_dir_path_form(path, d_pl_options, strip_prefix=False)
      blob_path = get_gcs_bucket(d_pl_options).blob(path_not_stripped_gcs_corrected)
      path_not_stripped_gcs_corrected_exists = blob_path.exists(gcs_client)
      if debug: print(f"{'-'*(r_level+1)} {path_not_stripped_gcs_corrected} (not stripped, gcs corrected): {blob_path}, exists: {path_not_stripped_gcs_corrected_exists}")
      if path_not_stripped_gcs_corrected_exists:
        if debug: print(f"{'-'*(r_level+1)} {path_not_stripped_gcs_corrected} (not stripped, gcs corrected): {blob_path}, exists: True (before delete attempt)")
        blob_path_delete_result = blob_path.delete(gcs_client)
        if debug: print(f"{'-'*(r_level+1)} {path_not_stripped_gcs_corrected} (not stripped, gcs corrected): {blob_path}, exists: {blob_path.exists(gcs_client)} (after delete attempt)")
        return blob_path_delete_result
      else:
        # stripped, not gcs corrected case
        path_stripped_not_gcs_corrected = gcs_path_strip_prefix(path, d_pl_options)
        blob_path = get_gcs_bucket(d_pl_options).blob(path_stripped_not_gcs_corrected)
        path_stripped_not_gcs_corrected_exists = blob_path.exists(gcs_client)
        if debug: print(f"{'-'*(r_level+1)} {path_stripped_not_gcs_corrected} (stripped, not gcs corrected): {blob_path}, exists: {path_stripped_not_gcs_corrected_exists}")
        if path_stripped_not_gcs_corrected_exists:
          if debug: print(f"{'-'*(r_level+1)} {path_stripped_not_gcs_corrected} (stripped, not gcs corrected): {blob_path}, exists: True (before delete attempt)")
          blob_path_delete_result = blob_path.delete(gcs_client)
          if debug: print(f"{'-'*(r_level+1)} {path_stripped_not_gcs_corrected} (stripped, not gcs corrected): {blob_path}, exists: {blob_path.exists(gcs_client)} (after delete attempt)")
          return blob_path_delete_result
        else:
          # stripped, gcs corrected case
          path_stripped_gcs_corrected = gcs_correct_dir_path_form(path, d_pl_options, strip_prefix=True)
          blob_path = get_gcs_bucket(d_pl_options).blob(path_stripped_gcs_corrected)
          path_stripped_gcs_corrected_exists = blob_path.exists(gcs_client)
          if debug: print(f"{'-'*(r_level+1)} {path_stripped_gcs_corrected} (stripped, gcs corrected): {blob_path}, exists: {path_stripped_gcs_corrected_exists}")
          if path_stripped_gcs_corrected_exists:
            if debug: print(f"{'-'*(r_level+1)} {path_stripped_gcs_corrected} (stripped, gcs corrected): {blob_path}, exists: True (before delete attempt)")
            blob_path_delete_result = blob_path.delete(gcs_client)
            if debug: print(f"{'-'*(r_level+1)} {path_stripped_gcs_corrected} (stripped, gcs corrected)): {blob_path}, exists: {blob_path.exists(gcs_client)} (after delete attempt)")
            return blob_path_delete_result
          else:
            if debug: print(f"{'-'*(r_level+1)} out of options trying to delete base path {path}!")
            return False

  else:
    return delete_paths([path])


def get_file_size(fpath):
  fs = FileSystems.get_filesystem(fpath)
  if type(fs) == GCSFileSystem:
    return gcsio.GcsIO().size(fpath)
  else:
    return FileIO(fpath, "rb").size()
