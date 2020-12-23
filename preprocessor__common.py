import os
import utils
import zipfile
import globals


def boostrap_video_index(d_vid_indexes_info):
  """
  d_vid_indexes_info MUST be a dict as follows:
    {
      'vid_indexes_dir': VIDEO_INDEXES_DIR, 
      'sel_vid_index_path': SELECTED_VIDEO_INDEX_PATH, 
      'video_indexes_archive': VIDEO_INDEXES_ARCHIVE, 
      'tmp_dir': TMP_DIR
    }

  this function downloads d_vid_indexes_info['video_indexes_archive'] from http://www.bu.edu/asllrp/ncslgr-for-download
    and extracts it to os.path.join(d_vid_indexes_info['tmp_dir'], d_vid_indexes_info['video_indexes_archive'])
    (assuming that has not already been done - i.e. if not os.path.isdir(d_vid_indexes_info['vid_indexes_dir']) or not os.path.isfile(d_vid_indexes_info['sel_vid_index_path']))

  this function returns d_vid_indexes_info['sel_vid_index_path'] only after the above has been done
  """

  if not os.path.isdir(d_vid_indexes_info['vid_indexes_dir']) or not os.path.isfile(d_vid_indexes_info['sel_vid_index_path']):
    print(f"VIDEO INDEX BOOTSTRAP INFO: {d_vid_indexes_info}")
    remote_archive_path = os.path.join('http://www.bu.edu/asllrp/ncslgr-for-download', d_vid_indexes_info['video_indexes_archive'])
    local_archive_parent_dir = d_vid_indexes_info['tmp_dir']
    local_archive_path = os.path.join(local_archive_parent_dir, d_vid_indexes_info['video_indexes_archive'])
    utils.download(
        remote_archive_path, 
        local_archive_path, 
        block_sz=globals._1MB
    )
    zip_ref = zipfile.ZipFile(local_archive_path, 'r')
    print(f"unzipping {local_archive_path} to {d_vid_indexes_info['vid_indexes_dir']}...")
    zip_ref.extractall(local_archive_parent_dir)
    zip_ref.close()
    print(f"\tDONE")
    print(f"deleting {local_archive_path}...")
    os.remove(local_archive_path)
    print(f"\tDONE")
  else:
    print(f"Found video index {d_vid_indexes_info['sel_vid_index_path']}")
  return d_vid_indexes_info['sel_vid_index_path']