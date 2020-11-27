
# depracated

# FPS = 30

# def stitch_video_segments(df_video_index=df_video_index, dest_dir=STICHED_VIDEO_FRAMES_DIR, fps=30, overwrite=False):
#   try:
#     if overwrite:
#       try:
#         shutil.rmtree(dest_dir)
#       except:
#         pass
#     os.mkdir(dest_dir)
#   except:
#     pass

#   df_decomposition = pd.DataFrame(columns=['src_video', 'dest_dir', 'n_frames'])

#   print(f"Stitching video-segment frames into {len(df_video_index)} videos (corresponding frames)...")
#   tqdm_pb = tqdm(range(0, len(df_video_index), 1))
#   tqdm_pb.set_description(desc='Media Record')
#   nested_tqdm_pb = trange(1, leave=True)
#   nested_tqdm_pb.set_description(desc='Segment Frame-Decomposition')

#   media_record_iterator = df_video_index.iterrows()

#   failed_target_videos = []

#   for tfblock in tqdm_pb:
#     idx, media_record = next(media_record_iterator) 

#     target_video_fname = media_record['filename']
#     target_stitched_vid_frames_dir = os.path.join(STICHED_VIDEO_FRAMES_DIR, target_video_fname.split('.')[0])
#     try:
#       os.mkdir(target_stitched_vid_frames_dir)
#     except:
#       pass

#     remote_vid_paths = media_record['compressed_mov_url'].split(';') # this can be a list, separated by ';'
#     segment_fnames = [remote_vid_path.split('/')[-1] for remote_vid_path in remote_vid_paths]
#     local_vid_segment_paths = [os.path.join(VIDEO_DIR, segment_fname) for segment_fname in segment_fnames]

#     fail = False
#     n_stitched_frames = 0
#     for local_vid_segment_path in local_vid_segment_paths:
#       if not os.path.isfile(local_vid_segment_path):
#         print(f"***WARNING!!!*** Cannot stitch together target video {target_video_fname} since segment {local_vid_segment_path} does not exist locally")
#         failed_target_videos.append(target_video_fname)
#         fail = True
#         break
#       else:
#         vidcap = cv2.VideoCapture(local_vid_segment_path)
#         vidcap.set(cv2.CAP_PROP_FPS, fps)
#         # vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000) # just capture all frames
#         n_frames_expected = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if n_frames_expected>0:
#           fblocks = range(0, n_frames_expected, 1)
#           nested_tqdm_pb.leave = True
#           nested_tqdm_pb.reset(total=n_frames_expected)
#           nested_tqdm_pb.refresh(nolock=False)
#           success, frame = vidcap.read()
#           n_frames = 0
#           while success:
#             cv2.imwrite(os.path.join(target_stitched_vid_frames_dir, f"{n_stitched_frames}.jpg"), frame)
#             n_frames += 1
#             n_stitched_frames += 1
#             nested_tqdm_pb.update(1)
#             success, frame = vidcap.read()
#           if n_frames != n_frames_expected:
#             print(f"\t***WARNING!!!*** Cannot stitch together target video {target_video_fname} since {n_frames_expected} frames were expected from segment {local_vid_segment_path} but only {n_frames} were successfully extracted")
#             failed_target_videos.append(target_video_fname)
#             fail = True
#             break
#           df_decomposition.loc[len(df_decomposition)] = [local_vid_segment_path, target_stitched_vid_frames_dir, n_frames]
#         else:
#           print(f"\t***WARNING!!!*** Cannot stitch together target video {target_video_fname} since cv2.CAP_PROP_FRAME_COUNT reports segment {local_vid_segment_path} has zero frames")
#           failed_target_videos.append(target_video_fname)
#           fail = True
#           break
#     if fail:
#       try:
#         shutil.rmtree(target_stitched_vid_frames_dir)
#       except:
#         pass

#   print("\tDONE")
  
#   return df_decomposition