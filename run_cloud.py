"""
python ./fids_capstone_asl_translation__dataflow__main.py \
  --work-dir gs://<YOUR-BUCKET-ID> \
  --max-target-videos <-1 for ALL | n max videos to process> \
  --beam-gcp-project YOUR-PROJECT \
  --beam-gcp-region us-central1 \
  --dataflow-job-name fids-capston-asl-translation-$USER
"""

from __future__ import absolute_import

import argparse
import logging
import os

from api import data_extractor

if __name__ == '__main__':
#   logging.getLogger().setLevel(logging.INFO) # too much output!

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
    '--work-dir',
    required=True,
    help='Directory for staging and working files. '
          'This should be a Google Cloud Storage path.'
  )

  parser.add_argument(
    '--max-target-videos',
    type=int,
    default=-1,
    help='Maximum number of target videos to process. '
          'Set to -1 to download/process all available target videos (and segments).'
  )

  parser.add_argument(
    '--beam-gcp-project',
    required=True,
    help='The GCP project containing the GCS bucket to use for beam temp as well as data storage.'
  )

  parser.add_argument(
    '--beam-gcp-region',
    required=True,
    help='The GCP region of the bucket.'
  )

  parser.add_argument(
    '--beam-gcp-dataflow-job-name',
    required=True,
    help='The name of the GCP Dataflow job to create.'
  )

  parser.add_argument(
    '--beam-gcp-dataflow-setup-file',
    default=None,
    help='The path to the setup.py file (used by Apache Beam worker nodes).'
  )

  args = parser.parse_args()
  print(f"args: {args}")

  # data_extractor.run(
  #   max_target_videos=args.max_target_videos if args.max_target_videos!=-1 else None, 
  #   data_dir=os.path.join(args.work_dir, 'data'), 
  #   use_beam=True,
  #   beam_runner='DataflowRunner',
  #   beam_gcp_project=args.beam_gcp_project,
  #   beam_gcp_region=args.beam_gcp_region,
  #   beam_gcp_dataflow_job_name=args.beam_gcp_dataflow_job_name,
  #   beam_gcp_dataflow_setup_file=args.beam_gcp_dataflow_setup_file
  # )

  # test with DirectRunner in GCP console
  data_extractor.run(
    max_target_videos=args.max_target_videos if args.max_target_videos!=-1 else None, 
    data_dir=os.path.join(args.work_dir, 'data'), 
    use_beam=True,
    beam_runner='DirectRunner'
  )
