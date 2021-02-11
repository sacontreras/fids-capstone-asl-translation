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
          'This can be a Google Cloud Storage path.'
  )

  parser.add_argument(
    '--max-target-videos',
    type=int,
    default=-1,
    help='Maximum number of target videos to process. '
          'Set to -1 to download/process all available target videos (and segments).'
  )

  # courtesy of https://stackoverflow.com/a/43357954
  #   script --use_beam
  #   script --use_beam <bool>
  def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
  parser.add_argument(
    "--use-beam", 
    type=str2bool, 
    nargs='?',
    const=True, 
    default=True,
    help=""
  )

  parser.add_argument(
    '--beam-gcp-project',
    default=None,
    help='The GCP project containing the GCS bucket to use for beam temp as well as data storage.'
  )

  parser.add_argument(
    '--beam-gcp-region',
    default=None,
    help='The GCP region of the bucket.'
  )

  parser.add_argument(
    '--beam-gcp-dataflow-job-name',
    default=None,
    help='The name of the GCP Dataflow job to create.'
  )

  parser.add_argument(
    '--beam-gcp-dataflow-setup-file',
    default=None,
    help='The path to the setup.py file (used by Apache Beam worker nodes).'
  )

  args = parser.parse_args()
  print(f"args: {args}")

  data_extractor.run(
    max_target_videos=args.max_target_videos if args.max_target_videos!=-1 else None, 
    work_dir=args.work_dir,
    use_beam=args.use_beam,
    beam_runner='DirectRunner',
    beam_gcp_project=args.beam_gcp_project,
    beam_gcp_region=args.beam_gcp_region,
    beam_gcp_dataflow_job_name=args.beam_gcp_dataflow_job_name,
    beam_gcp_dataflow_setup_file=args.beam_gcp_dataflow_setup_file
  )
