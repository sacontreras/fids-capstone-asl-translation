"""
python fids_capstone_asl_translation__dataflow__main \
  --work-dir gs://<YOUR-BUCKET-ID>
  --max-target-videos <-1 for ALL | n max videos to process>
  --dataflow-job-name fids-capston-asl-translation-$USER \
  --beam-gcp-project YOUR-PROJECT \
  --beam-gcp-region us-central1 \
  --beam-gcp-setup-file ./setup.py \
  --beam-gcs-staging-location gs://<YOUR-BUCKET-ID>/staging \
  --beam-gcs-temp-location gs://<YOUR-BUCKET-ID>/tmp
"""

from __future__ import absolute_import

import logging

# from apache_beam.examples.complete.juliaset.juliaset import juliaset
from api import data_extractor
import os

import argparse

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  # juliaset.run()

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
    '--dataflow-job-name',
    required=True,
    help='The name of the GCP Dataflow job to create.'
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
    '--beam-gcs-staging-location',
    default=None,
    help='The path to the Apache Beam staging location.'
          'Full path, prepended with GCS bucket - e.g. gs://<your GCS bucket id>/beam-staging'
  )

  parser.add_argument(
    '--beam-gcs-temp-location',
    default=None,
    help='The GCS path for Apache Beam temp storage.'
          'Full path, prepended with GCS bucket - e.g. gs://<your GCS bucket id>/beam-temp'
  )

  parser.add_argument(
    '--beam-gcp-setup-file',
    default=None,
    help='The (local) path to the python setup file (used by worker nodes to install dependencies).'
  )

  args = parser.parse_args()
  print(f"args: {args}")


  data_extractor.run(
    max_target_videos=args.max_target_videos if args.max_target_videos!=-1 else None, 
    data_dir=os.path.join(args.work_dir, 'data'), 
    beam_gcp_project=args.beam_gcp_project, 
    beam_gcp_region=args.beam_gcp_region, 
    beam_gcp_setup_file=args.beam_gcp_setup_file, 
    beam_gcs_staging_location=args.beam_gcs_staging_location, 
    beam_gcs_temp_location=args.beam_gcs_temp_location, 
    beam_runner='DataflowRunner', 
    use_beam=True
  )