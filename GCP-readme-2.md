From gcloud command shell:

## Set \<PROJECT_ID\> in GCP terminal
```
gcloud config set project <PROJECT_ID>
BEAM_GCP_PROJECT=$(gcloud config get-value project || echo $BEAM_GCP_PROJECT)
echo $BEAM_GCP_PROJECT
```

## Switch to virtual environment
```
virtualenv fids-capstone-asl-translation-env -p python3
```

## Activate virtual environment
```
source fids-capstone-asl-translation-env/bin/activate
```

## Clone the fids-capstone-asl-translation repo
```
git clone https://github.com/sacontreras/fids-capstone-asl-translation
```

## Change to the `fids-capstone-asl-translation` directory
```
cd fids-capstone-asl-translation
```

## Set up dependencies to run the apache beam pipelines (for dataflow)
```
python setup.py develop
```

python setup.py install (?)

## Mount GCS bucket as local file system
```
sudo echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
sudo curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
sudo apt -qq update
sudo apt -qq install gcsfuse
```

## Create GSC bucket
name: sc-fids-capstone-bucket-\<GCP-PROJECT\>
folder: fids-capstone-data

## Run the GCP Dataflow Apache Beam Pipeline!
```
BEAM_GCP_PROJECT=$(gcloud config get-value project || echo $BEAM_GCP_PROJECT)
./run-cloud --work-dir gs://sc-fids-capstone-bucket-$BEAM_GCP_PROJECT --max-target-videos -1 --beam-gcp-project $BEAM_GCP_PROJECT
```

python ./fids_capstone_asl_translation__dataflow__main.py \
  --work-dir gs://sc-fids-capstone-bucket-$BEAM_GCP_PROJECT  \
  --max-target-videos -1 \
  --beam-gcp-project $BEAM_GCP_PROJECT  \
  --beam-gcp-region us-central1 \
  --dataflow-job-name fids-capston-asl-translation-$USER

python fids-capstone-asl-translation/fids_capstone_asl_translation__dataflow__main.py \
  --work-dir gs://sc-fids-capstone-bucket-$BEAM_GCP_PROJECT  \
  --max-target-videos -1 \
  --beam-gcp-project $BEAM_GCP_PROJECT  \
  --beam-gcp-region us-central1 \
  --dataflow-job-name fids-capston-asl-translation-$USER  