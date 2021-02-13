# Introduction and Requirements

## IMPORTANT NOTES!!!

1. These instructions are for executing the **ETL** pipeline that extracts data and stores it to Google Cloud Storage.
2. DOING SO WILL INCUR A COST TO *YOU*!  I TAKE NO RESPONSIBILITY FOR ANY CHARGES FROM Google THAT YOU INCUR BY RUNNING THIS PROCEDURE!  **The ETL pipeline downloads (to GCS) more than 2600 videos!  It then extracts more than 500,000 frames from those videos, so consider your choice whether you want to do this for yourself carefully!**
3. When it is time to run the ETL pipeline, it is HIGHLY recommended to turn off your VPN in order to reduce the overall time required to run it (considering ETL uploads to GCS).  Additionally, having your VPN on can cause undue retries, further ext


## (Software) Requirements

### 1. Python 3.7.* (or higher)

### 2. virtualenv (20.4.2)
```
pip install virtualenv
```

### 3. Google Cloud Platform (GCP) SDK Command-line Tools

Please refer to GCP SDK installation instructions [here](https://cloud.google.com/sdk/docs/install).


### 4. gcsfuse

Please refer to gcsfuse installation instructions [here](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/installing.md).

### 5. Docker Desktop

**For Windows**, please efer to *Docker Desktop* installation [here](https://docs.docker.com/docker-for-windows/install/).

**For Mac**, please efer to *Docker Desktop* installation [here](https://docs.docker.com/docker-for-mac/install/).

<p><br><p><br>

## Setup Steps

All steps below are preferably executed via local bash.  This can be done via gcloud command shell but you will have to upload your Google Cloud Service Credentials to your instance.  See the **Configure Google Cloud Credentials** below.

### 1. Create New GCP Project via Google Cloud Console (browser)

**Note: *THIS STEP ONLY NEEDS TO BE ONCE, THE FIRST TIME***.

### 2. Initialize the GCP API in bash (or GCP console shell)

**Note: *THIS STEP IS EXCLUSIVELY EXECUTED WHEN DOING THIS VIA LOCAL BASH (not GCP cloud console shell)***.

```
gcloud init
```

This will authenticate and "connect" your local bash session to GCP (assuming you have installed and setup the GCP SDK correctly).

At some point, running this command in bash will open a new browser tab for YOU to authenticate to GCP (in order to use the GCP SDK from the command-line).

If it is done correctly, you will eventually see as output:
```
Your Google Cloud SDK is configured and ready to use!
```

### 3. Setup Google Cloud API (Service Account) Authentication

**Note: *THIS STEP ONLY NEEDS TO BE ONCE, THE FIRST TIME***.

Please refer to the instructions [here](https://cloud.google.com/docs/authentication/getting-started).

### a. After completing the above steps, move the file (downloaded to your local system) into an some appropriate location - e.g. somewhere in your home directory

### b. In bash (or GCP console shell), Make sure the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set and reflects the correct path

```
echo $(ls $GOOGLE_APPLICATION_CREDENTIALS)
```

*If this was done correctly, output will be non-empty and be the actual path to the JSON file that was created by GCP*.

### 4. Set `BEAM_GCP_PROJECT` Environment Variable
```
BEAM_GCP_PROJECT=$(gcloud config get-value project || echo $BEAM_GCP_PROJECT)
echo $BEAM_GCP_PROJECT
```

This should output the ID of the GCP Project you created.

#### a. If it doesn't, execute the following command:

```
gcloud config set project <GCP PROJECT ID>
```

gcloud config set project sc-fids-capstone

#### b. ... and, again, set `BEAM_GCP_PROJECT` Environment Variable:

```
BEAM_GCP_PROJECT=$(gcloud config get-value project || echo $BEAM_GCP_PROJECT)
echo $BEAM_GCP_PROJECT
```

### 5. Create New Folder (where all the work will be done)

From here on, this folder will be refered to as `<LOCAL PROJECT ROOT>`.

Note that the all source code (from my github repo) for the pipeline will be sync'ed and executed herein.

**Note: *THIS STEP ONLY NEEDS TO BE ONCE, THE FIRST TIME (unless you explicitly want to execute in a new location)***.

### 6. `cd` into `<LOCAL PROJECT ROOT>`

```
cd <LOCAL PROJECT ROOT>
```

### 7. Remove Existing `fids-capstone-asl-translation` and `fids-capstone-asl-translation-env` Sub-Directories

(*Note that, if you have already run these steps in the same (bash) session, you must first `deactivate` the currently activated `virtualenv` environment*.)

```
rm -rf fids-capstone-asl-translation
rm -rf fids-capstone-asl-translation-env
```

### 8. Create `fids-capstone-asl-translation-env` Virtual Environment
**Note: THIS REQUIRES Python 3.7.\* (or higher)**

Ensure this is the case via:
```
python --version
```

If so, then execute:

```
virtualenv fids-capstone-asl-translation-env -p python
```

### 9. Activate `fids-capstone-asl-translation-env` Virtual Environment
```
source fids-capstone-asl-translation-env/bin/activate
```

### 10. Clone My `fids-capstone-asl-translation` Github Repo
```
git clone https://github.com/sacontreras/fids-capstone-asl-translation
```

### 11. `cd` into the `fids-capstone-asl-translation` Sub-Directory
```
cd fids-capstone-asl-translation
```

### 12. Setup Dependencies (required to run the Apache Beam pipelines) in the `fids-capstone-asl-translation-env` Virtual Environment
```
python setup.py develop
```

This will take a couple of minutes for `setup.py` to download and install everything, depending on your local machine specs.

In the end, you should see final output:

```
Finished processing dependencies for sac-fids-capstone-asl-translation==0.0.1
```

### 13. Create GSC Bucket via Google Cloud Console (web)

**AGAIN, THIS WILL INCUR A COST TO *YOU*, WHICH I TAKE NO RESPONSIBILITY FOR!**

**In order to make absolutely sure this is something you might want to do, I am <u>intentionally</u> not including explicit steps to this.  So, if you really want to do this, you're going to have to do the research yourself.**

#### GCS Bucket Info
**name**: *sc-fids-capstone-bucket-\$BEAM_GCP_PROJECT*
**region**: *us-central1*

### 14. Mount GCS Bucket `sc-fids-capstone-bucket-\$BEAM_GCP_PROJECT` as `/tmp/fids-capstone-data` in your Local File System
```
mkdir /tmp/fids-capstone-data
gcsfuse --debug_fuse sc-fids-capstone-bucket-$BEAM_GCP_PROJECT /tmp/fids-capstone-data 
```

If you have completed all previous steps successfully, you will see output:

```
Using mount point: /tmp/fids-capstone-data
Opening GCS connection...
Mounting file system...
File system has been successfully mounted.
```

## 15. Run the ETL Apache Beam Pipeline!

**AGAIN, THIS WILL INCUR A COST TO *YOU* (SINCE IT WILL UPLOAD FILES TO YOUR GCS BUCKET), WHICH I TAKE NO RESPONSIBILITY FOR!**
```
FIDS_CAPSTONE_WRK_DIR=/tmp
echo $FIDS_CAPSTONE_WRK_DIR

FIDS_CAPSTONE_MAX_TARGET_VIDEOS=-1
echo $FIDS_CAPSTONE_MAX_TARGET_VIDEOS

FIDS_CAPSTONE_GCP_REGION=us-west2
echo $FIDS_CAPSTONE_GCP_REGION
```

OR

```
FIDS_CAPSTONE_WRK_DIR=gs://$BEAM_GCP_PROJECT-bucket-$BEAM_GCP_PROJECT
FIDS_CAPSTONE_MAX_TARGET_VIDEOS=-1
FIDS_CAPSTONE_GCP_REGION=us-west2

echo $FIDS_CAPSTONE_WRK_DIR
echo $FIDS_CAPSTONE_MAX_TARGET_VIDEOS
echo $FIDS_CAPSTONE_GCP_REGION
```




```
python ./run_local__etl.py \
  --work-dir $FIDS_CAPSTONE_WRK_DIR \
  --max-target-videos $FIDS_CAPSTONE_MAX_TARGET_VIDEOS \
  --use-beam 1 \
  --beam-gcp-project $BEAM_GCP_PROJECT \
  --beam-gcp-region $FIDS_CAPSTONE_GCP_REGION \
  --beam-gcp-dataflow-job-name $BEAM_GCP_PROJECT-etl \
  --beam-gcp-dataflow-setup-file ./setup.py
```

```
nohup python ./run_cloud__etl.py \
  --work-dir $FIDS_CAPSTONE_WRK_DIR \
  --max-target-videos $FIDS_CAPSTONE_MAX_TARGET_VIDEOS \
  --beam-gcp-project $BEAM_GCP_PROJECT \
  --beam-gcp-region $FIDS_CAPSTONE_GCP_REGION \
  --beam-gcp-dataflow-job-name $BEAM_GCP_PROJECT-etl \
  --beam-gcp-dataflow-setup-file ./setup.py &
```

Note that because there is some latency incurred in order to upload files to your GCS bucket (via `gcsfuse`), this takes a bit longer than if you were simply storing files locally.  But altogether there is more than 17 Gigs of data when all said and done.  Still, because the ETL pipeline does its processing in parallel, via Apache Beam, it should only take about 30 minutes to run.

### 16. Dismount GCS Bucket `sc-fids-capstone-bucket-\$BEAM_GCP_PROJECT` from `/tmp/fids-capstone-data` in your Local File System

**Note that this command differs based on operating system**.

According to [this post]()https://stackoverflow.com/a/42613541), the command for Linux operating systems is:
```
fusermount -u /tmp/fids-capstone-data
```

I personally use Mac, and the command is:
```
sudo umount /tmp/fids-capstone-data
```

# A Jupyter Notebook variant of this process has been captured within `etl.ipynb`
