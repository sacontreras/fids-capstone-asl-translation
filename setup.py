
import setuptools

# NOTE: Any additional file besides the `main.py` file has to be in a module
#       (inside a directory) so it can be packaged and staged correctly for
#       cloud runs.

# this is based on the dependencies article for runninng Apache Beam pipelines remotely, 
#   found at https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/

# also, this setup.py follows the example given (listed) in the article above,
#   from https://github.com/apache/beam/blob/master/sdks/python/apache_beam/examples/complete/juliaset/setup.py

# change to latest !
REQUIRED_PACKAGES = [
    # 'google-auth==1.21.1',
    'google-auth==1.21.1',

    # 'apache-beam[gcp]==2.25.*',
    'apache-beam==2.26.0',
    # 'apache-beam[interactive]==2.25.*',
    'apache-beam[interactive]==2.26.0',

    # 'tensorflow-transform==0.26.*',
    'tensorflow-transform==0.26.0',

    # 'tensorflow==2.3.*',
    'tensorflow==2.3.0',

    # 'avro-python3==1.8.1',
    'avro-python3==1.9.2.1',

    # 'opencv-python==4.4.*',
    'opencv-python==4.4.0.46',

    # 'protobuf==3.12.2',
    'protobuf==3.13.0',

    # 'absl-py==0.9',
    'absl-py==0.10.0',

    # 'numpy==1.16.5',
    'numpy==1.18.5',

    'tqdm==4.56.0'
    # tqdm @ file:///home/conda/feedstock_root/build_artifacts/tqdm_1594937875116/work
]

setuptools.setup(
    name='sac-fids-capstone-asl-translation',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description='Steven Contreras FIDS Capston Project: Deep ASL',
)