
import setuptools

# NOTE: Any additional file besides the `main.py` file has to be in a module
#       (inside a directory) so it can be packaged and staged correctly for
#       cloud runs.

# change to latest !
REQUIRED_PACKAGES = [
    'google-auth==1.21.1',
    'apache-beam[gcp]==2.25.*',
    'apache-beam[interactive]==2.25.*',
    'tensorflow-transform==0.26.*',
    'tensorflow==2.3.*',   
    'opencv-python==4.4.*',
    'protobuf==3.12.2',
    'absl-py==0.9',
    'numpy==1.16.5'
]

setuptools.setup(
    name='sac-fids-capstone-asl-translation',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description='Deep ASL',
)