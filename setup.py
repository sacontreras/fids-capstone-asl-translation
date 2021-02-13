# NOTE: Any additional file besides the `main.py` file has to be in a module
#       (inside a directory) so it can be packaged and staged correctly for
#       cloud runs.

# this is based on the dependencies article for runninng Apache Beam pipelines remotely, 
#   found at https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/

# also, this setup.py follows the example given (listed) in the article above,
#   from https://github.com/apache/beam/blob/master/sdks/python/apache_beam/examples/complete/juliaset/setup.py

from __future__ import absolute_import, print_function

import setuptools
import subprocess
from distutils.command.build import build as _build  # type: ignore


# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.
  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]


CUSTOM_COMMANDS = [
    ['echo', 'no custom commands to execute']
]
class CustomCommands(setuptools.Command):
  """A setuptools Command class able to run arbitrary commands."""
  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def RunCustomCommand(self, command_list):
    print('Running command: %s' % command_list)
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print('Command output: %s' % stdout_data)
    if p.returncode != 0:
      raise RuntimeError(
          'Command %s failed: exit code: %s' % (command_list, p.returncode))

  def run(self):
    for command in CUSTOM_COMMANDS:
      self.RunCustomCommand(command)


REQUIRED_PACKAGES = [
  'apache-beam',
  'tensorflow-transform',
  'google-cloud-storage',
  'google-auth',
  'absl-py',
  'opencv-python-headless',
  'tqdm'
]

setuptools.setup(
  name='sac-fids-capstone-asl-translation',
  version='0.0.1',
  description='Steven Contreras FIDS Capston Project: Deep ASL',

  install_requires=REQUIRED_PACKAGES,
  packages=setuptools.find_packages(),
  cmdclass={
      # Command class instantiated and run during pip install scenarios.
      'build': build,
      'CustomCommands': CustomCommands,
  }
  , include_package_data=True
)
