===========
dml-project
===========


Perform transfer learning for object detection.


Description
===========

This project aims to perform object detection as a foundation for future work with Universeum Göteborg.
In this stage of the project, detection and recognition of different exotic bird species should be performed.

Note
====

The code has been written in Python 3.8.5. 

Needed to downgrade torch to 1.5 (and torchvision to 0.6 for compatibility) for some issues that arose with the custom data loaders.

The environment can be set up with e.g.

  virtualenv -p /usr/bin/python3.8 ~/virtualenv/dmlproj
  
  source ${HOME}/virtualenv/dmlproj/bin/activate
  
  pip install -r requirements.txt
  
For additional test support, when in the environment, execute

  pip install -r test-requirements.txt
  
To help linking modules within the project, execute
  
  python setup.py develop
  
[Also, note that since we included vision (from pytorch/torchvision) as a submodule, the code should be pulled via

  git pull origin main --recurse-submodules

... or similar.]

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
