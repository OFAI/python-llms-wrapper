# Installation

### 1. Python environment 

* If necessary create a new python environment
  * e.g. using Ana/Miniconda: `conda create -y -n llms_wrapper python=3.11`

### 2. Install package

* From local source:
  * clone the repository and change into the directory
  * run `pip install -e .` 
  * To install for development: `python -m pip  install -e .[dev]`
* From source directly from github: `pip install -U git+https://github.com/OFAI/python-llms-wrapper.git`
* To create a notebook kernel run `python -m ipykernel install --user --name=llms_wrapper`  
