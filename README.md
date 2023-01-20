# ml-home
Repo to store ML scripts

## Installation and setup
```
git clone https://github.com/jonathon-langford/ml-home.git
cd ml-home
```
Analysis scripts depend on a number of python packages. Suggested way is to create a conda environment. If you need to install the miniconda (version 3) package then run:
```
source install_conda.sh
```
To automatically initiate conda upon loading the terminal, please add the first line into your ~/.bashrc file (with the correct path if this was changed when installing the conda package). For users on the Imperial clusters, to avoid issues with python versions you may also need to add the second line into your ~/.bashrc:
```
source /vols/cms/${USER}/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH
```
When installed you need to build the environment with all the dependencies with the following command:
```
conda env create -f environment.yml
```
Then activate the environment with
```
conda activate mlenv
```
For nice plotting also pip install the mplhep library:
```
pip install mplhep
```
