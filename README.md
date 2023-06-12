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

## Running the code
The first step is to convert the ROOT files into pandas dataframes, saved as parquet files. This script weights each event according to the cross sections and preselection efficiency, so that the MC weights correspond to the expected event yield after selection for 1fb^-1 of pp collision data. The only input parameter is the path to the directory containing the ROOT files:
```
python root_to_parquet.py --input-dir /path/to/root/samples/
```

There is a second script which will then merge the events into a single dataframe. In this script you must define the processes (samples) that you want to include and the input features which you want to save. THe output will be a single dataframe in the same directory named `input_data.parquet`:
```
python process_data.py
```

To train the classifier you can use the `train.py` script. The list of training features are included within the script, which you can change accordingly. By default, the sum of weights of each output class will be equalised so that the classifier is not biased towards one particular class. Also events with negative weights are treated by taking the absolute value of the weight. This may not be the most optimal approach.
```
python train.py --input-file /path/to/root/samples/input_data.parquet
```
This script will output a new dataframe named `output_data.parquet` with the ML classifer output appended. You will need to configure the BDT class in `training/bdt_utils.py` for your specific use case. In the future I will make this more automatic.


To then evaluate the performance of the classifier you can use the `evaluate.py` script. You will need to change the `plot_path` option in each function to point to one of your directories.
```
python evaluate.py --input-file /path/to/root/samples/output_data.parquet --do-input-features --do-ml-output --do-roc --do-confusion-matrix
```
