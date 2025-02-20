# MXgap: A Machine Learning Program to predict MXene Bandgaps

<br>
<p align="center">
<img src="https://raw.githubusercontent.com/diegonti/mxgap/master/tutorials/logo.png" alt= "MXgap logo" width=600>
</p>

[![PyPi](https://img.shields.io/pypi/v/mxgap)](https://pypi.org/project/mxgap/)
[![Tests](https://github.com/diegonti/mxgap/actions/workflows/python_tests.yaml/badge.svg)](https://github.com/diegonti/mxgap/actions/workflows/python_tests.yaml)



## About

`mxgap` is a computational tool designed to streamline electronic structure calculations for MXenes using hybrid functionals like PBE0. By employing Machine Learning (ML) models, `mxgap` predicts the PBE0 bandgap based on features extracted from a PBE calculation. Here’s a detailed overview of its functionality:

### 1. Feature Extraction:
- Automatically extracts essential features and key data from a PBE calculation output, specifically tailored for [VASP](https://www.vasp.at/) (Vienna *Ab initio* Simulation Package) outputs.
- It leverages the structural information from the CONTCAR file, and optionally users can choose to include the density of states (DOS) from the DOSCAR file to enhance prediction accuracy, depending on the selected ML model.
- The program is designed for periodic systems. So, currently the tool requires a *p*($1\times1$) terminated MXene unit cell in the CONTCAR file for proper functionality.

### 2. ML Prediction:
- Uses trained ML models to predict bandgap values, reducing the computational cost associated with performing full PBE0 calculations.
- Several ML models have been trained and are available to use. The default (and best) one is a combination of a Classifier (GBC) that discriminates metallic or semiconductor MXenes and a Regressor (RFR, trained with semiconductor MXenes) to predict the bandgap. More info about the ML models in the [models/](https://github.com/diegonti/mxgap/tree/master/mxgap/models) folder.


### 3. Output:
- Generates a report file, `mxgap.info`, which contains the ML predictions and results.


<br>

This program is based on the data gathered in our works: [*J. Mater. Chem. A*, 2023, 11, 13754-13764](https://doi.org/10.1039/D3TA01933K) and [*Energy Environ. Mater.*, 2024, 7, e12774](https://doi.org/10.1002/eem2.12774). 

<!-- And the ML program and results have been published in [paper4](paper4). If use this, please cite:
```
D. Ontiveros, S. Vela, F. Viñes, C. Sousa, _Journal_, Year, Volume, Pages. DOI: doi
```  -->


## Installation

`mxgap` works for python >= 3.9, and can be installed using the Python package manager `pip`:

```
pip install mxgap
```

If you use conda/anaconda, the safest thing to do is to create a new environment and then install `mxgap`:

```
conda create -n mxgap python
conda activate mxgap
pip install mxgap
```

If you wish, you can install the latest version of `mxgap` from GitHub source with the commands below:

```
git clone https://github.com/diegonti/mxgap.git
cd mxgap
pip install .
```

If installed via `pip`, it should handle all the dependencies, but If you encounter issues, install the exact package versions specified in requirements.txt using:

```
pip install -r requirements.txt
```

## Usage
The program is mainly used through the CLI:

```
mxgap [-h] [-f CONTCAR [DOSCAR]] [-m MODEL] [PATH]
```
With the arguments and options explained below:
```
positional arguments:
  path                  Specify the path to the directory containing the 
                        calculation output files, if empty, will select the
                        current directory. Must contain at least the optimized 
                        CONTCAR file and, for DOS-trained models, the PBE DOSCAR file.

options:
  -h, --help            Show this help message and exit.
  -f FILES [FILES ...], --files FILES [FILES ...]
                        Specify in order the direct CONTCAR and DOSCAR (if needed) paths manually. 
                        The path positional argument has preference over this.
  -m MODEL, --model MODEL
                        Choose the trained MXene-Learning model to use. 
                        By default, the most accurate version is selected (GBC+RFR_onlygap).
  -o OUTPUT, --output OUTPUT
                        Path of the output file. By default it will generate a mxgap.info in the CONTCAR folder.
  -p, --proba           Show also the probability of semiconductor class (p>=0.5: Semiconductor, p<0.5: Metallic), 
                        given by sklearn model.predict_proba().
  -v {0,1,2,3}, --verbose {0,1,2,3}
                        Verbosity level: 0 (None), 1 (File), 2 (Screen), 3 (Both). Defaults to 3.
  -l, --list            List of all trained ML models available to choose.
  -V, --version         Show program's version number and exit.
```
So, for a quick example, the below command will look for the CONTCAR and DOSCAR files in the specified folder and use the default (best) ML model to predict the bandgap:
```
mxgap examples/folder/
```
Or using the `-f` option to specify both CONTCAR and DOSCAR files:
```
mxgap -f path/to/CONTCAR path/to/DOSCAR
```

Also, the program can be imported as a python module. See the [Jupyter Notebook](https://github.com/diegonti/mxgap/blob/master/tutorials/tutorials.ipynb) for some tutorials. Here is a quick example:

```python
from mxgap import run_prediction

path         = "examples/La2C1Cl2/"
model        = "GBC+RFR"
prediction   =  run_prediction(path, model = model)
```


## Tests

The program has been tested using [GitHub Actions](https://github.com/diegonti/mxgap/blob/master/.github/workflows/python_tests.yaml) for Windows, Ubuntu and MacOS with python versions >=3.9. You can run tests locally using `pytest` in the project folder:
```
cd mxgap
pytest
```


## Help

To get information about the program and its use, run the command:

```
mxgap -h
```

For any more doubts or questions, feel free to contact [me](mailto:diegonti.doc@gmail.com).