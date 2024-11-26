"""
General user input handler and validation functions that the MXgap program uses.


Diego Ontiveros
"""

import os
import sys
from argparse import ArgumentParser

from mxgap.utils import add_path_ending, model_needsDOS, load_models_list
from . import PACKAGE_NAME


model_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'MODELS_LIST.txt')
models_list, models_list_string = load_models_list(model_list_path)

########################################################################
############################ Input Parsing #############################
########################################################################

def parse_user_input():
    """Uses ArgumentParser to get the user arguments and provide information."""

    # ArgumentParser parsing
    parser = ArgumentParser(usage="%(prog)s [-h] [-f CONTCAR [DOSCAR]] [-m MODEL] PATH",
                            description="Predict the PBE0 bandgap of terminated MXenes with trained ML models. \
                            For use, input the path of the folder of the calculation (with optimized CONTCAR and PBE DOSCAR present). \
                            The DOSCAR will be analyzed within a ±5 eV range from the Fermi level; it is recommended to select a sufficiently fine NEDOS for accurate results.\
                            This program is based on our works: J. Mater. Chem. A, 2023,11, 13754-13764; Energy Environ. Mater, 2024, 7, e12774")
    parser.add_argument("path",type=str,nargs="?",default=None,help="Specify the path to the directory containing the calculation output files, if empty, will select the current directory. Must contain at least the optimized CONTCAR, and the PBE DOSCAR for the PBE trained models.")
    parser.add_argument("-f","--files",type=str,nargs="+",required=False,help="Specify in order the direct CONTCAR and DOSCAR (if needed) paths manually. The path positional argument has preference over this.")
    parser.add_argument("-m","--model",type=str,default=None,help="Choose the trained MXene-Learning model to use. By default, the most accurate version is selected (RFR).")
    parser.add_argument("-o","--output",type=str,default=None,help="Path of the output file. By default it will generate a mxgap.info in the CONTCAR folder.")
    parser.add_argument("-l","--list", action="store_true",help="List of all trained ML models available to choose.")
    args = parser.parse_args()

    if args.list:
        print(models_list_string)
        sys.exit(0) 

    return args.path, args.model, args.files, args.output


########################################################################
########################### Input Validation ###########################
########################################################################

def input_path_exists(*paths):
    """Asserts that the paths given by the user exist."""
    for path in paths:
        if path is None: continue
        assert os.path.exists(path), f"The provided path {path} does not exist."


def model_exists(model:str,models_list):
    """Checks if the given model exists in the available list."""
    model_list = [m.strip() for m in model.split("+")]
    for m in model_list:
        assert m in models_list, f"The provided model {model} does not exist. Use {PACKAGE_NAME} -l to get the full list."


def validate_user_input(path,model,files,output=None,default_path="./", default_model="GBC+RFR_onlygap",default_output="mxgap.info"):
    """Validates the input given by the user. Checks input incompatibility, errors, etc.
    If valid, returns the CONTCAR, DOSCAR, and output paths."""

    if output is None:
        output = default_output
    else:
        if os.path.dirname(output) == "": output = "./" + output
        input_path_exists(os.path.dirname(output))

    if model is None:
        print(f"No ML model detected. The {default_model} model (most accurate) will be used.")
        model = default_model
    elif model == "best" or model == "default":
        model = default_model
    else: 
        model_exists(model,models_list)

    needDOS = model_needsDOS(model)

    if (path == None) and (files == None):
        print("WARNING: No folder path detected. The current directory will be used.")
        contcar_path, doscar_path = default_path + "CONTCAR", default_path + "DOSCAR"
    elif (path != None) and not (files == None):
        print(f"WARNING: Both folder path and -f files are given. The folder path {path} will take priority.")
        path = add_path_ending(path)
        contcar_path, doscar_path = path + "CONTCAR", path + "DOSCAR"
    elif (path != None) and (files == None):
        path = add_path_ending(path)
        contcar_path, doscar_path = path + "CONTCAR", path + "DOSCAR"
    elif (path == None) and not (files == None):
        if len(files) == 1 and needDOS:
            raise ValueError(f"DOSCAR file path not detected. For the {model} model used, the PBE DOSCAR file is needed.")
        if len(files) == 1 and not needDOS:
            contcar_path, doscar_path = files[0], None
        elif len(files) == 2 and needDOS:
            contcar_path, doscar_path = files
        elif len(files) == 2 and not needDOS:
            print(f"WARNING: The {model} model you selected does not need DOSCAR file. Calculation will continue.")
            contcar_path, doscar_path = files[0], None
        elif len(files) > 2:
            raise ValueError("Too many files provided. Please provide at most two files: CONTCAR and DOSCAR.")
        else: 
            raise ValueError("File paths not detected properly. Indicate the CONTCAR and DOSCAR (if needed) paths.")

    return contcar_path, doscar_path, model, output


def validate_user_files():
    """Validates the CONTCAR and DOSCAR files given by the user."""
    # To Do, although not necessary
    pass



if __name__ == "__main__":
    parse_user_input()
