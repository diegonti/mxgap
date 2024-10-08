"""
General utility functions that the MXgap program uses.


Diego Ontiveros
"""
import pickle
from datetime import datetime

import numpy as np

from mxgap import PACKAGE_NAME

########################################################################
############################ ML Models list ############################
########################################################################

classifiers     = ["GBC","RFC","SVC","MLPC","LR"]
regressors      = ["GBR","RFR","SVR","MLPR","KRR"]
regression_mode = ["full","onlygap","edges"]
levels          = ["DOS","notDOS"]

########################################################################
######################### Printing Functions ###########################
########################################################################

def print2(file,text,mode="a"):
    """Print text both into screen and file."""
    print(text)
    with open(file,mode) as outFile:
        outFile.write(text +  "\n")


def print_clf(file,pred):
    text = f"Predicted ML_isgap  =  {pred} ({'Semiconductor' if pred else 'Metallic'})"
    print2(file,text)


def print_reg(file,pred):
    text = f"Predicted ML_gap    =  {pred:.3f}"
    print2(file,text)


def print_header(file,path,model,contcar_path,doscar_path):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""
====================================================================
                            MXgap Report                           
====================================================================

Date:            {current_datetime}
Model Used:      {model}
Folder Path:     {path}
CONTCAR file:    {contcar_path}
DOSCAR file:     {doscar_path}

====================================================================
    """
    print2(file,report,mode="w")



########################################################################
########################## General Functions ###########################
########################################################################

def load_pickle(path):
    """Reads the data of a saved pickle object."""
    with open(path,"rb") as inFile:
        data = pickle.load(inFile)
    return data


def add_path_ending(path):
    """Adds / to ending of path if needed."""
    if path[-1] == "/": pass
    elif path[-1] == "\\": pass
    elif "/" in path: path = path + "/"
    elif "\\" in path: path = path + "\\"
    elif path == ".": path = "./"

    return path


def is_iterable(obj):
    """Assess if an object is iterable."""
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_close(a1,a2,atol=5e-3):
    """Assess if two arrays are close, within a certain tolerance."""
    return np.all(np.isclose(a1,a2,atol=atol))


def round_zeros(arr, tol=1e-7):
    """Sets to actual zero the values close to zero in an aray, with a given tolerance."""
    arr[np.isclose(arr, 0, atol=tol)] = 0.0
    return arr


def get_structure_indices(stack:str,hollow:str):
    """Gets the structure indices from the given stack and hollows."""
    try: stack_i = ["ABC","ABA"].index(stack)
    except ValueError: ("Stacking not detected. Check structure.")

    try: hollow_i = [["HM","HMX","HX"],["H","HMX","HX"]][stack_i].index(hollow)
    except ValueError: raise ValueError("Hollow termination position not detected. Check structure.")

    return stack_i, hollow_i



########################################################################
######################### ML Models Uitilities #########################
########################################################################

def load_models_list(path:str):
    """Loads available models list from file."""
    models_list = []
    with open(path,"r") as outFile:
        models_list_string = outFile.read()
        outFile.seek(0)
        for i,line in enumerate(outFile):
            if i==0: continue
            models_list.append(line.strip().split())

        models_list = sum(models_list, [])

    return models_list, models_list_string


def load_normalization(path:str):
    """Loads the min/max normalization parameters used when training the models."""
    norm_array = np.loadtxt(path,usecols=[1,2])
    norm_x, norm_y = norm_array[:-3],norm_array[-3:]
    norm_x_contcar, norm_x_doscar = norm_x[:-3], norm_x[-3:]

    return norm_x_contcar,norm_x_doscar, norm_y


def model_needsDOS(model:str):
    """Checks if the selected model needs the DOSCAR file."""

    # By looking at model name or loading model and searching for model.n_features_in_ (33 for nPBE 136 for PBE)
    model_list = [m.strip() for m in model.split("+")]
    
    return any([not "notDOS" in m for m in model_list])
    

def model_type(model:str):
    """Return the model type (C or R)."""
    if any(clf in model for clf in classifiers):
        return "C"
    elif any(reg in model for reg in regressors):
        return "R"
    else:
        raise ValueError(f"Model {model} not available. Use {PACKAGE_NAME} -l tu get the full list of models.")
    

def reorder_model_list(model_list,m_type):
    """Returns a ordered model_list, following C+R order."""
    if m_type == ["R","C"]: model_list = model_list[::-1]
    elif m_type == ["C","R"]: pass
    else: raise ValueError(f"Combination {'+'.join(model_list)} not available, use a C+R combination. Run main.py -l to get the full list of models.")
    return model_list



def rescale(array,norm,case):
    """Rescales output, depending on the model case (bandgap or VBM/CBM)."""
    min, max = norm[case]
    reescaled_array = array*(max-min)+min

    return reescaled_array


if __name__ == "__main__":
    pass