"""
General utility functions that the MXgap program uses.


Diego Ontiveros
"""
import os
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


def print_reg(file,pred,type="gap"):
    text = f"Predicted ML_{type}    =  {pred:.3f}"
    print2(file,text)


def print_proba(file,pred):
    text = f"Class probability   =  {pred:.3f}"
    if abs(pred-0.5) < 0.05: text += "\n WARNING: Low confidence prediction. The predicted probability is near 0.5. This result may be unreliable. Consider additional analysis."
    print2(file,text)


def print_header(file,path,model,contcar_path,doscar_path,output):
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
Output Path:     {output}

====================================================================
    """
    print2(file,report,mode="w")


def print_predictions(output, isgap=None, prob=None, gap=None, vbm=None, cbm=None):
    """Print predictions to the output file."""
    if isgap is not None:
        print_clf(output, isgap)
    if prob is not None:
        print_proba(output, prob)
    if vbm is not None:
        print_reg(output, vbm, "VBM")
    if cbm is not None:
        print_reg(output, cbm, "CBM")
    if gap is not None:
        print_reg(output, gap, "gap")


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
    else: path = path + "/"

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
    """Sets to actual zero the values close to zero in an array, with a given tolerance."""
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
######################### ML Models Utilities #########################
########################################################################

def load_models_list():
    """Loads available models list from file. Returns the list and the printable string."""
    model_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'MODELS_LIST.txt')
    models_list = []
    with open(model_list_path,"r") as outFile:
        models_list_string = outFile.read()
        outFile.seek(0)
        for i,line in enumerate(outFile):
            if i==0: continue
            models_list.append(line.strip().split())

        models_list = sum(models_list, [])

    return models_list, models_list_string


def load_normalization():
    """Loads the min/max normalization parameters used when training the models.
    In order, it returns the CONTCAR, DOSCAR, and output normalization parameters."""
    norm_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'NORM_INFO.txt')
    norm_array = np.loadtxt(norm_path,usecols=[1,2])
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