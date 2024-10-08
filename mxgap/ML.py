"""
Functions related to perform the ML prediction.

Diego Ontiveros
"""
"""
TO DO:
- Implement regression_edges
- Add verbosity in ML_prediction/run_prediction/cli
- Add output filename option

"""

import os
from time import time

from mxgap.features import make_data_array
from mxgap.utils import load_normalization, load_models_list, load_pickle, model_needsDOS, rescale, \
                            model_type, reorder_model_list, print2, print_clf, print_reg, print_header
from mxgap.input import validate_user_input, input_path_exists
from mxgap import PACKAGE_NAME


def model_prediction(model_str,data_array):
    """Loads model and makes ML prediction."""

    model = load_pickle(models_path + model_str)
    model_pred = model.predict([data_array])[0]

    return model_pred


def ML_prediction(contcar_path:str,doscar_path:str,model:str="GBC+RFR_onlygap"):
    """
    Main function for predicting bandgap with ML model, from CONTCAR and DOSCAR paths.

    `contcar_path` : Path for the CONTCAR file.
    `doscar_path` : Path for the DOSCAR file.
    `model` : ML model to use. Defaults to GBC+RFR_onlygap (best).
    """

    #! TODO: Implement _edge models and results

    base_path = os.path.dirname(contcar_path)
    base_file = os.path.join(base_path,"mxgap.info")

    norm_x_contcar,norm_x_doscar, norm_y = load_normalization(norm_path)

    data_array_dict = {True: make_data_array(contcar_path,doscar_path, True, norm_x_contcar,norm_x_doscar),
                        False: make_data_array(contcar_path,doscar_path, False, norm_x_contcar,norm_x_doscar)}

    model_list = [m.strip() for m in model.split("+")]
    m_type = [model_type(m) for m in model_list]

    if len(model_list) == 2:        #### C+R case

        model_list = reorder_model_list(model_list,m_type)

        clf_str,reg_str = model_list

        clf_pred = model_prediction(clf_str, data_array_dict[model_needsDOS(clf_str)])

        if clf_pred == 1:
            reg_pred = model_prediction(reg_str, data_array_dict[model_needsDOS(reg_str)])
            reg_pred_rescaled = rescale(reg_pred,norm_y,0)      #! Adapt for _edges
        else: 
            reg_pred_rescaled = 0

        ML_isgap, ML_gap = clf_pred, round(reg_pred_rescaled,3)

        print_clf(base_file,ML_isgap)
        print_reg(base_file,ML_gap)

        return ML_isgap, ML_gap

    elif len(model_list) == 1:      #### C or R only case

        pred = model_prediction(model_list[0], data_array_dict[model_needsDOS(model_list[0])])

        if m_type[0] == "R":
            pred = round(rescale(pred,norm_y,0), 3)             #! Adapt for _edges
            print_reg(base_file,pred)
        elif m_type[0] == "C":
            print_clf(base_file,pred)
            pass
        
        return [pred]

    else:
        raise ValueError(f"Model {model} not available. Use {PACKAGE_NAME} -l tu get the full list of models.")
    

def run_prediction(path:str=None, model:str=None, files:list=None):
    """Main function for predicting bandgap with ML model. Does the validation of inputs.

    Parameters
    ----------
    `path`  : Optional. Path of the folder of a calculation, where the CONTCAR and DOSCAR are found. By default cwd.
    `model` : Optional. ML model to use. By default GBC+RFR_onlygap (best).
    `files` : Optional. Specify the paths for the CONTCAR and DOSCAR files, in a list. By default None. 
            Use either `paths` or `files`, if both are specified, `path` will take preference.
    """
    print()
    initial_time = time()
    
    contcar_path, doscar_path, model = validate_user_input(path, model, files, default_path, default_model)

    input_path_exists(contcar_path,doscar_path)
    #! validate_files() (validate they are actual CONTCAR/DOSCAR files, maybe not necessary)

    #! open file and write intro + results
    base_path = os.path.dirname(contcar_path)
    base_file = os.path.join(base_path,f"{PACKAGE_NAME}.info")
    print_header(base_file,path,model,contcar_path,doscar_path)


    pred = ML_prediction(contcar_path,doscar_path,model)
    
    #! better output print/file

    final_time = time()
    print2(base_file,f"\nFinished successfully in {final_time-initial_time:.2f}s")

    return pred


# Initialization of some paths
default_path    =   "./"
default_model   =   "GBC+RFR_onlygap"
models_path     =   os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'models/')
norm_path       =   os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'NORM_INFO.txt')
model_list_path =   os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'MODELS_LIST.txt')

models_list, models_list_string = load_models_list(model_list_path)

if __name__ == "__main__":
    contcar_path    = "test/examples/La2C1Cl2/CONTCAR"
    doscar_path     = "test/examples/La2C1Cl2/DOSCAR"
    model           = "GBC+RFR_onlygap"
    ML_isgap, ML_gap = ML_prediction(contcar_path,doscar_path,model)
