"""
Functions related to perform the ML prediction.

Diego Ontiveros
"""

import os
from time import time

from mxgap.features import make_data_array
from mxgap.utils import load_normalization, load_pickle, model_needsDOS, rescale, \
                            model_type, reorder_model_list, print2, print_header, print_predictions
from mxgap.input import validate_user_input, input_path_exists
from mxgap import PACKAGE_NAME


def model_prediction(model_str, data_array, return_proba=False):
    """Loads model and makes ML prediction, including (optional) probabilities for classifiers."""
    model = load_pickle(models_path + model_str)
    has_proba = callable(getattr(model, "predict_proba", None))
    
    if return_proba and has_proba: 
        probabilities = model.predict_proba([data_array]).T[1][0]
        prediction = model.predict([data_array])[0]
        return prediction, probabilities
    elif return_proba and not has_proba:
        print(f"WARNING: Model {model_str} does not support probabilities. Ignoring the -p flag.")
        prediction = model.predict([data_array])[0]
        return prediction
    else:
        prediction = model.predict([data_array])[0]
        return prediction

def process_edge_predictions(pred, norm_y):
    """Process edge-based regressor predictions."""
    pred_rescaled = rescale(pred, norm_y, 1)
    ML_VBM, ML_CBM = round(pred_rescaled[0], 3), round(pred_rescaled[1], 3)
    ML_gap = round(ML_CBM - ML_VBM, 3)

    return [ML_VBM, ML_CBM, ML_gap]


def handle_classifier_and_regressor(model_list, m_type, data_array_dict, norm_y, output, return_proba=False):
    """Handle cases where a classifier and a regressor are used together."""

    # Ensure models are ordered as classifier + regressor
    clf_str, reg_str = reorder_model_list(model_list, m_type)

    # Classifier prediction
    clf_pred = model_prediction(clf_str, data_array_dict[model_needsDOS(clf_str)], return_proba=return_proba)

    if isinstance(clf_pred, tuple):
        ML_isgap, ML_proba = clf_pred
        ML_proba = round(ML_proba, 3)
    else:
        ML_isgap = clf_pred
        ML_proba = None

    if ML_isgap == 1:  # Only proceed with regressor if a gap is predicted
        reg_pred = model_prediction(reg_str, data_array_dict[model_needsDOS(reg_str)])
        if "_edges" in reg_str:
            ML_VBM, ML_CBM, ML_gap = process_edge_predictions(reg_pred, norm_y)
            print_predictions(output, isgap=ML_isgap, prob=ML_proba, vbm=ML_VBM, cbm=ML_CBM, gap=ML_gap)
            return [ML_isgap, ML_VBM, ML_CBM, ML_gap] + ([ML_proba] if return_proba else [])
        else:
            ML_gap = round(rescale(reg_pred, norm_y, 0), 3)
            print_predictions(output, isgap=ML_isgap, prob=ML_proba, gap=ML_gap)
            return [ML_isgap, ML_gap] + ([ML_proba] if return_proba else [])
    else:
        print_predictions(output, isgap=ML_isgap, prob=ML_proba, gap=0)
        return [ML_isgap, 0] + ([ML_proba] if return_proba else [])


def handle_single_model(model, m_type, data_array_dict, norm_y, output, return_proba=False):
    """Handle cases where only a single model (classifier or regressor) is used."""
    pred = model_prediction(model, data_array_dict[model_needsDOS(model)], return_proba=return_proba)

    if m_type == "R":
        if "_edges" in model:
            ML_VBM, ML_CBM, ML_gap = process_edge_predictions(pred, norm_y)
            print_predictions(output, vbm=ML_VBM, cbm=ML_CBM, gap=ML_gap)
            return [ML_VBM, ML_CBM, ML_gap]
        else:
            ML_gap = round(rescale(pred, norm_y, 0), 3)
            print_predictions(output, gap=ML_gap)
            return [ML_gap]
    elif m_type == "C":
        if isinstance(pred,tuple):
            ML_isgap, ML_proba = pred
            ML_proba = round(ML_proba, 3)
            print_predictions(output, isgap=ML_isgap, prob=ML_proba)
            return [ML_isgap, ML_proba]
        else:
            ML_isgap = pred
            print_predictions(output, isgap=ML_isgap)
            return [ML_isgap]



def ML_prediction(contcar_path:str,doscar_path:str,model:str="GBC+RFR_onlygap",output:str=None,return_proba:bool=False):
    """
    Main function for predicting bandgap with ML model, from CONTCAR and DOSCAR paths.

    Parameters
    -----------

    `contcar_path`  : Path for the CONTCAR file.
    `doscar_path`   : Path for the DOSCAR file.
    `model`         : ML model to use. Defaults to GBC+RFR_onlygap (best).
    `output`        : Path to save the output. Defaults to "mxgap.info" in the same directory as CONTCAR.
    `return_proba`  : Optional. Return also the probability of semiconductor class (p>=0.5: Semiconductor, p<0.5: Metallic), given by sklearn model.predict_proba().

    Returns
    ------------
    `pred` : Result of the prediction in a list. The length will vary depending on the used model and settings:
            \nSingle model
             If a single model is used (Classifier or Regressor) it will return 1 result (0/1 for metal/semiconductor if classification or bangap in case of regression). E.g. `[IsGap]` or `[gap]`.
             Except when using a R_edges approach, that it will return 3 results: VBM, CBM, and bandgap. E.g. `[VBM, CBM, gap]`.
            \nCombined model
             If a combination of Classifier and Regressor is used (C+R), it will return 2 results (0/1 for metal/semiconductor and bangap). E.g. `[IsGap, gap]`. 
             Except when using a R_edges approach, that it will return 4 results (Includes also VBM/CBM band edges). E.g. `[IsGap, VBM, CBM, gap]`.
            \nClassifier with probabilities
             If return_proba is True, it will add 1 more element at the end of the list, containing the class probability. E.g. `[IsGap, prob]` or `[IsGap, gap, prob]`.
    """

    if output is None:
        base_path = os.path.dirname(contcar_path)
        output = os.path.join(base_path, "mxgap.info")

    # Load normalization and data arrays
    norm_x_contcar, norm_x_doscar, norm_y = load_normalization()
    data_array_dict = {True: make_data_array(contcar_path, doscar_path, True, norm_x_contcar, norm_x_doscar),
                       False: make_data_array(contcar_path, doscar_path, False, norm_x_contcar, norm_x_doscar),}

    # Parse models and determine types
    model_list = [m.strip() for m in model.split("+")]
    m_type = [model_type(m) for m in model_list]

    # Single or combined model handling
    if len(model_list) == 2:
        return handle_classifier_and_regressor(
            model_list, m_type, data_array_dict, norm_y, output, return_proba=return_proba
        )
    elif len(model_list) == 1:
        return handle_single_model(
            model_list[0], m_type[0], data_array_dict, norm_y, output, return_proba=return_proba
        )
    else:
        raise ValueError(f"Model {model} not available. Use {PACKAGE_NAME} -l to get the full list of models.")


def run_prediction(path:str=None, model:str=None, files:list=None, output:str=None, return_proba:bool=False):
    """Main function for predicting bandgap with ML model. Does the validation of inputs.

    Parameters
    ----------
    `path`   : Optional. Path of the folder of a calculation, where the CONTCAR and DOSCAR are found. By default cwd.
    `model`  : Optional. ML model to use. By default GBC+RFR_onlygap (best).
    `files`  : Optional. Specify the paths for the CONTCAR and DOSCAR files, in a list. By default None. 
               Use either `paths` or `files`, if both are specified, `path` will take preference.
    `output` : Optional. Specify the output file. By default it will generate a mxgap.info in the CONTCAR folder. 
    `return_proba`  : Optional. Return also the probability of semiconductor class (p>=0.5: Semiconductor, p<0.5: Metallic), given by sklearn model.predict_proba().

    Returns
    ------------
    `pred` : Result of the prediction in a list. The length will vary depending on the used model and settings:
            \nSingle model
             If a single model is used (Classifier or Regressor) it will return 1 result (0/1 for metal/semiconductor if classification or bangap in case of regression). E.g. `[IsGap]` or `[gap]`.
             Except when using a R_edges approach, that it will return 3 results: VBM, CBM, and bandgap. E.g. `[VBM, CBM, gap]`.
            \nCombined model
             If a combination of Classifier and Regressor is used (C+R), it will return 2 results (0/1 for metal/semiconductor and bangap). E.g. `[IsGap, gap]`. 
             Except when using a R_edges approach, that it will return 4 results (Includes also VBM/CBM band edges). E.g. `[IsGap, VBM, CBM, gap]`.
            \nClassifier with probabilities
             If return_proba is True, it will add 1 more element at the end of the list, containing the class probability. E.g. `[IsGap, prob]` or `[IsGap, gap, prob]`.

    """
    print()
    initial_time = time()

    contcar_path, doscar_path, model, output = validate_user_input(path, model, files, output, default_path, default_model, default_output)

    input_path_exists(contcar_path, doscar_path)
    #! validate_files() (validate they are actual CONTCAR/DOSCAR files, maybe not necessary)

    # Open output file and write report (#! verbosity?)
    base_path = os.path.dirname(contcar_path)
    output = os.path.join(base_path,output).replace("\\","/") if output == default_output else output
    print_header(output,path,model,contcar_path,doscar_path,output)

    pred = ML_prediction(contcar_path,doscar_path,model,output,return_proba=return_proba)
    
    final_time = time()
    print2(output,f"\nFinished successfully in {final_time-initial_time:.2f}s")

    return pred


# Initialization of some paths
default_path    =   "./"
default_model   =   "GBC+RFR_onlygap"
default_output  =   "mxgap.info"
models_path     =   os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'models/')


if __name__ == "__main__":
    contcar_path    = "test/examples/La2C1Cl2/CONTCAR"
    doscar_path     = "test/examples/La2C1Cl2/DOSCAR"
    model           = "GBC+RFR_onlygap"
    ML_isgap, ML_gap = ML_prediction(contcar_path,doscar_path,model)
