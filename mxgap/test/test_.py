import os
import pytest

import numpy as np
from sklearn.base import BaseEstimator

from mxgap.dos import Doscar
from mxgap.structure import Structure
from mxgap.features import get_contcar_array, get_doscar_array
from mxgap.input import validate_user_input
from mxgap.utils import load_pickle, get_structure_indices, model_needsDOS, model_type, is_close
from mxgap.ML import ML_prediction


def get_test_info(case):
    """Creates a dictionary with the test MXene and information to test."""
    path = os.path.join(examples_folder, case, "test_info.txt")
    data = dict()
    with open(path,"r") as inFile:
        for line in inFile:
            key, value = line.strip().split()
            data[key] = value
    return data

models_folder = "mxgap/models/models/"
examples_folder = "mxgap/test/examples/"
examples_cases = os.listdir(examples_folder)
examples_dict = {mxt:get_test_info(mxt) for mxt in examples_cases}


examples_user_input = [
['test/examples/La2C1Cl2', 'RFC_notDOS', None, None,('test/examples/La2C1Cl2/CONTCAR', 'test/examples/La2C1Cl2/DOSCAR', 'RFC_notDOS','mxgap.info')],
[None, 'SVC+RFR_edges', ["test/examples/La2C1N2H2/CONTCAR","test/examples/La2C1N2H2/DOSCAR"], None,('test/examples/La2C1N2H2/CONTCAR', 'test/examples/La2C1N2H2/DOSCAR', 'SVC+RFR_edges','mxgap.info')],
['test/examples/La2C1Te2/', None, ["test/examples/La2C1Te2/CONTCAR","test/examples/La2C1Te2/DOSCAR"], None, ('test/examples/La2C1Te2/CONTCAR', 'test/examples/La2C1Te2/DOSCAR', 'GBC+RFR_onlygap','mxgap.info')],
['test/examples/La3C2O2H2', 'best', None, None, ('test/examples/La3C2O2H2/CONTCAR', 'test/examples/La3C2O2H2/DOSCAR', 'GBC+RFR_onlygap','mxgap.info')],
['test/examples/La3C2S2', 'model_not_exists', None, AssertionError, ('test/examples/La3C2S2/CONTCAR', 'test/examples/La3C2S2/DOSCAR', 'MLPR_edges','mxgap.info')],
[None, 'RFR', ["test/examples/La2C1N2H2/CONTCAR","test/examples/La2C1N2H2/DOSCAR", "extra"], ValueError, ('test/examples/La3N2H2/CONTCAR', 'test/examples/La3N2H2/DOSCAR', 'RFR','mxgap.info')],
[None, 'SVC+SVR', ["test/examples/La2C1N2H2/CONTCAR"], ValueError, ('test/examples/La4C3Cl2/CONTCAR', 'test/examples/La4C3Cl2/DOSCAR', 'SVR_notDOS','mxgap.info')],
[None, None, None, None, ('./CONTCAR', './DOSCAR', 'GBC+RFR_onlygap','mxgap.info')]
]

########################################################################
############################## Utils test #############################
########################################################################

class TestUtils:
    @pytest.mark.parametrize("path, expected_exception", [
        (models_folder + "GBC", None),
        (models_folder + "RFR", None),
        (models_folder + "SVR_onlygap", None),
        (models_folder + "KRR_onlygap_notDOS", None),
        (models_folder + "fail", FileNotFoundError)
    ])
    def test_load_pickle(self,path, expected_exception):
        if expected_exception is not None:
            # Expecting an exception
            with pytest.raises(expected_exception):
                load_pickle(path)
        else:
            # Expecting a successful result; check that the result is not None or empty
            result = load_pickle(path)
            
            # Check for properties
            assert result is not None                   # Check if the result is not None
            assert isinstance(result,BaseEstimator)     # Check if is sklern model
            assert hasattr(result,"predict")            # Check if it has .predict()


    @pytest.mark.parametrize("stack, hollow, expected_exception, expected_result", [
        ("ABC","HM", None, (0,0)),
        ("ABA","H", None, (1,0)),
        ("ABC","HMX", None, (0,1)),
        ("ABA","HX", None, (1,2)),
        ("ABA","HM", ValueError, None),
        ("ABC","H", ValueError, None),
    ])
    def test_get_structure_indices(self,stack,hollow,expected_exception,expected_result):
        if expected_exception is not None:
            # Expecting an exception
            with pytest.raises(expected_exception):
                get_structure_indices(stack,hollow)
        else:
            # Expecting a successful result; check that the result is not None or empty
            result = get_structure_indices(stack,hollow)
            
            # Check for properties
            assert result == expected_result


    @pytest.mark.parametrize("model, expected_result", [
        ("GBC+RFR", True),
        ("KRR", True),
        ("MLPC+GBC_notDOS", True),
        ("RFR_onlygap_notDOS", False),
    ])
    def test_model_needsDOS(self,model,expected_result):
        assert model_needsDOS(model) == expected_result


    @pytest.mark.parametrize("model, expected_result", [
        ("RFR", "R"), ("KRR", "R"), ("RFR_onlygap_notDOS", "R"),
        ("LR", "C"), ("MLPC_notDOS", "C"), ("RFC", "C")
    ])
    def test_model_type(self,model,expected_result):
        assert model_type(model) == expected_result


########################################################################
############################## Input test ##############################
########################################################################

class TestInput:

    @pytest.mark.parametrize("path,model,files,expected_exception,expected_result", examples_user_input)
    def test_validate_user_input(self,path,model,files,expected_exception,expected_result):
        if expected_exception is not None:
            with pytest.raises(expected_exception):
                validate_user_input(path,model,files,output=None)
        else:
            result = validate_user_input(path,model,files,output=None)
            assert result == expected_result


########################################################################
############################### DOS test ###############################
########################################################################

class TestDOS:

    @pytest.mark.parametrize("case, expected_result", [
        (k,(float(v["Ef"]),float(v["ispin"]))) for k,v in examples_dict.items()
    ])
    def test_get_dos(self,case,expected_result):
        doscar = Doscar(os.path.join(examples_folder, case, "DOSCAR"))
        E,dos,Ef,ispin = doscar.get_dos()
        assert Ef == expected_result[0]
        assert ispin == expected_result[1]


    @pytest.mark.parametrize("case, expected_result", [
        (k,float(v["Eg"])) for k,v in examples_dict.items()
    ])
    def test_get_bandgap(self,case,expected_result):
        doscar = Doscar(os.path.join(examples_folder, case, "DOSCAR"))
        Eg = doscar.get_bandgap()
        assert Eg == expected_result


    @pytest.mark.parametrize("case, expected_result", [
        (k,os.path.join(examples_folder, k, "doscar_array.npy")) for k,v in examples_dict.items()
    ])
    def test_make_histogram(self,case,expected_result):
        doscar = Doscar(os.path.join(examples_folder, case, "DOSCAR"))
        dos_hist, E_hist = doscar.make_histogram()
        assert np.all(dos_hist == np.load(expected_result)[3:])


########################################################################
############################ Structure test ############################
########################################################################

class TestStructure:

    @pytest.mark.parametrize("case, expected_result", [
        (k,(float(v["a"]),float(v["d"]))) for k,v in examples_dict.items()
    ])
    def test_get_geom(self,case,expected_result):
        structure = Structure(os.path.join(examples_folder, case, "CONTCAR"))
        assert np.all(np.isclose(structure.get_geom(),expected_result,atol=1e-5))


    @pytest.mark.parametrize("case, expected_result", [
        (k,(str(v["stack"]),str(v["hollow"]))) for k,v in examples_dict.items()
    ])
    def test_get_stack_hollows(self,case,expected_result):
        structure = Structure(os.path.join(examples_folder, case, "CONTCAR"))
        assert structure.get_stack_hollows() == expected_result


    @pytest.mark.parametrize("case, expected_result", [
        (k,int(v["nT"])) for k,v in examples_dict.items()
    ])
    def test_get_n_terminations(self,case,expected_result):
        structure = Structure(os.path.join(examples_folder, case, "CONTCAR"))
        assert structure.get_n_terminations() == expected_result


    @pytest.mark.parametrize("case, expected_result", [
        (k,int(v["rT"])) for k,v in examples_dict.items()
    ])
    def test_get_n_terminations(self,case,expected_result):
        structure = Structure(os.path.join(examples_folder, case, "CONTCAR"))
        assert structure.get_repeated_T() == expected_result


    @pytest.mark.parametrize("case, expected_result", [
        (k,(str(v["M"]),str(v["X"]),str(v["T"]))) for k,v in examples_dict.items()
    ])
    def test_getMXT(self,case,expected_result):
        structure = Structure(os.path.join(examples_folder, case, "CONTCAR"))
        assert tuple(e[0] for e in structure.getMXT(symbols=True)) == expected_result


########################################################################
############################ Features test #############################
########################################################################

class TestFeatures:

    @pytest.mark.parametrize("case, expected_result", [
        (k,os.path.join(examples_folder, k, "contcar_array.npy")) for k,v in examples_dict.items()
    ])
    def test_get_contcar_array(self,case,expected_result):
        contcar_array = get_contcar_array(os.path.join(examples_folder, case, "CONTCAR"))
        assert is_close(contcar_array,np.load(expected_result),atol=1e-3)


    @pytest.mark.parametrize("case, expected_result", [
        (k,os.path.join(examples_folder, k, "doscar_array.npy")) for k,v in examples_dict.items()
    ])
    def test_get_doscar_array(self,case,expected_result):
        doscar_array = get_doscar_array(os.path.join(examples_folder, case, "DOSCAR"))
        assert is_close(doscar_array,np.load(expected_result),atol=1e-3)

    #! Should test join_data_array or make_data_array (not really necessary)


########################################################################
############################### ML test ################################
########################################################################

@pytest.mark.parametrize("case, expected_result", [
    (k,(int(v["ML_isgap"]),float(v["ML_gap"]))) for k,v in examples_dict.items()
])
def test_main(case,expected_result):
    contcar_path = os.path.join(examples_folder,case,"CONTCAR")
    doscar_path = os.path.join(examples_folder,case,"DOSCAR")
    ML_isgap,ML_gap = ML_prediction(contcar_path,doscar_path)
    
    assert ML_isgap == expected_result[0]
    assert ML_gap == expected_result[1]