"""
Reads input files and extracts the feature array that will be passed to the ML model.

Diego Ontiveros
"""

import numpy as np
from pymatgen.core import periodic_table

from mxgap.structure import Structure
from mxgap.dos import Doscar
from mxgap.utils import get_structure_indices



def get_contcar_array(contcar_path):
    """Gets the raw input data array from the CONTCAR file. """

    # CONTCAR information (geometry)
    contcar = Structure(contcar_path)

    geom = contcar.get_geom(extra=True)
    a, d, hMT1, hMT2, dMT1, dMT2, dXT1,dXT2, dMX1,dMX2 = geom

    stack, hollow = contcar.get_stack_hollows()
    stack_i, hollow_i = get_structure_indices(stack,hollow)

    # Periodic Table information 
    mxt = contcar.symbols.formula
    n = contcar.n

    contcar = contcar.sortMXT()
    M,X,T_name = contcar.getMXT(symbols=True)
    M,X,T_name = M[0], X[0], T_name[0] if T_name is not None else None
    
    M_el = periodic_table.get_el_sp(M)
    X_el = periodic_table.get_el_sp(X)
    T_el = periodic_table.get_el_sp(T_name)

    # non-DOSCAR data array (CONTCAR + Periodic Table)
    data_array = np.array([
            n, stack_i, hollow_i, a, d, hMT1, hMT2, dMT1, dMT2, dXT1, dXT2, dMX1, dMX2,
            M_el.Z, M_el.group, M_el.row, M_el.X, M_el.electron_affinity, M_el.van_der_waals_radius, M_el.atomic_radius.real,
            X_el.Z, X_el.group,           X_el.X, X_el.electron_affinity, X_el.van_der_waals_radius, X_el.atomic_radius.real,
            T_el.Z, T_el.group, T_el.row, T_el.X, T_el.electron_affinity, T_el.van_der_waals_radius, T_el.atomic_radius.real,
        ])
    
    return data_array
    

def get_doscar_array(doscar_path):
    """Gets the raw input data array from the DOSCAR file. """

    # DOSCAR information (Eg, VBM, CBM, DOS_hist)
    doscar = Doscar(doscar_path)
    Ef = doscar.ef
    Eg = doscar.get_bandgap()
    VBM, CBM = Ef, round(Ef+Eg,3)
    DOS_hist,E_hist = doscar.make_histogram()

    # DOSCAR data array
    dos_array = np.concatenate([[VBM, CBM, Eg],DOS_hist])
    return dos_array


def join_data_array(contcar_array,doscar_array,norm_x_contcar,norm_x_doscar):
    """Joins the raw data arrays into the final normalized array that goes into the ML model."""

    # Normalize each array
    contcar_array_normalized = (contcar_array - norm_x_contcar[:,0]) / (norm_x_contcar[:,1] - norm_x_contcar[:,0])
    
    if doscar_array is None:
        doscar_array_normalized = []
    else: 
        # normalize 3 first positions of doscar array (VBM, CBM, Eg)
        doscar_array_temp = (doscar_array[:3] - norm_x_doscar[:,0]) / (norm_x_doscar[:,1] - norm_x_doscar[:,0])
        doscar_array_normalized = np.concatenate([doscar_array_temp,doscar_array[3:]])

    # Join arrays
    data_array = np.concatenate([contcar_array_normalized,doscar_array_normalized])

    return data_array


def make_data_array(contcar_path,doscar_path,needDOS,norm_x_contcar,norm_x_doscar):
    """Gets the normalized data_array from the input paths."""

    # Get data arrays #!(Should be for each model, in the case of two models)
    contcar_array = get_contcar_array(contcar_path)
    if needDOS: doscar_array = get_doscar_array(doscar_path)
    else: doscar_array = None

    # Normalize and join arrays 
    data_array = join_data_array(contcar_array,doscar_array,norm_x_contcar,norm_x_doscar)
    #! save data_array (.pkl or .dat)
    # normalized or normal?

    return data_array



if __name__=="__main__":

    contcar_array = get_contcar_array("test/CONTCAR")
    doscar_array = get_doscar_array("test/DOSCAR")

    print(doscar_array)