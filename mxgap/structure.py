"""
Module for modifying structures and extracting the information in an automatic way.
This program is optimized for 2D MXene structures.

Diego Ontiveros
"""

import numpy as np
from collections import Counter

from ase import Atoms
from ase.io import read, write
from ase.geometry import get_distances

from mxgap.utils import is_iterable, is_close, round_zeros

#################### Main Structure Class ####################

class Structure(Atoms):
    def __init__(self,path=None,**kwargs):
        """
        Creates an ASE object with the information of the given geometry file.
        Allows to modify its geometry with different functions and get parameters.
        Optimized for slab MXene compounds. Inherits from ase.Atoms.

        `path` : Path for to the geometry file. (e.g. CONTCAR)
        """
        if path:
            atoms = read(path)

            # Initialize the parent class (Atoms) with the data from the file
            super().__init__(symbols=atoms.get_chemical_symbols(), 
                             positions=atoms.get_positions(), 
                             cell=atoms.get_cell(), 
                             pbc=atoms.get_pbc(), 
                             **kwargs)
        else:
            # No path provided, directly call the parent constructor
            super().__init__(**kwargs)

        self.repeated_T = self.get_repeated_T()
        self.n = int((len(self) - self.get_n_terminations()*2 - 1) / 2)
        # self.a, self.d = self.get_geom()
        # self.stack, self.hollows = self.get_stack_hollows()
        # self.n_terminations = self.get_n_terminations()
       
        
    def write(self,path:str,format="vasp",sortMXT=False,**kwargs):
        """
        Writes the object to a geometry file. Any **kwargs will be passed to the ase.io.write() function.

        `format`  : Str. Format of the geometry file. Accepts any of the ase.io.formats types. By default vasp.
        `sortMXT` : Bool. For MXenes, if True, it follows M>X>T order on writing.
        """

        if sortMXT: atoms = self.sortMXT()
        else: atoms = self.copy()

        write(path,atoms,format=format,**kwargs)


    def get_geom(self,extra=False):
        """Returns lattice parameter a and width d in Angstrom. \n
        Returns extra distances if extra=True (For terminated MXenes)."""
        
        # Shift to zero (convenience) and get info
        self.to_zero()
        positions = self.get_positions()
        a = self.cell[0,0]
        d = positions[:,2].max()
        n_terminations = self.get_n_terminations()

        if extra and n_terminations > 0:
            # Indices are chosen following the sorting of self.getMXT() (by z)
            M,X,T = self.getMXT()
            hMT1 = M[0,2] - T[0,2]
            hMT2 = T[-1,2] - M[-1,2]

            dMT1 = get_distances(M[0],T[0],cell=self.cell,pbc=self.pbc)[-1][0][0]
            dMT2 = get_distances(M[-1],T[-1],cell=self.cell,pbc=self.pbc)[-1][0][0]

            dXT1 = get_distances(X[0],T[0],cell=self.cell,pbc=self.pbc)[-1][0][0]
            dXT2 = get_distances(X[-1],T[-1],cell=self.cell,pbc=self.pbc)[-1][0][0]

            dMX1 = get_distances(M[0],X[0],cell=self.cell,pbc=self.pbc)[-1][0][0]
            dMX2 = get_distances(M[-1],X[-1],cell=self.cell,pbc=self.pbc)[-1][0][0]

            if n_terminations > 1:
                # For now, the extra dist are given for the closest termination to the slab
                # If needed here could be implemented more distances (not my case)
                pass

            return a, d, hMT1,hMT2, dMT1,dMT2, dXT1,dXT2, dMX1,dMX2
        
        return a,d
   

    def get_stack_hollows(self):
        """Gets the stack and termination hollow position for pristine or terminated MXenes."""
        
        # Get geometry info and positions
        self.to_zero()
        n_terminations = self.get_n_terminations()
        M,X,T = [a[:, :2] if a is not None else None for a in self.getMXT(scaled=True)]

        # Simple hollow position, relative to X and M (pbc wrapped)
        hH = 2*X[0] - M[0]  
        hH = round_zeros(hH) % 1.0

        # Assess stacking
        M1,M2 = M[0], M[1]
        if np.all(np.isclose(M1,M2,atol=1e-3)): stack = "ABA"
        else: stack = "ABC"
        
        # Assess hollow position (if any)
        hollow = None
        if stack == "ABC" and n_terminations>0:
            if   is_close(T[0],M[1]) and is_close(T[1],M[-2]): hollow = "HM"
            elif is_close(T[0],M[1]) and is_close(T[1],X[-1]): hollow = "HMX"
            elif is_close(T[0],X[0]) and is_close(T[1],M[-2]): hollow = "HMX"
            elif is_close(T[0],X[0]) and is_close(T[1],X[-1]): hollow = "HX"
        elif stack == "ABA" and n_terminations>0:
            if   is_close(T[0],hH)   and is_close(T[1],hH):    hollow = "H"
            elif is_close(T[0],hH)   and is_close(T[1],X[-1]): hollow = "HMX"
            elif is_close(T[0],X[0]) and is_close(T[1],hH):    hollow = "HMX"
            elif is_close(T[0],X[0]) and is_close(T[1],X[-1]): hollow = "HX"
        #! add TOP hollow?

        return stack, hollow

    
    def get_n_terminations(self):
        """Returns the number of terminations (0 pristine, 1 T, 2, OH, etc). (For MXenes)"""

        n_atoms = len(set(self.get_chemical_symbols())) + self.get_repeated_T()

        return n_atoms - 2
    

    def get_repeated_T(self):
        """Returns the number of repeated atoms in the termination with X."""

        # Count the occurrences of each element
        element_counts = Counter(self.get_chemical_symbols())

        # Find the X symbol (should be one of 'C', 'N', or 'B')
        x_symbols = list(set(["C", "N", "B"]) & set(element_counts.keys()))

        # If multiple X symbols found, assume no repetition
        if len(x_symbols) != 1:
            return 0

        x_symbol = x_symbols[0]
        nX = element_counts[x_symbol]

        # Remove X symbol from element counts
        del element_counts[x_symbol]

        # Find the maximum count of remaining elements (M atoms)
        nM = max(element_counts.values())

        # Calculate repeated terminations (T)
        repeated_T = (nX - nM + 1) // 2

        self.repeated_T = repeated_T

        return repeated_T

    
    def getMXT(self,scaled=False,symbols=False):
        """
        Get M, X, T positions or symbols in order (sorted by z). For terminated MXenes.
        
        `scaled` : Bool. Will use scaled positions instead of absolute.
        `symbols` : Bool. Will return chemical symbols instead of positions.
        """
        self.to_zero()

        # For using fractional coordinates
        if scaled:  positions = self.get_scaled_positions()
        else: positions = self.get_positions()
        
        # To get the chemical symbols instead of the positions
        if symbols: 
            sorted_positions = np.array(sorted(zip(range(len(self)),self.get_chemical_symbols()), key=lambda i: positions[i[0]][2]))
            sorted_positions = sorted_positions.T[1]
        else: 
            sorted_positions = np.array(sorted(positions, key=lambda pos: pos[2]))

        # Get the indices of the atoms following z ordering
        n_terminations = self.get_n_terminations()
        n = int((len(self) - n_terminations*2 - 1) / 2)
        indices = np.arange(n_terminations, n_terminations + (n+1)*2, 2)

        M = sorted_positions[indices]
        X = sorted_positions[indices[:-1]+1]

        if n_terminations == 0: T = None
        else: T = sorted_positions[[n_terminations-1,-n_terminations]]

        return M, X, T


    def sortMXT(self):
        """Returns a M>X>T sorted version of the MXene."""

        # Get symbols for each atom type (M,X,T)
        M_pos,X_pos,T_pos = self.getMXT(symbols=False)
        M,X,T = self.getMXT(symbols=True)
        M,X,T = M[0], X[0], T[0] if T is not None else None
        extra_terminations = [symbol for symbol in self.get_chemical_symbols() if symbol not in {M, X, T}]

        priority = {M: 0, X: 1, T: 2}

        # Assign priorities to extra termination atoms
        for i, term in enumerate(sorted(set(extra_terminations))):
            priority[term] = 3 + i

        # Sort atoms by priority and z position
        # Special case for the lowest termination in repeated XT MXenes (Like in M,N,NH) is accounted
        sorted_indices = sorted(range(len(self)), 
                        key=lambda i: (priority[self[i].symbol], 
                                       self[i].position[2] if not ((self.repeated_T > 0) and (self[i].symbol == T) and is_close(self[i].position[2],T_pos.T[2].min())) else T_pos.T[2].max()-0.1))
        
        return self[sorted_indices]
      
     
    def to_zero(self):
        """Shifts positions of atoms to start at zero."""

        scaled_positions = self.get_scaled_positions()
        scaled_z = scaled_positions[:,2]

        # Un-wrap slab
        threshold = 0.8
        if np.any(scaled_z > threshold) and np.any(scaled_z < (1-threshold)):
            scaled_z[scaled_z > threshold] -= 1

        scaled_z -= scaled_z.min()
        self.set_scaled_positions(scaled_positions)


    def shift(self,displacement,scaled=True):
        """
        Shifts the structure a given vector and applies PBC. 
        If only one value is given, the z direction will be shifted.
        
        `displacement` : Value or 3D vector for the shift.
        `scaled` : Bool. If the vector is in fractional coordinates. By default True.
        """

        if is_iterable(displacement):
            displacement = np.array(displacement) @ self.cell if scaled else displacement
        else:
            displacement = [0, 0, displacement]
            
        self.translate(displacement)

        self.wrap()


    def add_vacuum(self,vacuum):
        """Adds the indicated vacuum to the slab."""
        self.to_zero()
        cell = self.get_cell()
        cell[2,2] = self.positions[:,2].max() + vacuum
        self.set_cell(cell)

    
    def addT(self,symbol,dist=1,hollow=None):
        """
        Adds single-atom termination (T) to the pristine MXene in the indicated hole position (hollows=HM/H,HMX,HX).\n
        HMX uses HM and HX for the lower and upper surfaces, respectively. Use HMXi for the inverted configuration. (For MXenes).

        `symbol` : Str. Chemical symbol of the termination.
        `dist`   : Float. Distance from the added termination to the slab. Defaults to 1.
        `hollow` : Str. Hollow position for the termination (HM/H, HMX, HX).
        """

        # Get geometry info and positions
        self.to_zero()
        d = self.positions[:,2].max()
        scaled_posz = (2*dist+d)/self.cell[2,2]

        stack, _hollows = self.get_stack_hollows()
        M,X,T = [a[:, :2] if a is not None else None for a in self.getMXT(scaled=True)]

        if symbol in self.get_chemical_symbols(): self.repeated_T += 1

        hH = 2*X[0] - M[0]  # Simple hollow posiion, relative to X and M

        # Shift upwards to make room for bottom T (T1)
        self.shift(dist)

        # Get the positions of T1, T2 depending on the hollow site 
        if stack == "ABC":
            if   hollow == "H":
                 print("Using H for ABC stacking, changing to HM ...")
                 hollow = "HM"
            if   hollow == "HM":   pos1,pos2 = M[1],M[-2]
            elif hollow == "HMX":  pos1,pos2 = M[1],X[-1]
            elif hollow == "HMXi": pos1,pos2 = X[0],M[-2]
            elif hollow == "HX":   pos1,pos2 = X[0],X[-1]
        elif stack == "ABA":
            if   hollow == "HM":  
                 print("Using HM for ABA stacking, changing to H ...")
                 hollow = "H"
            if   hollow == "H":    pos1,pos2 = hH, hH
            elif hollow == "HMX":  pos1,pos2 = hH, X[-1]
            elif hollow == "HMXi": pos1,pos2 = X[0], hH
            elif hollow == "HX":   pos1,pos2 = X[0],X[-1]
        #! add TOP hollow?

        # Add z component (scaled)
        pos1 = round_zeros(np.append(pos1,[0]))
        pos2 = round_zeros(np.append(pos2,[scaled_posz]))

        # Add new T atoms to self object
        T1 = Atoms(symbol,scaled_positions=[pos1],cell=self.cell,pbc=self.pbc)
        T2 = Atoms(symbol,scaled_positions=[pos2],cell=self.cell,pbc=self.pbc)

        self.extend(T1)
        self.extend(T2)
        self.wrap()



##########################  MAIN PROGRAM  ########################

if __name__ == "__main__":

    # Testing paths
    path = "CONTCAR"
    out_path = "CONTCAR_out"

    structure = Structure(path)

    ## Adds Vacuum to optimized M2X or M2XT2 structure.
    # structure.add_vacuum(vacuum=15)
    # structure.write(out_path,"vasp",direct=True)

    ## Adds Termination to optimized structure.
    # structure.addT("O",hollows="HX")
    # structure.addT("H",hollows="HX")
    # structure.write(out_path,"vasp",direct=True)

    ## Shifts the slab a certain amount
    # structure.shift(3)
    # structure.write()

    ## Shifts to zero/origin all the atoms 
    # structure.to_zero()
    # structure.write()

    ## Convert to FHI-AIMS geometry.in
    # structure.write(out_path,"aims",scaled=True)


    ##Prints cell parameters for input structures

    # print(structure.get_repeated_T())
    print(f"{structure.symbols.formula}: {structure.get_geom(extra=True)}")

