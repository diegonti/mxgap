"""
Script to analyze Density of States (DOS). 
For now, only VASP DOSCAR files are implemented.

Diego Ontiveros
"""

import numpy as np
from ase.calculators.vasp import VaspDos


def get_bandgap(E,DOS):
    """Gets the bandgap from the given E and DOS arrays. 
    Assumes E is corrected with Ef."""

    # Gets the sites of bandgap states at E>0 (E>Ef)
    sites = (DOS==0)[E>0]

    # Index of the Fermi level                     
    Ef_index = np.argmin(np.abs(E))

    # Gets the CBM index and bandgap from the first non-zero state
    if not sites[0]: cbm_idx = 0
    else: cbm_idx = np.where(sites==False)[0][0]+1
    Eg = E[Ef_index + cbm_idx]

    return round(Eg,3)


def make_histogram(E,DOS,n_bins=100,E_min=-5,E_max=5):
    """Makes a histogram from the given E and DOS arrays.

    `n_bins` : Number of histogram bins. By default 100.
    `E_min`  : Minimum energy to start histogram. By default -5.
    `E_max`  : Maximum energy to end histogram. By default +5.
    """
    E, DOS = np.array(E), np.array(DOS)

    # Selects the sites around the Fermi level
    sites = np.logical_and( E>=E_min , E<=E_max)
    E_slice = E[sites]
    dos_slice = DOS[sites]

    bin_width = (E_max-E_min)/n_bins
    bin_width_idx = len(E_slice)/n_bins

    # Computes the average for each bin
    dos_hist = np.zeros(n_bins)
    E_hist = np.zeros(n_bins)
    for bin in range(n_bins):
        start_idx = round(bin*bin_width_idx)
        finish_idx = round((bin+1)*bin_width_idx)
        DOS_bin = dos_slice[start_idx:finish_idx]
        E_bin = E_slice[start_idx:finish_idx]
        dos_hist[bin] = DOS_bin.mean()
        E_hist[bin] = E_bin.mean()

    return dos_hist,E_hist


class Doscar():
    def __init__(self,path:str):
        """Main class to anaylze the DOS of a given file (path). 
        Uses ASE VaspDos to read the DOSCAR."""

        self.path = path
        self.dos = VaspDos(path)

        self.E,self.tdos,self.ef,self.ispin = self.get_dos()


    def get_dos(self):
        """Gets the DOS and DOS information from DOSCAR file.

        Returns
        -------
        `E`     : Energy array of the DOS. Corrected with Ef.
        `dos`   : Total DOS array. up+down if spin polarized.
        `Ef`    : Fermi Level, in eV.
        `ispin` : Number of spin channels (1 or 2).

        """
        file = self.path

        # Open file to get Ef and spin
        with open(file,"r") as inFile:
            data = [inFile.readline() for _ in range(10)]
        Ef = round(float(data[5].strip().split()[3]),3)

        if len(data[8].strip().split()) == 5: ispin = 2
        else: ispin = 1

        # Gets total DOS
        if ispin == 2:
            up,down = self.dos._total_dos[1:3]
            dos = up+down
        else: dos = self.dos._total_dos[1]

        # Gets the exact Fermi Level
        energy = self.dos.energy
        index_closest = np.argmin(np.abs(energy - Ef))
        Ef_index = np.where(dos[index_closest-2:index_closest+2]==0)[0]

        if len(Ef_index) == 0: Ef = energy[index_closest]
        else: Ef = energy[index_closest-2:index_closest+2][Ef_index[0]-1]
    
        # Getting E (corrected with Ef)
        E = self.dos.energy -  Ef

        return E, dos, Ef, ispin


    def get_bandgap(self):
        """Gets the bandgap from the DOS. Assumes E is corrected with Ef."""

        return get_bandgap(self.E,self.tdos)


    def make_histogram(self,n_bins=100,E_min=-5,E_max=5):
        """Makes a histogram from the given DOS."""

        return make_histogram(self.E,self.tdos,n_bins,E_min,E_max)



if __name__=="__main__":
    doscar = Doscar("test/DOSCAR1")
    Eg = doscar.get_bandgap()
    print(doscar.make_histogram())


