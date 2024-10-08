"""
Main program for the MXap package utilization through CLI.
Use MXene-trained ML models to predict bandgap.

Diego Ontiveros - 07/10/2024 [diego.ontiveros@ub.edu]
"""

import os

from mxgap.input import parse_user_input
from mxgap.ML import run_prediction
   

def cli():
    """Command Line Interface. Get user inputs from terminal (ArgParse) and feed them to the main ML prediction."""
    
    path, model, files = parse_user_input()

    run_prediction(path, model, files)


##########################################################################
############################## CLI PROGRAM ###############################
##########################################################################

if __name__=="__main__":
    cli()
    