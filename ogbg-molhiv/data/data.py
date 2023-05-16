"""
    File to load dataset based on user control from main file
"""
from data.molhiv import MolHIVDataset
from data.molpcba import MolPCBADataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """

    if DATASET_NAME == 'MOL-HIV':
        return MolHIVDataset(DATASET_NAME)
    
    if DATASET_NAME == 'MOL-PCBA':
        return MolPCBADataset(DATASET_NAME)
    
