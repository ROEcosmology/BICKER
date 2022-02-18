'''
Flie contains Classes for the idividual component emulators.
'''
from tensorflow.keras.models import load_model
import numpy as np
from .training_funcs import UniformScaler, LogScaler
from scipy.interpolate import interp1d
import os
import pathlib

# Path to directory containing the NN weights as well as scalers needed produce
#  predictions with the NNs.
cache_path = os.fsdecode(pathlib.Path(os.path.dirname(__file__)
                                      ).parent.absolute())+"/bicker-cache/"

kbins = np.hstack([np.logspace(np.log(3.0e-4), np.log(.005), 4, base=np.e),
                   np.arange(.00501,.301,.0025),
                   np.logspace(np.log(.301), np.log(10.0), 25, base=np.e)])
kbins = kbins[5:47]

file_lists = [["full_c2_b2_f.npy", "full_c2_b1_b2.npy", "full_c2_b1_b1.npy",  
               "full_c2_b1_f.npy", "full_c2_b1_f.npy", "full_c1_b1_b1_f.npy",
               "full_c1_b2_f.npy", "full_c1_b1_b2.npy", "full_c1_b1_b1.npy",
               "full_c2_b1_b1_f.npy", "full_c1_b1_f.npy"], 
              ["full_c2_b1_f_f.npy", "full_c1_f_f.npy", "full_c1_f_f_f.npy",
               "full_c2_f_f.npy", "full_c2_f_f_f.npy", "full_c1_f_f.npy",
               "full_c1_b1_f_f.npy"], 
              ["full_c1_c1_f_f.npy", "full_c2_c2_b1_f.npy", "full_c2_c1_b1_f.npy",  
               "full_c2_c1_b1.npy", "full_c2_c1_b2.npy", "full_c2_c2_f_f.npy",
               "full_c1_c1_f.npy", "full_c2_c2_b1.npy", "full_c2_c2_b2.npy",
               "full_c2_c2_f.npy", "full_c2_c1_b1_f.npy", "full_c2_c1_f.npy",
               "full_c1_c1_b1_f.npy", "full_c1_c1_b1.npy", "full_c1_c1_b2.npy",
               "full_c1_c1_f_f.npy"], 
              ["full_c1_c1_bG2.npy", "full_c2_c2_bG2.npy", "full_c2_c1_bG2.npy"], 
              ["full_c1_b1_bG2.npy", "full_c1_bG2_f.npy", "full_c2_bG2_f.npy", 
               "full_c2_b1_bG2.npy"], 
              ["full_b1_f_f.npy", "full_b1_b1_f_f.npy", "full_b1_b1_b2.npy", 
               "full_b2_f_f.npy", "full_b1_b1_b1.npy", "full_b1_b1_b1_f.npy",
               "full_b1_b1_f.npy", "full_b1_f_f_f.npy", "full_f_f_f.npy",
               "full_f_f_f_f.npy", "full_b1_b2_f.npy"], 
              ["full_bG2_f_f.npy", "full_b1_b1_bG2.npy", "full_b1_bg2_f.npy"]]

def group_info(group, file_list=False):
    '''
    Args:
        group (int) : Group identifier.
        file_list (bool) : If ``True`` returns list of file containing
         the group kernels. Default is ``False``.

    Returns:
        Information about the kernel group.
    '''
    
    if file_list:
        return file_lists[group]
    else:
        return len(file_lists[group])


class component_emulator:
    '''
    Class componenet emulator.

    On initalisation the weights for the NN will be loaded,
    along with the scalers required to make predictions with the NN.

    Args:
        group (int) : Group identifier.
    '''

    def __init__(self, group):

        self.kbins = kbins
        '''The k-bins at which predictions will be made.'''

        components_path = cache_path+"components/"

        group_id = "group_{0}".format(group)
        self.nKer = group_info(group)

        # Load the NN.
        model = load_model(components_path+group_id+"/member_0", compile=False)

        self.model = model

        scalers_path = cache_path+"scalers/"

        xscaler = UniformScaler()
        yscaler = UniformScaler()

        # Load the variables that define the scalers.
        xmin_diff = np.load(scalers_path+"xscaler_min_diff.npy")
        ymin_diff = np.load(scalers_path+group_id+"/yscaler_min_diff.npy")

        xscaler.min_val = xmin_diff[0, :]
        xscaler.diff = xmin_diff[1, :]

        yscaler.min_val = ymin_diff[0, :]
        yscaler.diff = ymin_diff[1, :]

        self.scalers = (xscaler, yscaler)

    def emu_predict(self, X, split=True):
        '''
        Make predictions with the component emulator.

        Args:
            X (array) : Array containing the relevant input parameters. If making
             a single prediction should have shape (d,), if a batch prediction
             should have the shape (N,d).
            split (bool) : If ``True`` prediction is split into individual kernels.

        Returns:
            Array containing the predictions from the component emulator.
        '''

        # If making a prediction on single parameter set, input array needs to
        # be reshaped.
        X = np.atleast_2d(X)

        X_prime = self.scalers[0].transform(X)

        preds = self.scalers[1].inverse_transform(
            self.model(X_prime))

        if split:
            return np.split(preds, self.nKer, axis=1)
        else:
            return preds