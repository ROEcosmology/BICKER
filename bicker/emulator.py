from tensorflow.keras.models import load_model
import numpy as np
from .training_funcs import UniformScaler
from . import helper as helper_funcs
from scipy.interpolate import interp1d
import os
import pathlib

# Path to directory containing the NN weights as well as scalers needed produce
#  predictions with the NNs.
cache_path = os.fsdecode(pathlib.Path(os.path.dirname(__file__)
                                      ).parent.absolute())+"/bicker-cache/"

kbins = np.arange(0.005,0.2025,.0025)

class component_emulator:
    '''
    Class componenet emulator.

    On initalisation the weights for the NN will be loaded,
    along with the scalers required to make predictions with the NN.

    Args:
        group (int, str) : Group identifier. Can be ``'shot'`` or ``int`` 0-6.
        multipole (int) : Desired multipole. Can be either 0 or 2.
    '''

    def __init__(self, group, multipole):

        self.kbins = kbins
        '''The k-bins at which predictions will be made.'''

        if group is not 'shot':
            components_path = cache_path+"bispec/B{l}/".format(l=multipole)+"components/"
            scalers_path = cache_path+"bispec/B{l}/".format(l=multipole)+"scalers/"
            group_id = "group_{0}".format(group)
        elif group is 'shot':
            if multipole==2:
                raise NotImplementedError("Shot noise terms not yet implemented for B2.")
            else:
                components_path = cache_path+"B{l}/shot/".format(l=multipole)+"components/"
                scalers_path = cache_path+"B{l}/shot/".format(l=multipole)+"scalers/"
                group_id = "group_0"

        self.nKer = helper_funcs.group_info(group)

        # Load the NN.
        model = load_model(components_path+group_id+"/member_0", compile=False)

        self.model = model
        '''The NN that forms the component emulator.'''

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

class bispectrum:
    '''
    Class for emulating the galaxy bispectrum by combining emulated kernerl
    predictions with bias parameters.

    Args:
        multipole (int) : Desired multipole. Can be either 0 or 2.
        use_shot (bool) : Load shot noise kernels?
    '''

    def __init__(self, multipole, use_shot=False):

        # Initalise all component emulators.
        self.components = []
        '''List containg the component emulators.'''
        for g in range(7):
            self.components.append(component_emulator(g, multipole))

        self.use_shot = use_shot
        if use_shot:
            # Load shot noise component emulator.
            self.shot = component_emulator('shot', multipole)
            '''The shot noise component emulator.'''

    def emu_predict(self, cosmo, b1=None, b2=None, bG2=None, c1=None, c2=None):

        # Make predictions for all kernels.
        kernel_group_preds = []
        for g in range(7):
            kernel_group_preds.append(np.stack(self.components[g].emu_predict(cosmo)))

        bispec_pred = helper_funcs.combine_kernels(kernel_group_preds,
                                                   b1=b1, b2=b2, bG2=bG2,
                                                   c1=c1, c2=c2)

        # Make shot noise predictions.
        if self.use_shot:
            shot_preds = np.stack(self.shot.emu_predict(cosmo))
            shot_preds = helper_funcs.combine_kernels([shot_preds],
                                                      b1=b1, b2=b2, bG2=bG2,
                                                      c1=c1, c2=c2, 
                                                      groups=['shot'])

        if self.use_shot:
            return bispec_pred+shot_preds
        else:
            return bispec_pred

        