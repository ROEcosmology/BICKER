import numpy as np
import bicker.training_funcs as Train
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import argparse
import time
from bicker import helper
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--inputX", help="Directory with feature files.", 
                    required=True)
parser.add_argument("--cosmo_params", help="cosmological parameters included in the emulation.", nargs='+', type=str,
                    required=True)
parser.add_argument("--inputY", help="Directory with target function files.", 
                    required=True)
parser.add_argument("--cache", help="Path to save outputs.", 
                    required=True)
parser.add_argument("--new_split", help='Use a new train test split? 0 for no, 1 for yes', 
                    default=0)
parser.add_argument("--arch", help="Architecture for the component emulators. pass as a string i.e. '200 200'. This specifies two hidden layers with 200 nodes each.", 
                    default="300 300")
parser.add_argument("--verbose", help='Verbose for tensorflow.', default=0)
args = parser.parse_args()

inputX_path = args.inputX
cosmo_params = args.cosmo_params
inputY_path = args.inputY
cache_path = args.cache+'/powerspec/'
new_split = bool(args.new_split)
arch = [int(i) for i in args.arch.split(" ")]

print("Loading features...")
cosmos = []
# Combine cosmologies from multiple files if needed.
for file in sorted(os.listdir(inputX_path)):
    print(file)
    cosmos.append(np.load(inputX_path+file))
cosmos = np.vstack(cosmos)
Nsamp = cosmos.shape[0]
print("Done.")

os.makedirs(cache_path, exist_ok=True)

print("Splitting into train and test sets...")
# Check directory.
if os.path.isfile(cache_path+"split/train.npy") and not new_split:
    print("Loaded old split...")
    train_id = np.load(cache_path+"split/train.npy")
    test_id = np.load(cache_path+"split/test.npy")
else:
    # Do knew split.
    print("Doing new split...")
    test_id, train_id = Train.train_test_indices(Nsamp, 0.2)
    # Check directory.
    if not os.path.isdir(cache_path+"split"):
            print("Creating directory: ", cache_path+"split")
            os.mkdir(cache_path+"split")
    # Save new split indicies.
    np.save(cache_path+"split/train.npy", train_id)
    np.save(cache_path+"split/test.npy", test_id)
    print("Done.")

print("Rescaling features...")
xscaler = Train.UniformScaler()
xscaler.fit(cosmos[train_id])
trainx = xscaler.transform(cosmos[train_id])
testx = xscaler.transform(cosmos[test_id])
# Check directory.
if not os.path.isdir(cache_path+"scalers"):
    print("Creating directory: ", cache_path+"scalers")
    os.mkdir(cache_path+"scalers")
# Save parameters of scaler.
np.save(cache_path+"scalers/xscaler_min_diff", np.vstack([xscaler.min_val,xscaler.diff]))
print("Done.")

# Load powerspec kernels.
print("Loading kernels...")
kernels = []
for i in range(10000):
    kernels.append(np.loadtxt(f"{inputY_path}/kernels/pk_kernels_{i}.txt"))
kernels = np.stack(kernels)
print("Done.")

# Check the shape of the kernel array.
if not kernels.shape == (Nsamp, 37, 119):
    raise(ValueError(f"The kernel array has the wrong shape: {kernels.shape}. Should be: {(Nsamp, 37, 119)}."))

# Define dict with column ids in the kernel array for each multipole.
pole_col_dict = {0: np.arange(16), 2: np.arange(17,28), 4: np.append(np.arange(28,34), 16),
                 'extra': np.arange(34,37)}

for k in pole_col_dict:

    # Select the relevant columns from kernel array.
    trainY = np.hstack([kernels[:,i,:] for i in pole_col_dict[k]])[train_id]

    print(f"Rescaling kernels P{k}...")
    yscaler = Train.UniformScaler()
    yscaler.fit(trainY)
    trainy = yscaler.transform(trainY)
    # Check directory.
    os.makedirs(os.path.join(cache_path, f"P{k}", "scalers"), exist_ok=True)
    # Save parameters of scaler.
    np.save(cache_path+f"P{k}/scalers/yscaler_min_diff",
        np.vstack([yscaler.min_val,yscaler.diff]))
    print("Done.")

    # Define training callbacks.
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,
                                min_lr=0.0001, mode='min', cooldown=5)
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=20,
                            verbose=0, mode='min', baseline=None,
                            restore_best_weights=True)
    callbacks_list = [reduce_lr, early_stop]

    # Do directory check BEFORE training.
    if not os.path.isdir(cache_path+f"P{k}/components"):
            print("Creating directory: ", cache_path+f"P{k}/components")
            os.mkdir(cache_path+f"P{k}/components")

    print("Training NN...")
    start = time.perf_counter()
    model = Train.trainNN(trainx, trainy, validation_data=None, nodes=np.array(arch),
                             learning_rate=0.001, batch_size=64, epochs=10000, callbacks=callbacks_list,
                             verbose=args.verbose)
    # Print training time.
    print("Done. ({0} s)".format(time.perf_counter()-start))

    # Compute the final loss.
    print(f"P{k} final loss: {model.evaluate(trainx, trainy)}")

    # Save weights.
    model.save(cache_path+f"P{k}/components/member_0")

np.save(cache_path+'cosmo_params.npy', cosmo_params)