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
parser.add_argument("--inputY", help="Directory with target function files.", 
                    required=True)
parser.add_argument("--cache", help="Path to save outputs.", 
                    required=True)
parser.add_argument("--new_split", help='Use a new train test split? 0 for no, 1 for yes', 
                    default=0)
parser.add_argument("--arch", help="Architecture for the component emulators. pass as a string i.e. '200 200'. This specifies two hidden layers with 200 nodes each.", 
                    default="800 800")
parser.add_argument("--verbose", help='Verbose for tensorflow.', default=0)
args = parser.parse_args()

inputX_path = args.inputX
inputY_path = args.inputY
cache_path = args.cache
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
np.save(cache_path+"scalers/xscaler_min_diff",
        np.vstack([xscaler.min_val,xscaler.diff]))
print("Done.")

# Check the number of files in the components directory.
if not len(os.listdir(inputY_path)) == 52:
    raise(ValueError("The inputY directory only contains {0} files!. Should be 52.".format(len(os.listdir(inputY_path)))))

# Find start of all file names in components directroy.
# This is super hacky.
# TODO: Finda a better way of ignoring this start of the file name.
file_start = os.listdir(inputY_path)[0].split("_")[0]

for i in range(7):

    # Extract name for componenet from file name.
    component_name = "group_{0}".format(i)

    # Get list of components in group i.
    group_list = helper.group_info(i, file_list=True)

    # Load the data.
    print("Loading group: {0}".format(i))
    kernels = []
    for component in group_list:
            file = "{path}{start}_{component}.npy".format(path=inputY_path, 
                                                          start=file_start,
                                                          component=component)
            kernels.append(np.load(file))
            print("Loaded {0}".format(file))
    print("Done. Loaded {0} kernels.".format(len(group_list)))
    kernels = np.hstack(kernels)

    print("Rescaling kernels...")
    yscaler = Train.UniformScaler()
    yscaler.fit(kernels[train_id])
    trainy = yscaler.transform(kernels[train_id])
    # Check directory.
    if not os.path.isdir(cache_path+"scalers/"+component_name):
            print("Creating directory: ", cache_path+"scalers/"+component_name)
            os.mkdir(cache_path+"scalers/"+component_name)
    # Save parameters of scaler.
    np.save(cache_path+"scalers/"+component_name+"/yscaler_min_diff",
        np.vstack([yscaler.min_val,yscaler.diff]))
    print("Done.")

    # Define training callbacks.
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,
                                min_lr=0.0001, mode='min', cooldown=10, verbose=1)
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=20,
                            verbose=0, mode='min', baseline=None,
                            restore_best_weights=True)
    callbacks_list = [reduce_lr, early_stop]

    # Do directory check BEFORE training.
    if not os.path.isdir(cache_path+"components"):
            print("Creating directory: ", cache_path+"components")
            os.mkdir(cache_path+"components")
    if not os.path.isdir(cache_path+"components/"+component_name):
            print("Creating directory: ", cache_path+"components/"+component_name)
            os.mkdir(cache_path+"components/"+component_name)

    print("Training NN...")
    start = time.perf_counter()
    model = Train.trainNN(trainx, trainy, validation_data=None, nodes=np.array(arch),
                             learning_rate=0.001, batch_size=124, epochs=10000, callbacks=callbacks_list,
                             verbose=args.verbose)
    # Print training time.
    print("Done. ({0} s)".format(time.perf_counter()-start))

    # Compute the final loss.
    print("{0} final loss: {1}".format(component_name, model.evaluate(trainx, trainy)))

    # Save weights.
    model.save(cache_path+"components/"+component_name+"/member_0")

