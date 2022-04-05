import numpy as np
import bicker.training_funcs as Train
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import argparse
import time

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
train_id = np.load(cache_path+"split/train.npy")
test_id = np.load(cache_path+"split/test.npy")
print("Done.")

print("Rescaling features...")
xscaler = Train.UniformScaler()
xscaler.fit(cosmos[train_id])
trainx = xscaler.transform(cosmos[train_id])
testx = xscaler.transform(cosmos[test_id])
# Check directory.
if not os.path.isdir(cache_path+"shot/scalers"):
        print("Creating directory: ", cache_path+"shot/scalers")
        os.mkdir(cache_path+"shot/scalers")
# Save parameters of scaler.
np.save(cache_path+"shot/scalers/xscaler_min_diff",
        np.vstack([xscaler.min_val,xscaler.diff]))
print("Done.")

# Check the number of files in the components directory.
if not len(os.listdir(inputY_path)) == 8:
    raise(ValueError("The inputY directory only contains {0} files!. Should be 8.".format(len(os.listdir(inputY_path)))))



# Extract name for componenet from file name.
component_name = "group_0"

# Get list of components in group i.
file_list = os.listdir(inputY_path)

# Load the data.
print("Loading data...")
kernels = []
for component in file_list:
        file = "{path}{component}".format(path=inputY_path,
                                                component=component)
        kernels.append(np.load(file))
        print("Loaded {0}".format(file))
print("Done")
kernels = np.hstack(kernels)

print("Rescaling kernels...")
yscaler = Train.UniformScaler()
yscaler.fit(kernels[train_id])
trainy = yscaler.transform(kernels[train_id])
# Check directory.
if not os.path.isdir(cache_path+"shot/scalers/"+component_name):
        print("Creating directory: ", cache_path+"shot/scalers/"+component_name)
        os.mkdir(cache_path+"shot/scalers/"+component_name)
# Save parameters of scaler.
np.save(cache_path+"shot/scalers/"+component_name+"/yscaler_min_diff",
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
if not os.path.isdir(cache_path+"shot/components"):
        print("Creating directory: ", cache_path+"shot/components")
        os.mkdir(cache_path+"shot/components")
if not os.path.isdir(cache_path+"shot/components/"+component_name):
        print("Creating directory: ", cache_path+"shot/components/"+component_name)
        os.mkdir(cache_path+"shot/components/"+component_name)

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
model.save(cache_path+"shot/components/"+component_name+"/member_0")

