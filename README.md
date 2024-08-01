# BICKER
BIspeCtrum KErnel emulator

## Installation

```
git clone https://github.com/ROEcosmology/BICKER.git
cd BICKER
pip install -e .
```

## Basic usage

See the [`bicker_test.ipynb`](https://github.com/ROEcosmology/BICKER/blob/main/notebooks/bicker_test.ipynb) notebook.
More documentation coming soon!

We provide the script [`train_components--groups.py`](https://github.com/ROEcosmology/BICKER/blob/main/scripts/train_components--groups.py)
that can be used to retrain the component emulators.

```bash session
$ python train_components--groups.py -h
usage: train_components--groups.py [-h] --inputX INPUTX --inputY INPUTY
                                   --cache CACHE [--new_split NEW_SPLIT]
                                   [--arch ARCH] [--verbose VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  --inputX INPUTX       Directory with feature files.
  --inputY INPUTY       Directory with target function files.
  --cache CACHE         Path to save outputs.
  --new_split NEW_SPLIT
                        Use a new train test split? 0 for no, 1 for yes
  --arch ARCH           Architecture for the component emulators. pass as a
                        string i.e. '200 200'. This specifies two hidden
                        layers with 200 nodes each.
  --verbose VERBOSE     Verbose for tensorflow.
```
