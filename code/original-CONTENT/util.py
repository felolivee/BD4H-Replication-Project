import pickle
import numpy as np

def save_pkl(path, obj):
    with open(path, 'wb') as f:  # Changed 'w' to 'wb' for binary mode
        pickle.dump(obj, f)      # Changed cPickle to pickle
        print(f" [*] save {path}")

def load_pkl(path):
    with open(path, 'rb') as f:  # Changed to 'rb' for binary mode
        obj = pickle.load(f)     # Changed cPickle to pickle
        print(f" [*] load {path}")
        return obj

def save_npy(path, obj):
    np.save(path, obj)
    print(f" [*] save {path}")  # Using f-string instead of % formatting

def load_npy(path):
    obj = np.load(path, allow_pickle=True)  # Added allow_pickle=True for compatibility
    print(f" [*] load {path}")  # Using f-string
    return obj
