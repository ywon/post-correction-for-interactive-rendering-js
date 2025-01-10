import os
import numpy as np
import exr
import glob
import multiprocessing as mp
import pathlib
from pathlib import Path
import tqdm

import tensorflow as tf
from tensorflow.keras.utils import Sequence


# An wrapper class for images
class Var:
    # Iterate item with its key
    def __iter__(self):
        for key, value in self.__dict__.items():
            if key == "index": continue
            yield key, value
    
    def to_gpu(self):
        for key, value in self.__dict__.items():
            if key == "index": continue
            self.__dict__[key] = tf.convert_to_tensor(value)


def load_exr(path):
    numpy_img = exr.read_all(path)["default"]
    # Add batch dim
    numpy_img = np.expand_dims(numpy_img, axis=0)
    return numpy_img

def save_as_npy(path):
    npy_path = path.replace('.exr', '.npy')
    if os.path.exists(npy_path):
        return
    numpy_img = load_exr(path)

    # Invalid value handling
    if np.isnan(numpy_img).any():
        print('There is NaN in', npy_path, 'Set it to zero for training.')
        numpy_img = np.nan_to_num(numpy_img, copy=False)
    if np.isposinf(numpy_img).any() or np.isneginf(numpy_img).any():
        print("There is INF in", npy_path, 'Set it to zero for training.')
        numpy_img[numpy_img == np.inf] = 0
        numpy_img[numpy_img == -np.inf] = 0

    np.save(npy_path, numpy_img)

# Make symolic link to files in orig_dir in new_dir using exr_dict
def make_symbolic(orig_dir, new_dir, exr_dict):
    print('Start to make symbolic links from', orig_dir, 'to', new_dir)

    new_dict = {}

    # Make symbolic links
    for key, files in exr_dict.items():
        for file in files:
            basename = os.path.basename(file)
            orig_path = os.path.join(orig_dir, file)
            new_file = os.path.join(new_dir, basename)
            # Make symbolic link
            if os.path.exists(new_file):
                os.remove(new_file)
            os.symlink(orig_path, new_file)

def extract_name_frame(filename):
    # Split the path into parts using the appropriate separator
    if os.path.sep == "\\":
        parts = filename.split("\\")
    else:
        parts = filename.split("/")
    # Extract the name from the last part
    name_and_number, ext = parts[-1].split(".", 1)
    name, frame = name_and_number.rsplit("_", 1)
    return name, int(frame)

def extract_filenames(filenames):
    # Extract the names from the file paths
    names = []
    # Dicts for each type
    files_dict = {}
    for filename in filenames:
        name, frame = extract_name_frame(filename)

        # Add the name to the list if it is not already there
        if name not in names:
            names.append(name)

        # Add the file to the dict
        if name not in files_dict:
            files_dict[name] = []
        files_dict[name].append(filename)
    return names, files_dict

# Function to check if all types of files have the same number of files and equal frame indices
def check_files(files_dict):
    num_files = None
    frames_dict = {}

    for key, files in files_dict.items():
        # Check if all files have the same number of files
        if num_files is None:
            num_files = len(files)
        elif len(files) != num_files:
            raise Exception(f"Error: {key} has a different number of files ({len(files)}) than other types of files ({num_files})")

        # Check if all files have equal frame indices
        frames = set()
        for file in files:
            filename = os.path.basename(file)
            name, frame = os.path.splitext(filename)[0].rsplit('_', 1)
            frames.add(int(frame))
        if len(frames) != num_files:
            raise(f"Error: {key} has different frame indices than other types of files")
        frames_dict[key] = frames

    # Check if frame_dict is empty
    if not frames_dict:
        raise Exception('frames_dict is empty')

    # Check if any type of frames_dict has different frame indices than other types
    for key, frames in frames_dict.items():
        for key2, frames2 in frames_dict.items():
            if key != key2 and frames != frames2:
                raise(f"Error: {key} has different frame indices than {key2}")

class DataLoader(Sequence):
    def __init__(self, directory, types, *args, **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)
        self.setup(directory, types)

    def setup(self, directory, types):
        # List files in format "{type}_{frame:04d}.exr"
        img_list = sorted(glob.glob(os.path.join(directory, "*.exr")))
        img_list = [os.path.basename(x) for x in img_list]
        img_list = [x for x in img_list if x.rsplit("_", 1)[0] in types]
        assert len(img_list) > 0, directory # Check emtpy

        # Parse file type
        unique_types, exr_dict = extract_filenames(img_list)
        check_files(exr_dict)

        # Make a new directory
        scene_dir = './scenes'
        parent_dir = os.path.dirname(directory)
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
            # Write parent directory of input to directory.txt
            with open(os.path.join(scene_dir, 'directory.txt'), 'w') as f:
                f.write(parent_dir)

        # Check if given directory is same as the one in directory.txt
        with open(os.path.join(scene_dir, 'directory.txt'), 'r') as f:
            orig_dir = f.read().replace('\n', '')
            if orig_dir != parent_dir:
                raise Exception(f"Error:\n\tloadded: {parent_dir}\n\tstored: {orig_dir}")

        new_dir = os.path.join('./scenes', pathlib.PurePath(directory).name)
        # Make a data directory
        if not Path(new_dir).exists():
            print(f'Making a directory: {new_dir}')
            os.makedirs(new_dir)
            
        # Make a symbolic link of files in orig_dir in new_dir if not exist
        make_symbolic(directory, new_dir, exr_dict)

        # Save exr images as npy
        fullpath_list = [os.path.join(new_dir, x) for x in img_list]
        print('Making npy files for faster loading... ')
        with mp.Pool(16) as p:
            list(tqdm.tqdm(p.imap(save_as_npy, fullpath_list), total=len(fullpath_list)))
        print('Done')
        
        # Change extension to npy
        img_list = [x.replace('.exr', '.npy') for x in img_list]

        # Get unique types
        unique_types = set([x.rsplit("_", 1)[0] for x in img_list])
        print(unique_types)

        # Check each type has the same number of files
        num_frames = len(img_list) // len(unique_types)
        for t in unique_types:
            num_type = len([x for x in img_list if x.rsplit("_", 1)[0] == t])
            assert num_type == num_frames

        # Check each type has the same start number of frame
        start_frame = min([int(x.rsplit("_", 1)[1].split(".")[0]) for x in img_list])
        for t in unique_types:
            frame = min(
                [
                    int(x.rsplit("_", 1)[1].split(".")[0])
                    for x in img_list
                    if x.rsplit("_", 1)[0] == t
                ]
            )
            assert frame == start_frame

        # Check each type has the same max number of frame
        max_frame = max([int(x.rsplit("_", 1)[1].split(".")[0]) for x in img_list])
        for t in unique_types:
            frame = max(
                [
                    int(x.rsplit("_", 1)[1].split(".")[0])
                    for x in img_list
                    if x.rsplit("_", 1)[0] == t
                ]
            )
            assert frame == max_frame

        # Set for later use
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.directory = new_dir
        self.unique_types = unique_types

        print(f"Directory '{directory}' has types:")
        for t in unique_types:
            print(f"\t{t}")

    def unpack(self, frame, imgs):
        # Unpack images into a dictionary
        unpacked = Var()
        unpacked.index = frame
        for i, type in enumerate(self.unique_types):
            unpacked.__dict__[type] = imgs[i]
        # Make useful constants
        if 'ref' in unpacked.__dict__:
            unpacked.ones1 = tf.ones_like(unpacked.ref[..., 0:1])
            unpacked.zeros1 = tf.zeros_like(unpacked.ref[..., 0:1])
            unpacked.zeros2 = tf.zeros_like(unpacked.ref[..., 0:2])
            unpacked.zeros3 = tf.zeros_like(unpacked.ref[..., 0:3])
        #else:
        #    print("'ref' attribute not found. Skipping the creation of certain constants.")
        #unpacked.ones1 = tf.ones_like(unpacked.ref[..., 0:1])
        #unpacked.zeros1 = tf.zeros_like(unpacked.ref[..., 0:1])
        #unpacked.zeros2 = tf.zeros_like(unpacked.ref[..., 0:2])
        #unpacked.zeros3 = tf.zeros_like(unpacked.ref[..., 0:3])
        return unpacked

    def __getitem__(self, frame):
        # Make filename using directory, unique_types and frame
        cur_frame = self.start_frame + frame
        filenames = [
            os.path.join(self.directory, f"{t}_{cur_frame:04d}.npy")
            for t in self.unique_types
        ]
        ext_imgs = list(map(np.load, filenames))
        return self.unpack(cur_frame, ext_imgs)

    def __len__(self):
        return self.num_frames
