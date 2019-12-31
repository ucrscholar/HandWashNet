#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import argparse
import json
import os
import shutil
import sys
import glob
import bisect
import subprocess
import cv2
import numpy as np

coarse_dict = {
    "background": 0,
    "cut_and_mix_ingredients": 1,
    "prepare_dressing": 2,
    "serve_salad": 3
}

mid_dict = {
    "add_dressing": 0,
    "add_oil": 1,
    "add_pepper": 2,
    "add_salt": 3,
    "add_vinegar": 4,
    "background": 5,
    "cut_cheese": 6,
    "cut_cucumber": 7,
    "cut_lettuce": 8,
    "cut_tomato": 9,
    "mix_dressing": 10,
    "mix_ingredients": 11,
    "peel_cucumber": 12,
    "place_cheese_into_bowl": 13,
    "place_cucumber_into_bowl": 14,
    "place_lettuce_into_bowl": 15,
    "place_tomato_into_bowl": 16,
    "serve_salad_onto_plate": 17
}

fine_dict = {
    "add_dressing_core": 0,
    "add_dressing_post": 1,
    "add_dressing_prep": 2,
    "add_oil_core": 3,
    "add_oil_post": 4,
    "add_oil_prep": 5,
    "add_pepper_core": 6,
    "add_pepper_post": 7,
    "add_pepper_prep": 8,
    "add_salt_core": 9,
    "add_salt_post": 10,
    "add_salt_prep": 11,
    "add_vinegar_core": 12,
    "add_vinegar_post": 13,
    "add_vinegar_prep": 14,
    "background": 15,
    "cut_cheese_core": 16,
    "cut_cheese_post": 17,
    "cut_cheese_prep": 18,
    "cut_cucumber_core": 19,
    "cut_cucumber_post": 20,
    "cut_cucumber_prep": 21,
    "cut_lettuce_core": 22,
    "cut_lettuce_post": 23,
    "cut_lettuce_prep": 24,
    "cut_tomato_core": 25,
    "cut_tomato_post": 26,
    "cut_tomato_prep": 27,
    "mix_dressing_core": 28,
    "mix_dressing_post": 29,
    "mix_dressing_prep": 30,
    "mix_ingredients_core": 31,
    "mix_ingredients_post": 32,
    "mix_ingredients_prep": 33,
    "peel_cucumber_core": 34,
    "peel_cucumber_post": 35,
    "peel_cucumber_prep": 36,
    "place_cheese_into_bowl_core": 37,
    "place_cheese_into_bowl_post": 38,
    "place_cheese_into_bowl_prep": 39,
    "place_cucumber_into_bowl_core": 40,
    "place_cucumber_into_bowl_post": 41,
    "place_cucumber_into_bowl_prep": 42,
    "place_lettuce_into_bowl_core": 43,
    "place_lettuce_into_bowl_post": 44,
    "place_lettuce_into_bowl_prep": 45,
    "place_tomato_into_bowl_core": 46,
    "place_tomato_into_bowl_post": 47,
    "place_tomato_into_bowl_prep": 48,
    "serve_salad_onto_plate_core": 49,
    "serve_salad_onto_plate_post": 50,
    "serve_salad_onto_plate_prep": 51
}

custom_dict = {
    "add_dressing": 0,
    "add_oil": 1,
    "add_pepper": 2,
    "background": 3,
    "cut_into_pieces": 4,
    "mix_dressing": 5,
    "mix_ingredients": 6,
    "peel_cucumber": 7,
    "place_into_bowl": 8,
    "serve_salad_onto_plate": 9
}

fine_to_mid = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 3,
    12: 4,
    13: 4,
    14: 4,
    15: 5,
    16: 6,
    17: 6,
    18: 6,
    19: 7,
    20: 7,
    21: 7,
    22: 8,
    23: 8,
    24: 8,
    25: 9,
    26: 9,
    27: 9,
    28: 10,
    29: 10,
    30: 10,
    31: 11,
    32: 11,
    33: 11,
    34: 12,
    35: 12,
    36: 12,
    37: 13,
    38: 13,
    39: 13,
    40: 14,
    41: 14,
    42: 14,
    43: 15,
    44: 15,
    45: 15,
    46: 16,
    47: 16,
    48: 16,
    49: 17,
    50: 17,
    51: 17
}

mid_to_custom = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 4,
    8: 4,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 8,
    14: 8,
    15: 8,
    16: 8,
    17: 9
}


def create_dir(new_dir: os.path):
    if os.path.isdir(new_dir):
        print(new_dir + " already exists.")
        return False
    else:
        os.mkdir(new_dir)
        return True


def check_valid_dir(dir_path: os.path):
    if not os.path.isdir(dir_path):
        raise ValueError(dir_path + " is not a valid directory.")


def get_pattern_files(files_dir: os.path, pattern: str) -> list:
    check_valid_dir(files_dir)
    files = glob.glob1(files_dir, pattern)
    files.sort()
    files = [os.path.join(files_dir, f) for f in files]

    return files


def get_annotation_files(sync_files: list) -> list:
    annotation_files = [
        sync_file.replace('synchronization', 'activityAnnotation')
        for sync_file in sync_files
    ]
    annotation_files = [
        annotation_file.replace('activityAnnotation', 'activityAnnotations', 1)
        for annotation_file in annotation_files
    ]
    return annotation_files


def validate_dataset_files(dataset_files_dict: dict):
    #keys = ['rgb', 'depth', 'timestamp', 'annotation']
    keys = ['rgb', 'timestamp', 'annotation']
    # Check lengths - initial pass
    initial_entry_length = len(dataset_files_dict[keys[0]])
    for k in keys:
        entry_length = len(dataset_files_dict[k])
        if entry_length != initial_entry_length:
            raise ValueError("{} dataset dir should have {} entries, "
                             "but has {}.".format(k,
                                                  initial_entry_length,
                                                  entry_length))

    # Check label nums - initial pass
    # TODO: split by filename
    #label_num_begin = [-8, -8, -8, -27]
    label_num_begin = [-8, -8, -27]
    num_length = 4
    for entry_idx in range(initial_entry_length):
        # Check each adjacent pair
        for key_idx in range(len(keys) - 1):
            entry_a = dataset_files_dict[keys[key_idx]][entry_idx]
            entry_b = dataset_files_dict[keys[key_idx + 1]][entry_idx]

            entry_num_a = entry_a[label_num_begin[key_idx]:
                                  label_num_begin[key_idx] + num_length]
            entry_num_b = entry_b[label_num_begin[key_idx + 1]:
                                  label_num_begin[key_idx + 1] + num_length]

            if entry_num_a != entry_num_b:
                raise ValueError("{} and {} do not match".format(entry_a,
                                                                 entry_b))

    return True


def get_dataset_files(raw_dir: os.path) -> dict:
    rgb_files = get_pattern_files(os.path.join(raw_dir, 'rgb'), '*.avi')
    #depth_files = get_pattern_files(os.path.join(raw_dir, 'depth'), '*.zip')
    timestamp_files = get_pattern_files(os.path.join(raw_dir, 'timestamps'),
                                        '*.txt')
    sync_files = get_pattern_files(os.path.join(raw_dir, 'synchronization'),
                                   '*.txt')
    annotation_files = get_annotation_files(sync_files)

    dataset_files_dict = {
        'rgb': rgb_files,
    #    'depth': depth_files,
        'timestamp': timestamp_files,
        'annotation': annotation_files
    }

    validate_dataset_files(dataset_files_dict)

    return dataset_files_dict


def extract_rgb_files(rgb_avi_loc: os.path, output_dir_loc: os.path):
    # TODO: Implement in python to remove a dependency
    create_dir(output_dir_loc)

    vid = cv2.VideoCapture(rgb_avi_loc)
    ret, im = vid.read()
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(n_frames):
        print(i, "of", n_frames)
        if i > n_frames:
            print("New video")
            break

        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, im = vid.read()
        if not ret:
            print("No image")
            break

        im = cv2.resize(im, (100,100))
        #cv2.imshow("im", im)
        #ret = cv2.waitKey(1)
        #if ret >= 0:
        #    break

        cv2.imwrite("{}/{}.jpg".format(output_dir_loc, i), im)

    '''extract_pngs_cmd = ['ocv_extract_pngs',
                        str(rgb_avi_loc),
                        str(output_dir_loc)]
    subprocess.call(extract_pngs_cmd)'''


def unzip_depth(depth_zip_loc: os.path, output_dir_loc: os.path):
    os.mkdir(output_dir_loc)
    unzip_depth_cmd = ['unzip', str(depth_zip_loc), '-d', str(output_dir_loc)]
    subprocess.call(unzip_depth_cmd)

    # for certain zips they may be in a subfolder
    subfolders = [d.path for d in os.scandir(output_dir_loc) if d.is_dir()]
    if len(subfolders) > 0:
        subfiles = [f for f in os.scandir(output_dir_loc) if f.is_file()]
        # There should be one subfolder and no subfiles.
        if len(subfolders) == 1 and len(subfiles) == 0:
            pgm_folder = subfolders[0]
            pgm_files = glob.glob1(pgm_folder, '*.pgm')
            for pgm_file in pgm_files:
                shutil.move(os.path.join(pgm_folder, pgm_file), output_dir_loc)
            shutil.rmtree(pgm_folder)


def make_pcds(rgb_sample_dir: os.path,
              depth_sample_dir: os.path,
              timestamps: list,
              output_pcd_path: os.path):
    os.mkdir(output_pcd_path)

    rgb_filenames = [rgb_file for rgb_file in os.listdir(rgb_sample_dir)
                     if os.path.isfile(os.path.join(rgb_sample_dir, rgb_file))]
    rgb_filenames.sort()

    old_depth_filepath = os.path.join(depth_sample_dir, timestamps[0][1])

    for idx, ts in enumerate(timestamps):
        if idx >= len(rgb_filenames):
            print("Not enough rgb files in {} "
                  "for timestamps, wait for validation".format(rgb_sample_dir))
            return

        rgb_filepath = os.path.join(rgb_sample_dir, rgb_filenames[idx])
        depth_filepath = os.path.join(depth_sample_dir, ts[1])

        if not os.path.exists(depth_filepath):
            depth_filepath = old_depth_filepath

        old_depth_filepath = depth_filepath

        pcd_filename = ts[0].zfill(16) + '.pcd'
        pcd_filepath = os.path.join(output_pcd_path, pcd_filename)

        png_to_pcd_cmd = ['pcl_pgm2pcd',
                          depth_filepath,
                          rgb_filepath,
                          pcd_filepath,
                          '0.5']
        subprocess.call(png_to_pcd_cmd)


def calc_scene_flow(pcd_dir: os.path,
                    output_scene_flow_path: os.path):
    os.mkdir(output_scene_flow_path)
    scene_flow_cmd = ['pcl_pd_flow',
                      '-p',
                      str(pcd_dir),
                      '-d',
                      str(output_scene_flow_path)]
    subprocess.call(scene_flow_cmd)


def extract_depth_images(pcd_dir: os.path,
                         output_depth_path: os.path):
    os.mkdir(output_depth_path)

    pcd_filenames = [pcd_file for pcd_file in os.listdir(pcd_dir)
                     if os.path.isfile(os.path.join(pcd_dir, pcd_file))]
    for pcd_filename in pcd_filenames:
        depth_filename = pcd_filename.split('.')[0] + '.png'
        depth_file_path = os.path.join(output_depth_path, depth_filename)
        pcd_file_path = os.path.join(pcd_dir, pcd_filename)
        extract_depth_cmd = ['pcl_pcd2png',
                             '--no-nan',
                             '--field',
                             'z',
                             pcd_file_path,
                             depth_file_path]

        subprocess.call(extract_depth_cmd)


def extract_colour_images(pcd_dir: os.path,
                          output_rgb_path: os.path):
    create_dir(output_rgb_path)

    pcd_filenames = [pcd_file for pcd_file in os.listdir(pcd_dir)
                     if os.path.isfile(os.path.join(pcd_dir, pcd_file))]
    for pcd_filename in pcd_filenames:
        rgb_filename = pcd_filename.split('.')[0] + '.png'
        rgb_file_path = os.path.join(output_rgb_path, rgb_filename)
        pcd_file_path = os.path.join(pcd_dir, pcd_filename)
        extract_depth_cmd = ['pcl_pcd2png',
                             pcd_file_path,
                             rgb_file_path]

        subprocess.call(extract_depth_cmd)


def validate_sample_dir(output_sample_dir: os.path, timestamps):
    # Check that all the necessary dirs exist.
    if not os.path.isdir(output_sample_dir):
        raise ValueError("Invalid sample, "
                         "{} is not a directory".format(output_sample_dir))

    rgb_dir = os.path.join(output_sample_dir, 'rgb')
    depth_pgm_dir = os.path.join(output_sample_dir, 'depth_pgm')
    pcd_dir = os.path.join(output_sample_dir, 'pcd')
    depth_dir = os.path.join(output_sample_dir, 'depth')
    flow_dir = os.path.join(output_sample_dir, 'flow')

    sample_sub_dirs = [rgb_dir, depth_pgm_dir, pcd_dir, depth_dir, flow_dir]

    for ssd in sample_sub_dirs:
        if not os.path.isdir(ssd):
            raise ValueError("Invalid sample, "
                             "{} is not a directory".format(ssd))

    coarse_label_path = os.path.join(output_sample_dir, 'labels_coarse.npy')
    fine_label_path = os.path.join(output_sample_dir, 'labels_fine.npy')
    mid_label_path = os.path.join(output_sample_dir, 'labels_mid.npy')
    custom_label_path = os.path.join(output_sample_dir, 'labels_custom.npy')

    label_file_paths = [coarse_label_path,
                        fine_label_path,
                        mid_label_path,
                        custom_label_path]

    for label_file in label_file_paths:
        if not os.path.isfile(label_file):
            raise ValueError("Invalid sample, "
                             "{} is not present".format(label_file))

    # Check that the dirs have correct number of files.
    # rgb, depth, pcd, flow - 1, labels
    rgb_len = len(glob.glob1(rgb_dir, '*.png'))
    depth_len = len(glob.glob1(depth_dir, '*.png'))
    pcd_len = len(glob.glob1(pcd_dir, '*.pcd'))
    flow_len_plus_one = len(glob.glob1(flow_dir, "*.pcd")) + 1

    def invalid_num_msg(location, expected_num, found_num):
        return "Invalid sample, {} expected {} items " \
               "but found {}.".format(location, expected_num, found_num)

    num_samples = len(timestamps)
    num_samples_changed = False

    if rgb_len < num_samples:
        if rgb_len == depth_len and rgb_len == pcd_len:
            depth_pgms = glob.glob1(depth_pgm_dir, '*.pgm')
            depth_pgms.sort()
            last_pgm = depth_pgms[-1]
            pcds = glob.glob1(pcd_dir, '*.pcd')
            pcds.sort()
            last_pcd = pcds[-1]
            last_pcd_time = str(int(last_pcd.split('.')[0]))

            if last_pcd_time == timestamps[rgb_len - 1][0] and \
                    os.path.exists(os.path.join(depth_pgm_dir,
                                                timestamps[rgb_len - 1][1])):
                # last_pgm == timestamps[rgb_len - 1][1]:
                num_samples = rgb_len
                num_samples_changed = True

    if rgb_len < num_samples:
        raise ValueError(invalid_num_msg(rgb_dir, num_samples, rgb_len))

    if depth_len < num_samples:
        raise ValueError(invalid_num_msg(depth_dir, num_samples, depth_len))

    if pcd_len < num_samples:
        raise ValueError(invalid_num_msg(pcd_dir, num_samples, pcd_len))

    if flow_len_plus_one < num_samples:
        raise ValueError(invalid_num_msg(flow_dir,
                                         num_samples,
                                         flow_len_plus_one))

    # Check that the correct number of labels are in each label file.
    # Check that there are no entries with two labels
    for label_file in label_file_paths:
        labels = np.load(label_file)

        if num_samples_changed:
            labels = labels[:num_samples]
            np.save(label_file, labels)
        elif labels.shape[0] != num_samples:
            raise ValueError(
                invalid_num_msg(label_file, num_samples, labels.shape[0]))

        sample_sum_minus_one = np.sum(labels, axis=-1, dtype=np.int8) - 1

        non_zero_els = np.nonzero(sample_sum_minus_one)

        if len(np.transpose(non_zero_els)) > 0:
            raise ValueError("{} has invalid labels at the following "
                             "indices {}\n{}".format(label_file,
                                                     non_zero_els,
                                                     labels[non_zero_els]))


def read_timestamp_file(timestamp_file: os.path):
    timestamps = []
    prev_pgm = None
    with open(timestamp_file, 'r') as file_pointer:
        for line in file_pointer:
            timestamp_pair = line.split()
            # rgb_stamp = timestamp_pair[0].zfill(16) + '.png'
            if len(timestamp_pair) > 1:
                prev_pgm = timestamp_pair[1]
                timestamps.append((timestamp_pair[0],
                                   # rgb_stamp,
                                   timestamp_pair[1]))
            elif len(timestamp_pair) == 1:
                timestamps.append((timestamp_pair[0],
                                   # rgb_stamp,
                                   prev_pgm))

    return timestamps


def get_sample_labels(annotation_file: os.path,
                      timestamps: list,
                      output_path: os.path):
    # Read the annotations, converting to label numbers
    coarse_annotations = []
    fine_annotations = []
    with open(annotation_file, 'r') as file_pointer:
        for line in file_pointer:
            annotation_tuple = line.split()
            if annotation_tuple[2] in coarse_dict:
                coarse_annotations.append((int(annotation_tuple[0]),
                                           int(annotation_tuple[1]),
                                           coarse_dict[annotation_tuple[2]]))
            elif annotation_tuple[2] in fine_dict:
                fine_label_num = fine_dict[annotation_tuple[2]]
                mid_label_num = fine_to_mid[fine_label_num]
                custom_label_num = mid_to_custom[mid_label_num]
                fine_annotations.append((int(annotation_tuple[0]),
                                         int(annotation_tuple[1]),
                                         fine_label_num,
                                         mid_label_num,
                                         custom_label_num))
            else:
                raise ValueError(
                    "{} contains unrecognised annotation {}".format(
                        annotation_file, annotation_tuple))

    coarse_annotations.sort(key=lambda annotation: int(annotation[0]))
    fine_annotations.sort(key=lambda annotation: int(annotation[0]))

    # Create the output label arrays
    num_samples = len(timestamps)
    coarse_labels = np.zeros((num_samples, len(coarse_dict)), dtype=np.bool)
    fine_labels = np.zeros((num_samples, len(fine_dict)), dtype=np.bool)
    mid_labels = np.zeros((num_samples, len(mid_dict)), dtype=np.bool)
    custom_labels = np.zeros((num_samples, len(custom_dict)), dtype=np.bool)

    # Get the background label number
    background_coarse_label_num = coarse_dict['background']
    background_fine_label_num = fine_dict['background']
    background_mid_label_num = mid_dict['background']
    background_custom_label_num = custom_dict['background']

    # Assume background
    coarse_labels[:, background_coarse_label_num] = True
    fine_labels[:, background_fine_label_num] = True
    mid_labels[:, background_mid_label_num] = True
    custom_labels[:, background_custom_label_num] = True

    def index(sorted_array, search_value):
        idx = bisect.bisect_left(sorted_array, search_value)
        if idx != len(sorted_array) and sorted_array[idx] == search_value:
            return idx
        else:
            raise ValueError(
                "Value {} not found in array {}".format(search_value,
                                                        sorted_array))

    timestamp_ints = [int(ts[0]) for ts in timestamps]

    # Change the labels between the annotation bounds
    for annotation in coarse_annotations:
        start = index(timestamp_ints, annotation[0])
        end = index(timestamp_ints, annotation[1])
        label_num = annotation[2]

        coarse_labels[start:end, :] = False
        coarse_labels[start:end, label_num] = True

    for annotation in fine_annotations:
        start = index(timestamp_ints, annotation[0])
        end = index(timestamp_ints, annotation[1])
        fine_label_num = annotation[2]
        mid_label_num = annotation[3]
        custom_label_num = annotation[4]

        # We assume that there can only be one label
        fine_labels[start:end, :] = False
        fine_labels[start:end, fine_label_num] = True
        mid_labels[start:end, :] = False
        mid_labels[start:end, mid_label_num] = True
        custom_labels[start:end, :] = False
        custom_labels[start:end, custom_label_num] = True

    # Save the labels
    output_coarse_path = os.path.join(output_path, 'labels_coarse.npy')
    output_fine_path = os.path.join(output_path, 'labels_fine.npy')
    output_mid_path = os.path.join(output_path, 'labels_mid.npy')
    output_custom_path = os.path.join(output_path, 'labels_custom.npy')

    np.save(output_coarse_path, coarse_labels)
    np.save(output_fine_path, fine_labels)
    np.save(output_mid_path, mid_labels)
    np.save(output_custom_path, custom_labels)


def process_sample(rgb_file: os.path,
                   depth_file: os.path,
                   timestamp_file: os.path,
                   annotation_file: os.path,
                   cache_dir: os.path,
                   output_dir: os.path):
    sample_num_str = os.path.split(annotation_file)[1][0:4]
    print(sample_num_str)

    output_sample_dir = os.path.join(output_dir, sample_num_str)

    # Check that output directory does not already exist
    timestamps = read_timestamp_file(timestamp_file)

    if os.path.isdir(output_sample_dir):
        # If it does check that it does not already include all the data
        try:
            validate_sample_dir(output_sample_dir, timestamps)
            print("{} processing complete.".format(output_sample_dir))
            return
        except Exception as e:
            print(e)
            clean_sample_dir(output_dir, sample_num_str)
            print("Cleaning and retrying...")

    cache_sample_dir = os.path.join(cache_dir, sample_num_str)

    if os.path.isdir(cache_sample_dir):
        try:
            validate_sample_dir(cache_sample_dir, timestamps)
            shutil.move(cache_sample_dir, output_sample_dir)
            print("{} processing complete.".format(output_sample_dir))
            return
        except Exception as e:
            print(e)
            clean_sample_dir(cache_dir, sample_num_str)
            print("Cleaning and retrying...")
            os.mkdir(cache_sample_dir)
    else:
        os.mkdir(cache_sample_dir)

    # Extract rgb from file, renaming as you go
    cache_rgb_dir = os.path.join(cache_sample_dir, 'rgb')
    extract_rgb_files(rgb_file, cache_rgb_dir)

    # Extract depth from file
    #cache_depth_pgm_dir = os.path.join(cache_sample_dir, 'depth_pgm')
    #unzip_depth(depth_file, cache_depth_pgm_dir)

    # Calculate pcds
    #cache_pcd_dir = os.path.join(cache_sample_dir, 'pcd')
    #make_pcds(cache_rgb_dir, cache_depth_pgm_dir, timestamps, cache_pcd_dir)

    # Extract pcd depth
    #cache_depth_dir = os.path.join(cache_sample_dir, 'depth')
    #extract_depth_images(cache_pcd_dir, cache_depth_dir)

    # Extract pcd rgb
    # First clean the rgb dir
    #clean_cache_dir(cache_rgb_dir)
    #extract_colour_images(cache_pcd_dir, cache_rgb_dir)

    # Calculate flow
    #cache_flow_dir = os.path.join(cache_sample_dir, 'flow')
    #calc_scene_flow(cache_pcd_dir, cache_flow_dir)

    # Gather labels
    get_sample_labels(annotation_file, timestamps, cache_sample_dir)

    # Validate output dir
    #validate_sample_dir(cache_sample_dir, timestamps)

    # Copy to output_dir
    shutil.move(cache_sample_dir, output_sample_dir)


def clean_sample_dir(output_dir: os.path, sample_num_str):
    shutil.rmtree(os.path.join(output_dir, sample_num_str))


def clean_cache_dir(cache_dir: os.path):
    for entry in os.listdir(cache_dir):
        entry_path = os.path.join(cache_dir, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        elif os.path.isfile(entry_path):
            os.remove(entry_path)


def save_inverse_dict(original_dict, save_path):
    inverse_dict = {value: key for key, value in original_dict.items()}

    with open(save_path, 'w') as output_file:
        json.dump(inverse_dict, output_file, indent=4, sort_keys=True)


def generate_dataset(raw_dir, cache_dir, output_dir):
    # Validate the input directories, creating if necessary.

    # Check that the raw_dir exists
    if not os.path.isdir(raw_dir):
        raise ValueError("Specified raw_dir, {}, "
                         "is not a valid directory".format(raw_dir))

    # If cache_dir does not exist, attempt to create it.
    if not os.path.isdir(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as error:
            raise ValueError("Invalid cache_dir, {}, "
                             "unable to create directory".format(cache_dir))
    else:
        clean_cache_dir(cache_dir)

    # If output_dir does not exist, attempt to create it.
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError as error:
            raise ValueError("Invalid output_dir, {}, "
                             "unable to create directory".format(output_dir))

    dataset_files = get_dataset_files(raw_dir)

    rgb_files = dataset_files['rgb']
    depth_files = dataset_files['rgb'] #dataset_files['depth']
    timestamp_files = dataset_files['timestamp']
    annotation_files = dataset_files['annotation']

    for idx in range(len(rgb_files)):
    #for idx in range(1,2):
        process_sample(rgb_files[idx],
                       depth_files[idx],
                       timestamp_files[idx],
                       annotation_files[idx],
                       cache_dir,
                       output_dir)

    # Output the label dicts.
    coarse_dict_path = os.path.join(output_dir, 'coarse_labels.json')
    fine_dict_path = os.path.join(output_dir, 'fine_labels.json')
    mid_dict_path = os.path.join(output_dir, 'mid_labels.json')
    custom_dict_path = os.path.join(output_dir, 'custom_labels.json')

    save_inverse_dict(coarse_dict, coarse_dict_path)
    save_inverse_dict(fine_dict, fine_dict_path)
    save_inverse_dict(mid_dict, mid_dict_path)
    save_inverse_dict(custom_dict, custom_dict_path)


def run_script(args):
    raw_dir = os.path.abspath(args.raw_dir)
    cache_dir = os.path.abspath(args.cache_dir)
    output_dir = os.path.abspath(args.output_dir)

    generate_dataset(raw_dir, cache_dir, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process the 50 Salads dataset"
    )
    parser.add_argument('--raw_dir', default='/data1/shengjun/db/', help='Path to the raw dataset directory.')
    parser.add_argument('--output_dir', default='/data1/shengjun/db/output1',help='Path to store the dataset.')
    parser.add_argument('--cache_dir',default='/data1/shengjun/db/cache', help='Path to use as a cache.')

    sys.exit(run_script(parser.parse_args()))
