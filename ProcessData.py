import os
import sys
import yaml
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from os.path import join, dirname, abspath

# Extend Python path to include project root directory
PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from CIF import fun  # Import custom feature extraction function

# Load configuration from YAML file
config_path = join("config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Retrieve dataset parameters from config
WEBSITE_NUM = config['dataset_info']['monitored_site_num']
TRAIN_NUM = config['dataset_info']['monitored_inst_num']
MAX_FINETUNE_NUM = 50

# Define data paths
FINE_TUNE_DATA_PATH = config['path']['finetune_data']
PRETRAIN_DATA_PATH = config['path']['pretrain_data']


def process_single_file(file_name, folder_path):
    """
    Extract features from a single raw trace file.
    """
    if '-' not in file_name:
        return None

    label, num = tuple(int(x) for x in file_name.split('-'))
    if label >= WEBSITE_NUM or num >= TRAIN_NUM:
        return None

    file_path = os.path.join(folder_path, file_name)

    # Load and parse raw TCP dump
    with open(file_path, 'r') as f:
        tcp_dump = f.readlines()
    seq = pd.Series(tcp_dump[:]).str.slice(0, -1).str.split("\t", expand=True).astype("float")
    times = np.array(seq.iloc[:, 0]) - np.array(seq.iloc[0, 0])
    length_seq = np.array(seq.iloc[:, 1]).astype("int")

    # Extract statistical features
    feature, load_time, slot_duration = fun(times, length_seq)

    return (feature, label, num, load_time, slot_duration)


def process_file_wrapper(args):
    """
    Wrapper for multiprocessing to unpack arguments.
    """
    file_name, folder_path = args
    return process_single_file(file_name, folder_path)


def parallel_process(file_list, folder_path, n_jobs=15):
    """
    Use multiprocessing to process files in parallel.
    """
    args_list = [(file_name, folder_path) for file_name in file_list]

    pool = Pool(n_jobs)
    results = list(tqdm(
        pool.imap(process_file_wrapper, args_list),
        total=len(file_list),
        desc="Processing files"
    ))
    pool.close()
    return results


def process_dataset(folder_path, name, is_pretrain=False):
    """
    Main function to process a dataset and save results to disk.
    """
    print(f"\n{'=' * 50}")
    print(f"Processing dataset: {name}")
    print(f"Input path: {folder_path}")

    # Collect list of valid files
    file_list = []
    for i in tqdm(range(WEBSITE_NUM), desc="Building file list"):
        for j in range(TRAIN_NUM):
            file_path = folder_path + "/" + f"{i}-{j}"
            if os.path.exists(file_path):
                file_list.append(f"{i}-{j}")

    print(f"Found {len(file_list)} files to process")

    # Parallel feature extraction
    results = parallel_process(file_list, folder_path)

    # Organize and filter extracted data
    features = []
    labels = []
    nums = []
    load_times = []
    slot_durations = []

    for result in results:
        if result is not None:
            feature, label, num, load_time, slot_duration = result
            features.append(feature)
            labels.append(label)
            nums.append(num)
            load_times.append(load_time)
            slot_durations.append(slot_duration)

    # Convert to numpy arrays for saving
    features = np.array(features)
    if len(features.shape) < 3:
        features = features[:, np.newaxis, :]
    labels = np.array(labels)
    nums = np.array(nums)
    load_times = np.array(load_times)
    slot_durations = np.array(slot_durations)

    print(f"\nDataset statistics:")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    if is_pretrain:
        # Save pretraining dataset
        save_path = join(PRETRAIN_DATA_PATH, f"{name}-PreTrain.npy")
        np.save(save_path, {
            'dataset': features,
            'label': labels,
            'load_time': load_times,
            'slot_duration': slot_durations
        })
    else:
        # Split and save fine-tuning dataset
        train_mask = nums < MAX_FINETUNE_NUM
        x_train = features[train_mask]
        y_train = labels[train_mask]
        x_test = features[~train_mask]
        y_test = labels[~train_mask]

        print(f"\nSplit statistics:")
        print(f"Training samples: {len(x_train)}")
        print(f"Testing samples: {len(x_test)}")

        save_path = join(FINE_TUNE_DATA_PATH, f'{name}-FineTune.npz')
        np.savez(save_path,
                 X_tune_train=x_train,
                 Y_tune_train=y_train,
                 X_tune_test=x_test,
                 Y_tune_test=y_test)

    print(f"\nOutput saved to: {save_path}")
    print(f"{'=' * 50}\n")


if __name__ == '__main__':

    # Process pretraining datasets
    defence_list = ["D1-Undefence", "D1-WTF-PAD", "D1-Front"]
    for defence_name in defence_list:
        traces_path = join(config["path"]["trace"], defence_name)
        process_dataset(traces_path, defence_name, is_pretrain=True)

    # Process fine-tuning datasets
    defence_list = ["D2-Undefence", "D2-WTF-PAD", "D2-Front",
                    "D3-Undefence", "D3-WTF-PAD", "D3-Front",
                    "D4-Undefence", "D4-WTF-PAD", "D4-Front",
                    "D5-Undefence", "D5-WTF-PAD", "D5-Front"]
    for defence_name in defence_list:
        traces_path = join(config["path"]["trace"], defence_name)
        process_dataset(traces_path, defence_name, is_pretrain=False)
