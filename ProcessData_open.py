import os
import sys
import yaml
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from os.path import join, dirname, abspath

# Add the project root directory to Python's module search path
PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from CIF import fun  # Import the feature extraction function

# Load parameters and paths from configuration file
config_path = join(PROJECT_ROOT, "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Define dataset constants based on configuration
MAX_FINETUNE_NUM = 10
WEBSITE_NUM = config['dataset_info']['monitored_site_num']
TRAIN_NUM = config['dataset_info']['monitored_inst_num'] + MAX_FINETUNE_NUM
OPEN_WORLD_NUM = config['dataset_info']['unmonitored_test_num'] + MAX_FINETUNE_NUM * WEBSITE_NUM

# Extract paths for saving processed datasets
FINE_TUNE_DATA_PATH = config['path']['finetune_data']
PRETRAIN_DATA_PATH = config['path']['pretrain_data']

def process_single_file(file_name, folder_path):
    """
    Extract features from a single TCP trace file (monitored or unmonitored).
    """
    if '-' not in file_name:
        label = 0
        num = int(file_name)
    else:
        label, num = tuple(int(x) for x in file_name.split('-'))
        label = label + 1  # Label 0 is reserved for open-world/unmonitored class

    file_path = os.path.join(folder_path, file_name)

    # Parse and convert TCP dump content into feature vectors
    with open(file_path, 'r') as f:
        tcp_dump = f.readlines()
    seq = pd.Series(tcp_dump[:]).str.slice(0, -1).str.split("\t", expand=True).astype("float")
    times = np.array(seq.iloc[:, 0]) - np.array(seq.iloc[0, 0])
    length_seq = np.array(seq.iloc[:, 1]).astype("int")
    feature, load_time, slot_duration = fun(times, length_seq)

    return (feature, label, num, load_time, slot_duration)

def process_file_wrapper(args):
    """
    Helper to unpack arguments for multiprocessing compatibility.
    """
    file_name, folder_path = args
    return process_single_file(file_name, folder_path)

def parallel_process(file_list, folder_path, n_jobs=15):
    """
    Efficiently process files using a multiprocessing pool.
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
    Load raw files, extract features, and save as pretrain or finetune dataset.
    """
    print(f"\n{'=' * 50}")
    print(f"Processing dataset: {name}")
    print(f"Input path: {folder_path}")

    # Collect all monitored and unmonitored file names
    file_list = []
    for i in tqdm(range(WEBSITE_NUM), desc="Building file list"):
        for j in range(TRAIN_NUM):
            file_path = join(folder_path, f"{i}-{j}")
            if os.path.exists(file_path):
                file_list.append(f"{i}-{j}")

    if not is_pretrain:
        for i in tqdm(range(OPEN_WORLD_NUM)):
            file_path = join(folder_path, f"{i}")
            if os.path.exists(file_path):
                file_list.append(f"{i}")

    print(f"Found {len(file_list)} files to process")

    # Run feature extraction in parallel
    results = parallel_process(file_list, folder_path)

    # Organize extracted results into arrays
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

    # Format feature and label arrays
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
        # Save full dataset for pretraining
        save_path = join(PRETRAIN_DATA_PATH, f"{name}-PreTrain.npy")
        np.save(save_path, {
            'dataset': features,
            'label': labels,
            'load_time': load_times,
            'slot_duration': slot_durations
        })
    else:
        # Split and save fine-tuning dataset into train/test
        train_mask = ((labels == 0) & (nums < MAX_FINETUNE_NUM * WEBSITE_NUM)) | \
                     ((labels > 0) & (nums < MAX_FINETUNE_NUM))

        x_train = features[train_mask]
        y_train = labels[train_mask]
        x_test = features[~train_mask]
        y_test = labels[~train_mask]

        print(f"\nSplit statistics:")
        print(f"Training samples: {len(x_train)}")
        print(f"Testing samples: {len(x_test)}")

        save_path = join(FINE_TUNE_DATA_PATH, f'{name}-FineTune-open.npz')
        np.savez(save_path,
                 X_tune_train=x_train,
                 Y_tune_train=y_train,
                 X_tune_test=x_test,
                 Y_tune_test=y_test)

    print(f"\nOutput saved to: {save_path}")
    print(f"{'=' * 50}\n")

if __name__ == '__main__':
    # Run dataset processing for fine-tuning mode
    defence_list = ['DF-TrafficSliver']
    for defence_name in defence_list:
        traces_path = join(config["path"]["trace"], f'{defence_name}')
        process_dataset(traces_path, defence_name, is_pretrain=False)
