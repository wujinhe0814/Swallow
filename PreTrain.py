import os
import time
import torch
import yaml
import numpy as np
import warnings
import logging
from torch.utils.data import Dataset, ConcatDataset, DataLoader

# Import custom modules and models
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer
from loader.dataAugmentation import Augmentor

torch.manual_seed(0)
warnings.filterwarnings("ignore")

# Suppress TensorFlow and logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class TrainData(Dataset):
    """Custom PyTorch Dataset class for training with BYOL-style augmentations."""

    def __init__(self, x_train, y_train, x_load_time, x_slow_duration, augmentor, n_views):
        self.x = x_train
        self.y = y_train
        self.x_load_time = x_load_time
        self.x_slow_duration = x_slow_duration
        self.augmentor = augmentor
        self.n_views = n_views

    def __getitem__(self, index):
        # Generate multiple augmented views of the same sample
        res = [self.augmentor.augment(self.x[index], self.x_load_time[index], self.x_slow_duration[index])
               for _ in range(self.n_views)]
        return res, self.y[index]

    def __len__(self):
        return len(self.x)

def load_config(config_path="config.yaml"):
    """Load configuration dictionary from YAML file."""
    print("Loading config...")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def prepare_data(model_name, config):
    """Load dataset, apply augmentation, and create data loader."""
    print(f"\nProcessing model: {model_name}")
    data_path = config["path"]["pretrain_data"] + "%s-PreTrain.npy"%(model_name)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    print(f"Device: {device}")

    print("Loading data...")
    data = np.load(data_path, allow_pickle=True).item()
    x_train = data['dataset']
    y_train = data['label']
    x_load_time = data['load_time']
    x_slot_duration = data['slot_duration']

    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")
    print(f'Train data shapes: {x_train.shape}, {y_train.shape}')

    # Create multiple augmented versions of the dataset
    print("Creating augmented datasets...")
    augmentor = Augmentor()
    multi_times = config['trainer']['multi_times']
    n_views = config['trainer']['n_views']

    datasets = []
    for i in range(multi_times):
        train_dataset_i = TrainData(x_train, y_train, x_load_time, x_slot_duration, augmentor, n_views)
        datasets.append(train_dataset_i)
    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True)
    return train_loader, device

def build_model(config, device):
    """Initialize online, target, and predictor networks with optional pretraining."""
    print("Initializing networks...")
    online_network = ResNet18(**config['network']).to(device)
    target_network = ResNet18(**config['network']).to(device)

    # Load pretrained weights if specified
    pretrained_folder = config['network'].get('fine_tune_from', None)
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')
            load_params = torch.load(os.path.join(checkpoints_folder, 'model.pth'),
                                     map_location=device)
            online_network.load_state_dict(load_params['online_network_state_dict'])
            print("Pre-trained weights loaded successfully.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    optimizer = torch.optim.SGD(
        list(online_network.parameters()) + list(predictor.parameters()),
        **config['optimizer']['params']
    )

    return online_network, target_network, predictor, optimizer

def train_model(model_name, config, train_loader, device, online_network, target_network, predictor, optimizer):
    """Train the model using the BYOL self-supervised learning framework."""
    print("\nStarting training...")

    trainer = BYOLTrainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        predictor=predictor,
        device=device,
        model_path=model_name,
        **config['trainer']
    )
    trainer.train(train_loader)

def main():
    """Main training loop for multiple model variants."""
    model_name_list = ["D1-Undefence","D1-WTF-PAD","D1-Front"]
    for model_name in model_name_list:
        config = load_config()
        train_loader, device = prepare_data(model_name, config)
        online_net, target_net, predictor, optimizer = build_model(config, device)
        train_model(model_name, config, train_loader, device, online_net, target_net, predictor, optimizer)

if __name__ == '__main__':
    # Suppress specific user warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="loaded more than 1 DLL from .libs")

    main()
