import torch
import yaml
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.resnet_base_network import ResNet18
import warnings
import os
import logging
import csv
import matplotlib.pyplot as plt
import matplotlib
import math
warnings.filterwarnings("ignore")

# Suppress TensorFlow and plotting backend warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
is_lengend = False

def sample_traces_open(x, y, N):
    """Sample N instances per class, with special treatment for open-world (label 0)."""
    train_index = []
    labels = np.unique(y)

    for label in labels:
        idx = np.where(y == label)[0]
        if label == 0:
            sample_num = N * (len(labels) - 1)
            sample_num = min(sample_num, len(idx))
        else:
            sample_num = min(N, len(idx))
        idx = np.random.choice(idx, sample_num, False)
        train_index.extend(idx)

    train_index = np.array(train_index)
    np.random.shuffle(train_index)
    return x[train_index], y[train_index]

class Data(Dataset):
    """Torch dataset wrapper for feature-label data pairs."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class LogisticRegression(nn.Module):
    """Simple linear classifier for few-shot evaluation."""
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def finetune(model_name, defence_name, scenario_type):
    # Load preprocessed data and config
    fine_data_base_path = config["path"]["finetune_data"]
    data_path = fine_data_base_path + '%s-FineTune-open.npz' % (defence_name)
    data = np.load(data_path)
    x_train_total = data['X_tune_train']
    y_train_total = data['Y_tune_train']
    x_test_sup = data['X_tune_test']
    y_test_sup = data['Y_tune_test']

    # Normalize labels to consecutive integers
    unique_values_sorted = np.unique(y_train_total)
    mapping_dict = {value: idx for idx, value in enumerate(unique_values_sorted)}
    y_train_total = np.array([mapping_dict[value] for value in y_train_total]).astype(np.int64)
    y_test_sup = np.array([mapping_dict[value] for value in y_test_sup]).astype(np.int64)

    # Prepare test dataloader
    test_dataset = Data(x_test_sup, y_test_sup)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    # Iterate over different N-shot settings
    for N in N_shots:
        accuracies = []
        print("%s -> %s %s" % (model_name, defence_name, N))
        for run in range(num_runs):
            # Load pretrained encoder model
            encoder = ResNet18(**config['network']).to(device)
            model_path = config['path']['model_path']
            load_params = torch.load(
                f'{model_path}/{model_name}/model_epoch_200.pth',
                map_location=device)

            if 'online_network_state_dict' in load_params:
                encoder.load_state_dict(load_params['online_network_state_dict'])
                print("Parameters successfully loaded.")

            # Remove projection head for feature extraction
            output_feature_dim = encoder.projection.net[0].in_features
            encoder = nn.Sequential(*list(encoder.children())[:-1])

            # Sample N-shot training data
            x_train, y_train = sample_traces_open(x_train_total, y_train_total, N)
            print(f"x_train shape: {x_train.shape} ")
            print(f"y_train shape: {y_train.shape} ")

            train_dataset = Data(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # Initialize classifier and optimizer
            logreg = LogisticRegression(output_feature_dim, len(unique_values_sorted)).to(device)
            optimizer = torch.optim.Adam(list(encoder.parameters()) + list(logreg.parameters()), lr=1e-4)
            criterion = nn.CrossEntropyLoss()

            # Train classifier
            best_acc = 0.0
            encoder.train()
            logreg.train()

            for epoch in range(num_epochs):
                correct_train, total_train = 0, 0
                total_loss = 0

                for x, y in train_loader:
                    x = x.float().to(device)
                    y = y.to(device)
                    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

                    features = encoder(x).view(x.size(0), -1)
                    logits = logreg(features)
                    loss = criterion(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    predictions = torch.argmax(logits, dim=1)
                    correct_train += (predictions == y).sum().item()
                    total_train += y.size(0)
                    total_loss += loss.item()

                # Evaluate performance on test set periodically
                if epoch > 20 and epoch % eval_every_n_epochs == 0:
                    encoder.eval()
                    logreg.eval()
                    correct_test, total_test = 0, 0
                    website_res = []

                    with torch.no_grad():
                        for x, y in test_loader:
                            x = x.float().to(device)
                            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                            y = y.to(device)

                            features = encoder(x).view(x.size(0), -1)
                            logits = logreg(features)
                            predictions = torch.argmax(logits, dim=1)

                            correct_test += (predictions == y).sum().item()
                            total_test += y.size(0)

                            for i in range(len(predictions)):
                                website_res.append([y[i].item(), predictions[i].item()])

                    test_acc = 100 * correct_test / total_test
                    if test_acc > best_acc:
                        best_acc = test_acc

                        # Save evaluation result as CSV
                        save_path = './results/open'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        savename = f'results-{defence_name}-open.csv'
                        result_file = os.path.join(save_path, savename)

                        with open(result_file, 'w', encoding='utf-8', newline='') as file:
                            csvwriter = csv.writer(file)

                            upper_bound = 1.0
                            thresholds = upper_bound - upper_bound / np.logspace(0.05, 2, num=15, endpoint=True)
                            csvwriter.writerow(['TH', 'TP', 'TN', 'FP', 'FN', 'Pre.', 'Rec.'])
                            fmt_str = '{:.2f}:\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}'

                            for th in thresholds:
                                print(f'--------------------- threshold = {th:.3f}')
                                TP, FP, TN, FN = 0, 0, 0, 0
                                ow_label = 0

                                with torch.no_grad():
                                    for x, y in test_loader:
                                        x = x.float().to(device)
                                        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
                                        y = y.to(device)

                                        features = encoder(x).view(x.size(0), -1)
                                        logits = logreg(features)
                                        predictions = torch.softmax(logits, dim=1)
                                        output = predictions.cpu().detach().numpy()

                                        for pred, label in zip(output, y):
                                            best_n = np.argmax(pred)
                                            label = int(label)

                                            if label != ow_label:
                                                if best_n != ow_label and pred[best_n] >= th:
                                                    TP += 1
                                                else:
                                                    FN += 1
                                            else:
                                                if best_n != ow_label and pred[best_n] >= th:
                                                    FP += 1
                                                else:
                                                    TN += 1

                                eps = 1e-6
                                precision = float(TP) / (TP + FP + eps)
                                recall = float(TP) / (TP + FN + eps)
                                res = [th, TP, TN, FP, FN, precision, recall]
                                print(fmt_str.format(*res))
                                csvwriter.writerow(res)

                    print(f"Epoch [{epoch}/{num_epochs - 1}], Testing accuracy: {test_acc:.2f}%")

            accuracies.append(best_acc)

        # Log final statistics to summary file
        print(f"avg -> {np.mean(accuracies):.2f}, std -> {np.std(accuracies):.2f}")
        with open('results/open/results-%s.txt' % (scenario_type), 'a') as file:
            file.write(f'{model_name}-{defence_name} - {N} \n')
            file.write(f" avg -> {np.mean(accuracies):.2f}, std -> {np.std(accuracies):.2f}, "
                       f"max -> {np.max(accuracies):.2f}\n")

def read_csv(path):
    """Parse precision and recall data from evaluation CSV."""
    with open(path, 'r', encoding='UTF-8') as f:
        rows = csv.reader(f)
        precison, recall = [], []
        for i, row in enumerate(rows):
            if i == 0:
                continue
            if float(row[-2]) == 0:
                continue
            if float(row[-1]) == 0:
                continue
            precison.append(float(row[-2]))
            recall.append(float(row[-1]))

    return precison, recall

def plot_result_grid(plot_list):
    """Plot precision-recall curves for each evaluated defense method."""
    n = len(plot_list)
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    for i, path in enumerate(plot_list):
        precision, recall = read_csv(path)
        ax = axes[i]

        ax.plot(recall, precision, linestyle='-', color='#2ca02c', lw=3.5, marker='^', markersize=8, label='Swallow')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.00, 1.05)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        name = path.split('/')[-1][8:-9]
        ax.set_title(f'PR Curve: {name}', fontsize=14)
        ax.grid(axis="y", linestyle='--')
        ax.tick_params(labelsize=10)
        ax.legend(loc='lower right', fontsize=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("./results/open/open_result.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # Initialize configuration and training parameters
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    N_shots = [10]
    num_runs = 1
    num_epochs = 41
    eval_every_n_epochs = 40
    batch_size = 16

    # Open-world evaluation pipeline for multiple defense methods
    defence_type_list = ['Undefence']#['Undefence','WTF-PAD','Front','RegulaTor','Palette','Surakav' ,'TrafficSliver']
    plot_list = []
    for defence in defence_type_list:
        model_name = f"Wang100-{defence}-Ori"
        data_list = ["DF"]
        for data in data_list:
            defence_name = f"{data}-{defence}"
            if not os.path.exists(f'./results/open/results-{defence_name}-open.csv'):
                finetune(model_name, defence_name, "open")
            plot_list.append(f'./results/open/results-{defence_name}-open.csv')

    plot_result_grid(plot_list)
