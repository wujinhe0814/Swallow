# Swallow-AE

This repository contains the artifacts for the paper:
**"Swallow: A Transfer-Robust Website Fingerprinting Attack via Consistent Feature Learning"**
to appear at CCS 2025.

You can read our paper at `/paper/Swallow.pdf`. The published version will be updated later.

---

## Project Structure
```
Swallow/
├── config.yaml                  # Configuration file for all parameters
├── ProcessData.py               # Data processing implementation
├── PreTrain.py                  # Pre-training implementation
├── fineTuning.py                # Fine-tuning implementation for closed-world
├── fineTuning_open.py           # Fine-tuning implementation for open-world
├── fineTuning-AE.py             # Fine-tuning implementation for AE scenario
├── trainer.py                   # Training utilities and functions
├── utils.py                     # Utility functions and helpers
├── models/                      # Model architectures and definitions
│   ├── resnet_base_network.py   # ResNet model implementation
│   └── mlp_head.py              # MLP model implementation
├── loader/                      # Data augmentation
│   └── dataAugmentation.py      # Data augmentation file
├── SaveModel/                   # Saved model checkpoints
├── SaveData/                    # Processed datasets
│   ├── PreTrain/                # Pre-training datasets
│   └── FineTune/                # Fine-tuning datasets
├── results/                     # Experiment results and outputs
│   ├── close/                   # Closed-world evaluation results
│   └── open/                    # Open-world evaluation results
└── Traces/                      # Raw traffic traces
    ├── D1-Undefence             # Raw dataset 1 (undefended)
    ├── D2-Undefence             # Raw dataset 2 (undefended)
    └── ...                      # Additional raw trace directories
```
## Supported Environments
The project is compatible with the following operating systems and environments:

### Operating System
- Windows or Ubuntu (Linux)
### Programming Language & Version
- Python 3.10 is required.
- **Must use the `python3` com**mand (do **not** use `python`, which may point to Python 2.x)
### Software & Libraries
- CUDA-compatible GPU with appropriate drivers installed
- PyTorch-compatible CUDA version
### Hardware Requirements
- At least one GPU for model training and evaluation
- 100GB of available disk space to store raw traces, processed datasets, and model checkpoints
### Recommended Configuration
- GPU memory: At least 8GB or more (e.g., NVIDIA RTX 3080 / Tesla V100)
- RAM: 16GB or higher
- SSD storage: For faster data loading and processing


## Install

### Note
- We provide the pre-trained models and corresponding datasets for the first two scenarios in the closed-world setting, as well as the pre-trained model and data for the first scenario in the open-world setting. 
- In addition, we also share some of the original traces from the scenario #1 in the closed-world setting. More details about this dataset can be found in Table 1 in the paper.
- You can download the above datasets and pre-trained models via https://zenodo.org/records/16607834.

### Installation steps:
1. Download or copy the project files to your local environment.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The `config.yaml` file contains all necessary configuration parameters. Key paths to configure:

```yaml
path:
  trace: "./Traces/"              # Path to raw traffic traces
  pretrain_data: "./SaveData/PreTrain/"  # Path to pre-training datasets
  finetune_data: "./SaveData/FineTune/"  # Path to fine-tuning datasets
  model_path: "./SaveModel/"      # Path to save models
```




## Direct Testing of the Results

You can directly run the trained model to obtain the corresponding results.

### Closed-World Evaluation (Section 6.2, Scenario #1 and #2, ~7 hours)
```bash
python3 fineTuning.py
```
Results are output as `.txt` files in the `results` directory, corresponding to Tables 2, 3, and 12 in the paper.

Details:
- Lines 184–192 of `fineTuning.py` produce `./results/close/results-Scenario#1.txt`, which corresponds to Table 2 (N=5) and Table 12 (N=10,15,20) in Section 6.2 Scenario #1.
- Lines 196–202 of `fineTuning.py` produce `./results/close/results-Scenario#2.txt`, which corresponds to Table 3 in Section 6.2 Scenario #2.

The `.txt` format includes:
- Format: `<PretrainDataset>-<FinetuneDataset>-N=<value>`
- Fields: `avg` (mean accuracy), `std` (standard deviation), `max` (maximum accuracy)

You can adjust the following macro parameters to reproduce specific rows in the tables:
- `N = []`: list of sample sizes to evaluate
- `defence_type_list = []`: list of defense methods to include

```bash
python3 ./results/close/txt2table.py
```
After completing the execution of `fineTuning.py`, run the above command to convert the original results.txt into a more readable tabular format.

`results/close/results-Scenario#1-table.csv` corresponds to Table 2 (N=5) and Table 12 (N=10,15,20) in Section 6.2 Scenario #1.

`results/close/results-Scenario#2-table.csv` corresponds to Table 3 in Section 6.2 Scenario #2.


### Open-World Evaluation (Section 6.3, ~30 minutes)
```bash
python3 fineTuning_open.py
```

Outputs include raw precision-recall values saved as `.csv` files and a generated figure `open_result.png` in `./results/open/`.  
If `.csv` results are already present, the script will directly generate the plot.  
These correspond to Section 6.3 and Figure 8 in the paper.

## Complete Training and Testing Pipeline

You also can reproduce the entire training and testing pipeline, including feature extraction from the raw Traces, pre-training, and fine-tuning.


### 1. Extract data for pre-training and fine-tuning ~15 minutes
```bash
python3 ProcessData.py
```

### 2. Conduct pre-training ~6 hours
```bash
python3 PreTrain.py
```

### 3. Conduct fine-tuning ~2 hours
```bash
python3 fineTuning-AE.py
```

> **Note**: This full pipeline demonstrates feature extraction → pre-training → fine-tuning using raw trace data.  
> Due to dataset size, we only include traces from Datasets D1–D5 under the Undefense, WTF-PAD, and Front scenarios. These correspond to partial results in Table 2 and Table 12 (Section 6.2 Scenario #1).  
> As in *Direct Testing of the Results*, you can modify `N = []` to reproduce specific rows in the tables.



## Citation
The published version and the standard citation format will be updated later.
## Contact
If you have any questions, please get in touch with us.
* Jinhe Wu ([jinhewu@bit.edu.cn](jikexin@bit.edu.cn))
* Junyu Ai (aijunyu@bit.edu.cn)

More detailed information about the research of Meng Shen Lab can be found here (https://mengshen-office.github.io/).
