# results

This directory stores experiment result files, including model evaluation outputs, figures, and logs.

- Result file types include .txt, .png, etc.
- It is recommended to organize subdirectories by scenario or date.

In the `close` folder, the `fineTuning.py` script generates `.txt` result files. These can be further organized into easily comparable tables by running [`txt2table.py`](close/txt2table.py), which outputs a `.csv` file.

In the `open` folder, the `fineTuning_open.py` script produces raw `.csv` data files along with `.png` result figures.
