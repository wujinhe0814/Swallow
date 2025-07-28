import os
from shutil import copyfile
import csv


def _create_model_training_folder(writer, files_to_same):
    """Create a directory to store model checkpoints and copy source files into it."""
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))


def pre_recCal(readFile, writeFile, n):
    """
    Compute precision, recall, weighted metrics and confusion matrix from prediction results.

    Args:
        readFile (str): Path to the prediction CSV file with [true_label, predicted_label].
        writeFile (str): Output CSV file to store metrics and confusion matrix.
        n (int): Number of unique classes.

    Returns:
        float: Overall accuracy.
    """
    import csv
    tp = {}  # True positives per class
    tp_fn = {}  # True positives + false negatives per class
    tp_fp = {}  # True positives + false positives per class
    Tp = 0  # Total true positives
    Fn = 0  # Total false negatives
    all_l = 0  # Total samples
    total = [[0] * n for i in range(n)]  # Confusion matrix

    # Open file for writing metrics
    csvwritefile = open(writeFile, "w", newline='')
    fieldnames = ['appName', 'precision', 'recall', 'Weight', 'number', 'tp']
    writer = csv.DictWriter(csvwritefile, delimiter=",", fieldnames=fieldnames)
    writer.writerow({'appName': 'appName', 'precision': 'precision', 'recall': 'recall',
                     'Weight': 'Weight', 'number': 'number', 'tp': 'tp'})

    # Read prediction results
    csvreadfile = open(readFile, "r")
    reader = csv.reader(csvreadfile, delimiter=",")
    for real, classres in reader:
        total[int(real)][int(classres)] += 1
        if not tp_fn.__contains__(real):
            tp_fn[real] = 0
            tp[real] = 0
        if not tp_fp.__contains__(real):
            tp_fp[real] = 0
        if not tp_fp.__contains__(classres):
            tp_fp[classres] = 0
        if real == classres:
            tp[real] += 1
            Tp += 1
        else:
            Fn += 1
        tp_fn[real] += 1
        tp_fp[classres] += 1
        all_l += 1
    csvreadfile.close()

    # Write precision/recall per class
    for key in tp_fn:
        try:
            preci = 1.0 * tp[key] / tp_fp[key]
        except:
            preci = 0
        recall = 1.0 * tp[key] / tp_fn[key]
        weight = 1.0 * tp_fn[key] / all_l
        writer.writerow({'appName': key, 'precision': '{:.4f}'.format(preci), 'recall': '{:.4f}'.format(recall),
                         'Weight': '{:.4f}'.format(weight), 'number': tp_fn[key], 'tp': tp[key]})

    print(Tp, all_l)

    # Write overall accuracy
    writer.writerow({'appName': 'accuracy', 'precision': '{:.4f}'.format(Tp * 1.0 / all_l),
                     'recall': '', 'Weight': ' ', 'number': ' ', 'tp': ' '})

    # Write confusion matrix
    row_writer = csv.writer(csvwritefile)
    row_writer.writerow([])
    row_writer.writerow([' '] + [i for i in range(n)])
    for i in range(len(total)):
        row_writer.writerow([i] + total[i])
    row_writer.writerow([])
    for i in range(n):
        info = []
        info.append(i)
        for j in range(n):
            if total[i][j] > 0:
                info.append(str(j) + '@' + str(total[i][j]))
        row_writer.writerow(info)

    csvwritefile.close()
    return Tp * 1.0 / all_l
