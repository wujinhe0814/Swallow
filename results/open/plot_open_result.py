import matplotlib.pyplot as plt
import csv
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
is_lengend = False
plt.rcParams["font.family"] = "Times New Roman"

def plot(precisons, recalls):
    colors = ['#2ca02c']
    linestyles = [ '-']
    markers = ['^']
    labels = ["Swallow"]

    plt.figure(figsize=(8, 6))
    plt.ylim((0.00, 1.05))
    plt.xlim((-0.05, 1.05))

    for i, (precison, recall) in enumerate(zip(precisons, recalls)):
        plt.plot(recall, precison, linestyle=linestyles[i], color=colors[i], lw=3.5, marker=markers[i],
                 label=labels[i], markersize=10)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.ylabel('Precision', fontsize=30)
    plt.xlabel('Recall', fontsize=30)

    plt.tight_layout()
    plt.grid(axis="y", linestyle='--')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()

    if is_lengend:
        plt.legend(loc='lower right', fontsize=20, ncol=1, labelspacing=0.1, columnspacing=0.3)

        handles, labels1 = plt.gca().get_legend_handles_labels()
        print(labels1)

        legend_fig = plt.figure(figsize=(2, 1))
        ax = legend_fig.add_subplot(111)
        ax.legend(handles, labels1, fontsize=30, handlelength=2.7, ncol=8, loc="center", frameon=False)
        ax.axis('off')

        legend_fig.savefig("lengend.pdf", dpi=1204, bbox_inches='tight')

    plt.show()

def read_csv(path):
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

if __name__ == '__main__':
    defence_type = 'DF-TrafficSliver'
    csv_paths = [
        f'./results-{defence_type}-open.csv',
    ]

    precisons, recalls = [], []
    for path in csv_paths:
        print(path)

        data = read_csv(path)
        precisons.append(data[0])
        recalls.append(data[1])

    plot(precisons, recalls)