import matplotlib.pyplot as plt
import numpy as np
import os
from math import ceil
import sys


def read_overall_file():
    # read all experiments overall stats
    stats = open(OVERALL, newline="").readlines()
    # get level of computation on mesh (points/faces/components)
    while not "Point" in stats[0]:
        stats.pop(0)
    prefix = stats.pop(0).strip().split(",")
    prefix = [p for p in prefix if p != ""]
    # prefix.pop(0)

    # get stat names/description
    metrics = stats.pop(0).strip().split(",")[2:]
    metrics = list(dict.fromkeys(metrics))

    # get experiment name/description and stats
    exp_stats = {'nc': [], 'wc': []}
    exp_name = {'nc': [], 'wc': []}
    idx = ""
    for line in stats:
        line = line.split(",")
        if line[0] != "":
            idx = 'nc' if "no_colour" in line[0] else "wc"
            exp_name[idx].append(line[0])
        exp_stats[idx].append([float(j) for j in line[2:]])

    best_exp_stats = {'nc': [], 'wc': []}
    for key in exp_stats.keys():
        exp_stats[key] = np.array(exp_stats[key])
        # get highest scores per experiment
        for i in range(len(exp_name[key])):
            best_exp_stats[key].append(np.amax(exp_stats[key][i * 4:i * 4 + 4], axis=0))
        best_exp_stats[key] = np.array(best_exp_stats[key])

        # create experiment graph
        create_overall_graphs(metrics, exp_name[key], best_exp_stats[key], prefix, key)


def read_per_label_file():
    # read all experiments per label stats
    stats = open(PER_LABEL, newline="").readlines()
    # get level of computation for labels
    prefix = stats.pop(0).strip().split(",")
    prefix = [p for p in prefix if p != ""]
    prefix.pop(0)

    # get experiment name/description and stats
    exp_stats = {'nc': [], 'wc': []}
    exp_name = {'nc': [], 'wc': []}
    idx = ""
    for line in stats:
        line = line.split(",")
        if line[0] != "":
            idx = 'nc' if "no_colour" in line[0] else "wc"
            exp_name[idx].append(line[0])
        exp_stats[idx].append([float(j) for j in line[2:]])

    best_exp_stats = {'nc': [], 'wc': []}
    for key in exp_stats.keys():
        exp_stats[key] = np.array(exp_stats[key])

        for i in range(len(exp_name[key])):
            best_exp_stats[key].append(np.amax(exp_stats[key][i * 4:i * 4 + 4], axis=0))
        best_exp_stats[key] = np.array(best_exp_stats[key])
        # create experiments per experiment per label
        create_per_exp_per_label_graphs(exp_name[key], exp_stats[key], prefix, key)
        create_per_label_overall_graphs(exp_name[key], best_exp_stats[key], prefix, key)


def create_per_exp_per_label_graphs(exp_name, exp_stats, prefix, type=""):
    # decide colour and style of line graph
    def graph_style(idx):
        switcher = {
            0: "r-o",
            1: "g--^",
            2: "b:s"
        }
        return switcher.get(idx, "Error")

    N = np.arange(len(prefix))

    for i in range(len(exp_name)):
        fig, ax = plt.subplots(figsize=[15, 10])
        ax.grid(axis='both', which='both', color='black',
                linestyle='-.', linewidth=0.5)
        plt.title(exp_name[i] + "\nPer Label Metrics")
        ax.set_xticks(N)
        plt.xticks(rotation=90)
        ax.set_yticks(np.arange(0, 101, 5.0))
        ax.set_xticklabels(prefix)
        for j in range(3):
            plt.plot(N, exp_stats[i * 3 + j], graph_style(j))
        ax.legend(["points", "faces", "components"])
        plt.savefig(fname=os.path.join(OUT_DIR, '{}_per_label_{}.png'.format(exp_name[i], type)), bbox_inches='tight')
        plt.close()


def create_per_label_overall_graphs(exp_name, exp_stats, label_names, type=""):
    N = len(exp_name)
    M = len(label_names)
    bar_size = 0.5

    for k in range(M):
        fig, ax = plt.subplots(figsize=[15, 10])
        ax.grid(axis='y', which='major', color='black',
                linestyle='-', linewidth=0.5)
        ax.set_xticks(np.arange(N))
        plt.xticks(rotation=90)
        data = exp_stats[..., k]
        ax.set_yticks(np.arange(0, ceil(max(data)) + 6, 5))
        ax.set_xticklabels(exp_name)
        plt.title("Per Experiment Part IoU for Label: " + label_names[k])
        for i in range(N):
            ax.bar(i, exp_stats[i][k], width=bar_size, label=exp_name[i])
            ax.text(x=i - 0.2, y=exp_stats[i][k] + 0.05,
                    s=str(exp_stats[i][k]), color='black', size='medium')

        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(fname=os.path.join(OUT_DIR, 'over_all_results_{}_{}.png'.format(label_names[k], type)),
                    bbox_inches='tight')
        plt.close()


def create_overall_graphs(metrics, exp_name, best_exp_stats, prefix, type=""):
    N = len(exp_name)
    bar_size = 0.25
    bar_idx = np.arange(N)

    for k in range(len(prefix)):
        fig, ax = plt.subplots(figsize=[10, 10])
        ax.set_yticks(bar_idx + bar_size * (len(metrics) - 1) / 2)
        ax.set_yticklabels(exp_name)
        ax.grid(axis='both', which='both', color='black',
                linestyle='-.', linewidth=0.5)
        for i in range(len(metrics)):
            ax.barh(bar_idx, best_exp_stats[:, k * len(metrics) + i], height=bar_size - 0.1, label=metrics[i])
            for j, v in enumerate(best_exp_stats[:, k * len(metrics) + i]):
                ax.text(v + 1, bar_idx[j] - 0.05, str(v), color='black', size='small')
            bar_idx = bar_idx + bar_size

        ax.set_xticks(np.arange(0, 101, 10.0))
        plt.title("Per " + prefix[k] + " Overall Metrics")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(fname=os.path.join(OUT_DIR, 'over_all_results_per_{}_{}.png'.format(prefix[k], type)),
                    bbox_inches='tight')
        plt.close()


OVERALL = "/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/buildnet_exp/graphs/BuildNet_ICCV.csv"
assert (os.path.exists(OVERALL))

PER_LABEL = "/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/buildnet_exp/graphs/BuildNet_ICCV_Part_IoU.csv"
assert (os.path.exists(PER_LABEL))

OUT_DIR = "/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/buildnet_exp/graphs/"
if len(sys.argv) > 1:
    print(sys.argv)
    OUT_DIR = sys.argv[1]
os.makedirs(OUT_DIR, exist_ok=True)

read_overall_file()
read_per_label_file()
print("Done.")
