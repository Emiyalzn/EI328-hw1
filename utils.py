import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from model import MLP, MLQP
from copy import deepcopy
from multiprocessing import Pool

dataset_dir = "dataset"
result_dir = "results"
train_data_name = "two_spiral_train_data.txt"
test_data_name = "two_spiral_test_data.txt"

plt.rcParams['axes.facecolor']='gray'

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def load_data(mode):
    if mode == 'train':
        file = open(os.path.join(dataset_dir, train_data_name), 'r')
        xs, ys, labels = [], [], []
        for line in file:
            x, y, label = line.split()
            xs.append(float(x))
            ys.append(float(y))
            labels.append(int(label))
    else:
        file = open(os.path.join(dataset_dir, test_data_name), 'r')
        xs, ys, labels = [], [], []
        for line in file:
            x, y, label = line.split()
            xs.append(float(x))
            ys.append(float(y))
            labels.append(int(label))
    file.close()
    return np.array(xs), np.array(ys), np.array(labels)

def partition_data(data, mode, num_partitions):
    xs, ys, labels = data
    white_indices = labels == 1
    black_indices = labels == 0
    xw, yw, labelw = xs[white_indices], ys[white_indices], labels[white_indices]
    xb, yb, labelb = xs[black_indices], ys[black_indices], labels[black_indices]
    white_subsets, black_subsets = [], []
    if mode == 'random':
        white_part_indices = np.random.choice(num_partitions, len(xw))
        black_part_indices = np.random.choice(num_partitions, len(xb))
        for i in range(num_partitions):
            indices = white_part_indices == i
            white_subsets.append((xw[indices], yw[indices], labelw[indices]))
            indices = black_part_indices == i
            black_subsets.append((xb[indices], yb[indices], labelb[indices]))
    elif mode == 'yaxis':
        x_indice = list(range(-6, 6, int(12/num_partitions))); x_indice.append(6)
        for i in range(num_partitions):
            indices = np.logical_and(xw >= x_indice[i], xw <= x_indice[i+1])
            white_subsets.append((xw[indices], yw[indices], labelw[indices]))
            indices = np.logical_and(xb >= x_indice[i], xb <= x_indice[i+1])
            black_subsets.append((xb[indices], yb[indices], labelb[indices]))
    elif mode == 'yaxis+overlap':
        x_indice = list(range(-6, 6, int(12/num_partitions))); x_indice.append(6)
        for i in range(num_partitions):
            indices = np.logical_and(xw >= x_indice[i]-0.1, xw <= x_indice[i+1]+0.1)
            white_subsets.append((xw[indices], yw[indices], labelw[indices]))
            indices = np.logical_and(xb >= x_indice[i]-0.1, xb <= x_indice[i+1]+0.1)
            black_subsets.append((xb[indices], yb[indices], labelb[indices]))
    else:
        raise ValueError("Unknown partition mode!")

    return white_subsets, black_subsets

def plot_datapoints(white_subsets, black_subsets):
    num_partition = len(white_subsets)
    fig, ax = plt.subplots(num_partition, num_partition, figsize=(9, 9))
    for i in range(num_partition):
        for j in range(num_partition):
            ax[i][j].set_xlim([-7,7])
            ax[i][j].set_ylim([-7,7])
            # we transpose the matrix here to keep corresponding with the pcolormesh function
            ax[i][j].scatter(white_subsets[i][1], white_subsets[i][0], c='w')
            ax[i][j].scatter(black_subsets[j][1], black_subsets[j][0], c='black')
    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, "partition_data_distribution.png"), bbox_inches='tight')

def plot_minmax_boundaries(models, model_name, mode):
    partition_num = len(models)
    fig, ax = plt.subplots(partition_num, partition_num, figsize=(9, 9))
    fig_min, ax_min = plt.subplots(1, partition_num, figsize=(9, 3))
    fig_max, ax_max = plt.subplots(figsize=(9, 9))
    xrange = np.arange(start=-7., stop=7., step=.05)
    yrange = np.arange(start=-7., stop=7., step=.05)
    preds = []
    min_preds = []
    coor_predictions = np.zeros((len(xrange), len(yrange)))
    for i in range(partition_num):
        preds.append([])
        for j in range(partition_num):
            for m, x in enumerate(xrange):
                for n, y in enumerate(yrange):
                    coor_predictions[m,n] = 1 if models[i][j].forward(np.array([x,y])) > 0.5 else 0
            preds[i].append(deepcopy(coor_predictions))
            ax[i,j].pcolormesh(xrange, yrange, coor_predictions, cmap="gray", antialiased=True)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, f"{model_name}_minmaxnet.png"))
    for i in range(partition_num):
        min_pred = np.min(preds[i],axis=0)
        min_preds.append(deepcopy(min_pred))
        ax_min[i].pcolormesh(xrange, yrange, min_pred, cmap="gray", antialiased=True)
    fig_min.tight_layout()
    fig_min.savefig(os.path.join(result_dir, f"{model_name}_mingates.png"))
    max_pred = np.max(min_preds,axis=0)
    ax_max.pcolormesh(xrange, yrange, max_pred, cmap="gray", antialiased=True)
    fig_max.tight_layout()
    fig_max.savefig(os.path.join(result_dir, "partition", f"{model_name}_{mode}_{partition_num}_maxgate.png"))

def plot_boundaries(model, model_name, lr1, lr2, alpha):
    fig, ax = plt.subplots()
    xrange = np.arange(start=-7., stop=7., step=.05)
    yrange = np.arange(start=-7., stop=7., step=.05)
    coor_predictions = np.zeros((len(xrange),len(yrange)))
    for i, x in enumerate(xrange):
        for j, y in  enumerate(yrange):
            coor_predictions[i,j] = 1 if model.forward(np.array([x,y])) > 0.5 else 0
    ax.pcolormesh(xrange, yrange, coor_predictions, cmap="gray", antialiased=True)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "lr+alpha", f"{model_name}_{lr1}_{lr2}_{alpha}_prediction.png"))

def plot_train_curves():
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(1,2, figsize=(8,3))
    # load files
    file = open(os.path.join(result_dir, "mlp_data.pkl"), 'rb')
    mlp_times = pickle.load(file)
    mlp_losses = pickle.load(file)
    mlp_train_accs = pickle.load(file)
    mlp_test_accs = pickle.load(file)
    file.close()

    file = open(os.path.join(result_dir, "mlqp_data.pkl"), 'rb')
    mlqp_times = pickle.load(file)
    mlqp_losses = pickle.load(file)
    mlqp_train_accs = pickle.load(file)
    mlqp_test_accs = pickle.load(file)
    file.close()

    # plot loss curve
    ax[0].plot(mlp_times, mlp_losses, label='MLP', color='purple')
    ax[0].plot(mlqp_times, mlqp_losses, label='MLQP', color='darkorange')
    ax[0].legend(loc='upper right', fontsize=9)
    ax[0].set_xlabel('CPU time(s)', size=12)
    ax[0].set_ylabel('Loss', size=12)

    # plot train acc curve
    ax[1].plot(mlp_train_accs, label='MLP_train', linestyle='-', color='purple', antialiased=True)
    ax[1].plot(mlqp_train_accs, label='MLQP_train', linestyle='-', color='darkorange', antialiased=True)
    ax[1].plot(mlp_test_accs, label='MLP_test', linestyle='--', color='purple', antialiased=True)
    ax[1].plot(mlqp_test_accs, label='MLQP_test', linestyle='--', color='darkorange', antialiased=True)
    ax[1].legend(loc='upper left', fontsize=9)
    ax[1].set_xlabel('Epoch', size=12)
    ax[1].set_ylabel("Accuracy", size=12)

    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "train_curve.png"))

def go_async(index, value):
    return str(index * int(value))

if __name__ == '__main__':
    plot_train_curves()