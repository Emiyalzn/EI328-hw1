import time
import pickle
import os
import numpy as np
import argparse
from utils import load_data, mse_loss, plot_boundaries, partition_data, plot_minmax_boundaries, result_dir, plot_datapoints
from model import MLP, MLQP
from copy import deepcopy
from multiprocessing import Pool

def parse_arguments():
    parser = argparse.ArgumentParser("Training hyperparameters.")
    parser.add_argument('--lr_1', type=float, default=1e-2, help="learning rate for v and b")
    parser.add_argument('--lr_2', type=float, default=1e-2, help="learning rate for u")
    parser.add_argument('--alpha_1', type=float, default=0., help="momentum rate for v and b")
    parser.add_argument('--alpha_2', type=float, default=0., help="momentum rate for u")
    parser.add_argument('--n_hid', type=int, default=32, help="size of hidden layer")
    parser.add_argument('--n_epoch', type=int, default=30000, help="Max training epochs")
    parser.add_argument('--type', choices=['vanilla', 'minmax'], default='vanilla', help="choose the training model type")
    parser.add_argument('--model', choices=['mlp', 'mlqp'], default='mlp', help="use mlp or mlqp for training")
    parser.add_argument('--partition_num', type=int, default=2, help="number of training set partitions")
    parser.add_argument('--partition_mode', choices=['random', 'yaxis', 'yaxis+overlap'], default='random', help="data partition method")
    parser.add_argument('--train_mode', choices=['sequential', 'parallel'], default='parallel', help="how to train the submodules")
    parser.add_argument('--seed', type=int, default=None, help="fix the random seed")
    args = parser.parse_args()
    return args

def train_vanillanet(args):
    xs, ys, labels = load_data('train')
    x_s, y_s, label_s = load_data('test')
    train_data = (xs, ys, labels)
    test_data = (x_s, y_s, label_s)
    train_accs, test_accs, losses, times = [], [], [], []
    best_epoch, best_loss = 0, 1e6

    # initialize models
    if args.model == 'mlp':
        model = MLP([2, args.n_hid, 1])
    else:
        model = MLQP([2, args.n_hid, 1])

    # start training
    start_time = time.time()
    for epoch in range(args.n_epoch):
        for i in range(len(xs)):
            pred = model.forward(np.array([xs[i], ys[i]]))
            model.backward(labels[i])
            if args.model == 'mlp':
                model.update(args.alpha_1, args.lr_1)
            else:
                model.update(args.alpha_1, args.alpha_2, args.lr_1, args.lr_2)
        curr_time = time.time() - start_time
        times.append(curr_time)
        # predict on train and test data
        train_acc, train_loss = predict(model, train_data)
        test_acc, _ = predict(model, test_data)
        losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Time {curr_time:.2f}, Loss {train_loss:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:.4f}")
        
        # early stopping
        if best_loss - train_loss > 0.0001:
            best_loss = train_loss
            best_epoch = epoch
        elif epoch - best_epoch >= 200:
            break

    # save files
    plot_boundaries(model, args.model, args.lr_1, args.lr_2, args.alpha_1)
    save_file = open(os.path.join(result_dir, f"{args.model}_data.pkl"), 'wb')
    pickle.dump(times, save_file)
    pickle.dump(losses, save_file)
    pickle.dump(train_accs, save_file)
    pickle.dump(test_accs, save_file)
    save_file.close()

def train_minmax_sequential(args):
    xs, ys, labels = load_data('train')
    x_s, y_s, label_s = load_data('test')
    train_data = (xs, ys, labels)
    test_data = (x_s, y_s, label_s)
    white_subsets, black_subsets = partition_data(train_data, args.partition_mode, args.partition_num)
    plot_datapoints(white_subsets, black_subsets)
    train_accs, test_accs, losses, times = [], [], [], []
    best_epoch, best_loss = 0, 1e6

    # initialize models
    models = []
    for i in range(args.partition_num):
        models.append([])
        for j in range(args.partition_num):
            if args.model == 'mlp':
                models[i].append(MLP([2, args.n_hid, 1]))
            else:
                models[i].append(MLQP([2, args.n_hid, 1]))

    # start training
    start_time = time.time()
    for epoch in range(args.n_epoch):
        for i in range(args.partition_num):
            for j in range(args.partition_num):
                x_train = np.concatenate((white_subsets[i][0], black_subsets[j][0]), axis=0)
                y_train = np.concatenate((white_subsets[i][1], black_subsets[j][1]), axis=0)
                labels_train = np.concatenate((white_subsets[i][2], black_subsets[j][2]), axis=0)
                for k in range(len(x_train)):
                    pred = models[i][j].forward(np.array([x_train[k], y_train[k]]))
                    models[i][j].backward(labels_train[k])
                    if args.model == 'mlp':
                        models[i][j].update(args.alpha_1, args.lr_1)
                    else:
                        models[i][j].update(args.alpha_1, args.alpha_2, args.lr_1, args.lr_2)
        curr_time = time.time() - start_time
        times.append(curr_time)
        # predict on train and test data
        train_acc, train_loss = predict_minmax(models, train_data)
        test_acc, _ = predict_minmax(models, test_data)
        losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Time {curr_time:.2f}, Loss {train_loss:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:.4f}")

        # early stopping
        if best_loss - train_loss > 0.0001:
            best_loss = train_loss
            best_epoch = epoch
        elif epoch - best_epoch >= 200:
            break

    # Visualization and save files
    plot_minmax_boundaries(models, args.model, args.partition_mode)
    save_file = open(os.path.join(result_dir, f"minmax_{args.model}_data.pkl"), 'wb')
    pickle.dump(times, save_file)
    pickle.dump(losses, save_file)
    pickle.dump(train_accs, save_file)
    pickle.dump(test_accs, save_file)
    save_file.close()

def train_one_model(model, data, args):
    xs, ys, labels = data
    best_epoch, best_loss = 0, 1e6
    for epoch in range(args.n_epoch):
        for k in range(len(xs)):
            pred = model.forward(np.array([xs[k], ys[k]]))
            model.backward(labels[k])
            if args.model == 'mlp':
                model.update(args.alpha_1, args.lr_1)
            else:
                model.update(args.alpha_1, args.alpha_2, args.lr_1, args.lr_2)
        _, train_loss = predict(model, data)

        # early stopping
        if best_loss - train_loss > 0.0001:
            best_loss = train_loss
            best_epoch = epoch
        elif epoch - best_epoch >= 200:
            break
    return model

def step_func_feeder(args, models, white_subsets, black_subsets):
    partition_num = len(white_subsets)
    for i in range(partition_num):
        for j in range(partition_num):
            x_train = np.concatenate((white_subsets[i][0], black_subsets[j][0]), axis=0)
            y_train = np.concatenate((white_subsets[i][1], black_subsets[j][1]), axis=0)
            labels_train = np.concatenate((white_subsets[i][2], black_subsets[j][2]), axis=0)
            data_train = (x_train, y_train, labels_train)
            yield  models[i][j], data_train, args

def train_minmax_parallel(args):
    num_workers = args.partition_num * args.partition_num
    mp_pool = Pool(num_workers)
    xs, ys, labels = load_data('train')
    x_s, y_s, label_s = load_data('test')
    train_data = (xs, ys, labels)
    test_data = (x_s, y_s, label_s)
    white_subsets, black_subsets = partition_data(train_data, args.partition_mode, args.partition_num)
    plot_datapoints(white_subsets, black_subsets)

    # initialize models
    models = []
    for i in range(args.partition_num):
        models.append([])
        for j in range(args.partition_num):
            if args.model == 'mlp':
                models[i].append(MLP([2, args.n_hid, 1]))
            else:
                models[i].append(MLQP([2, args.n_hid, 1]))

    start_time = time.time()

    pool_map = mp_pool.starmap_async(train_one_model, step_func_feeder(args, models, white_subsets, black_subsets))
    results = pool_map.get()
    mp_pool.close()
    mp_pool.join()

    for i in range(args.partition_num):
        for j in range(args.partition_num):
            models[i][j] = results[i*args.partition_num+j]
    train_acc, train_loss = predict_minmax(models, train_data)
    test_acc, _ = predict_minmax(models, test_data)
    curr_time = time.time() - start_time
    print(f"Time {curr_time:.2f}, Loss {train_loss:.4f}, Train acc {train_acc:.4f}, Test acc {test_acc:.4f}")

    # Visualization
    plot_minmax_boundaries(models, args.model, args.partition_mode)

def predict(model, data):
    xs, ys, labels = data
    preds = []
    for i in range(len(xs)):
        preds.append(model.forward(np.array([xs[i],ys[i]])))
    preds = np.squeeze(np.array(preds))
    loss = mse_loss(preds, labels)
    acc = ((preds > 0.5) == labels).mean()
    return acc, loss

def predict_minmax(models, data):
    num_partitions = len(models)
    xs, ys, labels = data
    preds = []
    min_results = []
    single_pred = np.zeros(len(xs))
    for i in range(num_partitions):
        preds.append([])
        for j in range(num_partitions):
            for k in range(len(xs)):
                single_pred[k] = models[i][j].forward(np.array([xs[k],ys[k]]))
            preds[i].append(deepcopy(single_pred))
    for i in range(num_partitions):
        min_result = np.min(preds[i],axis=0)
        min_results.append(deepcopy(min_result))
    max_result = np.max(min_results, axis=0)
    loss = mse_loss(max_result, labels)
    acc = ((max_result > 0.5) == labels).mean()
    return acc, loss

if __name__ == '__main__':
    args = parse_arguments()
    if args.seed != None:
        np.random.seed(args.seed)
    if args.type == 'vanilla':
        train_vanillanet(args)
    elif args.type == 'minmax':
        if args.train_mode == 'parallel':
            train_minmax_parallel(args)
        else:
            train_minmax_sequential(args)
    else:
        raise ValueError("Unknown model type!")
