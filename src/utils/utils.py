import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale


def data_normalization(X_train, X_test):
    X_train_normalized = []
    X_test_normalized = []

    for x in X_test:
        X_test_normalized.append(minmax_scale(x.flatten()).reshape(x.shape))
    for s1, s2 in X_train:
        X_train_normalized.append(
            ((minmax_scale(s1.flatten()).reshape(s1.shape)), (minmax_scale(s2.flatten()).reshape(s2.shape))))

    X_test = np.array(X_test_normalized)
    X_train = np.array(X_train_normalized)

    X_train_normalized = []
    X_test_normalized = []
    for x in X_test:
        mean_val = np.mean(x)
        std_val = np.std(x)
        normalized = (x - mean_val) / std_val
        X_test_normalized.append(normalized)
    for s1, s2 in X_train:
        mean_val_s1 = np.mean(s1)
        std_val_s1 = np.std(s1)
        normalized_s1 = (s1 - mean_val_s1) / std_val_s1
        mean_val_s2 = np.mean(s2)
        std_val_s2 = np.std(s2)
        normalized_s2 = (s2 - mean_val_s2) / std_val_s2
        X_train_normalized.append((normalized_s1, normalized_s2))

    X_test_normalized = np.array(X_test_normalized)
    X_train_normalized = np.array(X_train_normalized)
    return X_train_normalized, X_test_normalized


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return ind, sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def display_random_fcgr_pairs(dataset: torch.utils.data.Dataset,
                              classes: List[str] = None,
                              n: int = 10,
                              display_shape: bool = True,
                              seed: int = None):
    # Adjust display if n is too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display")

    # set the seed
    if seed:
        random.seed(seed)

    # Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(8, 8))

    # 6. Loop through the indexes and plot them
    for i, targ_sample in enumerate(random_samples_idx):
        org_fcgr, mim_fcgr = dataset[targ_sample]
        # print(targ_fcgr.unsqueeze(dim=0).shape)
        # print(targ_label)

        # 7. Adjust tensor dimension for plotting
        org_fcgr_adjust = org_fcgr.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
        mim_fcgr_adjust = mim_fcgr.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

        # plot adjusted sample
        plt.subplot(n, 3, 3 * i + 1)
        plt.imshow(1 - org_fcgr_adjust, cmap="gray")
        plt.axis(False)
        plt.title("Original FCGR")

        # plot adjusted sample
        plt.subplot(n, 3, 3 * i + 2)
        plt.imshow(1 - mim_fcgr_adjust, cmap="gray")
        plt.axis(False)
        plt.title("Augmented FCGR")

        # plot adjusted sample
        plt.subplot(n, 3, 3 * i + 3)
        plt.imshow(org_fcgr_adjust - mim_fcgr_adjust, cmap="gray")
        plt.axis(False)
        plt.title("Difference")


def plot_loss_curves(results: Dict[str, List[float]],
                     total_time: float):
    """Plots training curve of a result dictionary"""
    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary
    # acc = results["train_acc"]
    test_acc = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))
    plt.suptitle(f"Training time:  {total_time:.3f} seconds")

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, torch.tensor(loss).detach().cpu().numpy(), label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    # plt.plot(epochs, acc, label="train_accuracy")
    plt.plot(epochs, test_acc, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();
