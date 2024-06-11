from utils import utils
from tqdm.auto import tqdm
import torch
from torch import nn
import numpy as np

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion_instance,
               criterion_cluster,
               optimizer: torch.optim.Optimizer,
               scheduler,
               weight,
               device=device):
    # Put the model in train_mode
    model.train()

    # Setup train loss
    train_loss = 0

    # Loop through data loader batches
    for batch, (X_original, X_augmented) in enumerate(dataloader):
        # Send data to target device
        X_original = X_original.to(device)
        X_augmented = X_augmented.to(device)

        # 1. Forward pass
        z_org, z_aug, c_org, c_aug = model(X_original, X_augmented)

        # 2. Calculate the loss
        loss_instance = criterion_instance(z_org, z_aug)
        loss_cluster = criterion_cluster(c_org, c_aug)
        loss = weight * loss_instance + (1 - weight) * loss_cluster
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        scheduler.step()

    train_loss = train_loss / len(dataloader)
    return train_loss


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
              device=device):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy
    test_loss, test_acc = 0, 0
    # Turn on inference mode
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model.forward_cluster(X)
            # print(test_pred_logits)

            # # 2. Calculate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. Calculate the accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            ind, batch_acc = utils.cluster_acc(y.cpu().numpy(), test_pred_labels.cpu().numpy())
            test_acc += batch_acc

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler,
          weight,
          criterion_instance,
          criterion_cluster,
          epochs: int,
          device=device):
    # create empty results dictionary
    results = {"train_loss": [],
               "test_loss": [],
               "test_acc": []}
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        lr = optimizer.param_groups[0]["lr"]

        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                criterion_instance=criterion_instance,
                                criterion_cluster=criterion_cluster,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                weight=weight,
                                device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        device=device)
        if epoch % 5 == 0:
            print(
                f"Epoch: {epoch} | Train loss: {train_loss:.6f} | Test loss: {test_loss:.6f} | Test accuracy: {test_acc * 100:.4f}%")
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def model_evaluation(model: torch.nn.Module,
                     X_test,
                     y_test):
    model.eval()
    y_pred = []
    y_prob = []
    with torch.inference_mode():
        for x in X_test:
            x = torch.Tensor(x).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
            y_pred.append(model.forward_cluster(x).argmax(dim=1).item())
            y_prob.append(model.forward_cluster(x).cpu().numpy()[0])
    ind, acc = utils.cluster_acc(np.array(y_test), np.array(y_pred))
    return np.array(y_prob), y_pred, ind, acc
