import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn,
    dataloader: DataLoader,
    device: torch.device,
    use_am,
    lr_schedule=None,
):
    """
    :param epoch: current epoch
    :param model: model to train
    :param optimizer: optimizer used to train model
    :param loss_fn: loss function
    :param dataloader: dataloader to train on
    :param device: device to use (e.g cuda, cpu, mps,...)
    :param lr_schedule: scheduler use for learning rate decay
    """
    model.train()

    loss_values = []
    correct_count = 0
    total_count = 0
    with tqdm(dataloader, unit="batch", ncols=0) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        if lr_schedule is not None:
            lr_schedule.step()
        for samples, labels in tepoch:
            if use_am:
                am = samples[:, -3:, :, :, :]
                am = am.to(device)
                samples = samples[:, :-3, :, :, :]
            else:
                am = None

            samples = samples.to(device)
            labels = labels.to(device)

            preds = model(samples, am)
            optimizer.zero_grad()
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())

            with torch.no_grad():
                pred_classes = torch.argmax(preds, dim=1)
                correct_count += (pred_classes == labels).sum().item()
                total_count += len(labels)
                tepoch.set_postfix(
                    lr=optimizer.param_groups[0]["lr"],
                    avg_loss=np.mean(np.array(loss_values)),
                    acc=correct_count / total_count,
                )

        avg_loss = sum(loss_values) / len(loss_values)

    acc = correct_count / total_count
    return avg_loss, acc


def valid_one_epoch(
    model: nn.Module, loss_fn, dataloader: DataLoader, device: torch.device,use_am,
):
    """
    :param model: model to evaluate
    :param loss_fn: loss function
    :param dataloader: dataloader to evaluate on
    :param device: device to use (e.g cuda, cpu, mps,...)
    """
    model.eval()

    preds_arr = []
    labels_arr = []

    loss_values = []
    correct_count = 0
    total_count = 0

    score = None
    with torch.no_grad():
        with tqdm(dataloader, unit="batch", ncols=0) as tepoch:
            tepoch.set_description("Validation")
            for samples, labels in tepoch:
                if use_am:
                    am = samples[:, -3:, :, :, :]
                    am = am.to(device)
                    samples = samples[:, :-3, :, :, :]
                else:
                    am = None
                samples = samples.to(device)
                labels = labels.to(device)

                preds = model(samples, am)

                score = preds if score is None else torch.cat([score, preds])

                loss = loss_fn(preds, labels)

                loss_values.append(loss.item())

                pred_classes = torch.argmax(preds, dim=1)
                correct_count += (pred_classes == labels).sum().item()

                preds_arr.extend(pred_classes.tolist())
                labels_arr.extend(labels.tolist())

                total_count += len(labels)
                tepoch.set_postfix(
                    avg_loss=np.mean(np.array(loss_values)),
                    acc=correct_count / total_count,
                )

            avg_loss = sum(loss_values) / len(loss_values)
            acc = correct_count / total_count

    return avg_loss, acc, preds_arr, labels_arr, score


def valid_ensemble_one_epoch(
    models: nn.ModuleList,
    dataloaders: list[DataLoader],
    device: torch.device,
    use_am,
    alphas=None,
):
    """
    :param models: list of models used to essemble
    :param dataloaders: list of dataloaders corresponding to provided models
    :param device: device to use (e.g cuda, cpu, mps,...)
    :param alphas: weight for scores of each model
    """
    dataloader_iters = []
    for dataloader in dataloaders:
        dataloader_iters.append(iter(dataloader))

    models.eval()

    preds_arr = []
    labels_arr = []

    correct_count = 0
    total_count = 0
    with torch.no_grad():
        with tqdm(range(len(dataloaders[0])), unit="batch", ncols=0) as tepoch:
            tepoch.set_description("Validation")
            for _ in tepoch:
                probs = torch.zeros(0)
                labels = torch.empty(0)
                for i in range(len(models)):
                    samples, labels = next(dataloader_iters[i])
                    if use_am:
                        am = samples[:, -3:, :, :, :]
                        am = am.to(device)
                        samples = samples[:, :-3, :, :, :]
                    else:
                        am = None

                    samples = samples.to(device)
                    labels = labels.to(device)
                    preds = models[i](samples, am)

                    if alphas is not None:
                        probs = (
                            probs + alphas[i] * preds
                            if probs.size(0) != 0
                            else alphas[i] * preds
                        )
                    else:
                        probs = probs + preds if probs.size(0) != 0 else preds

                pred_classes = torch.argmax(probs, dim=1)
                correct_count += (pred_classes == labels).sum().item()

                preds_arr.extend(pred_classes.tolist())
                labels_arr.extend(labels.tolist())

                total_count += len(labels)
                tepoch.set_postfix(
                    acc=correct_count / total_count,
                )

            acc = correct_count / total_count

    return acc, preds_arr, labels_arr
