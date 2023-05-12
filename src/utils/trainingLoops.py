from time import time

import torch
import torch.nn as nn
import torch.utils.tensorboard as tboard
import torchdata.dataloader2 as dl2
import torchmetrics as tmetrics
from tqdm import trange


def train_epoch(
    model: nn.Module,
    optimizer,
    criterion: nn.Module,
    data_loader: dl2.DataLoader2,
    device: torch.DeviceObjType,
    metrics: tmetrics.MetricCollection,
    tboard_writer: tboard.SummaryWriter,
    n_iter: int,
) -> float:
    # modelo em modo de treinamento
    model.train()

    running_loss = 0.0

    for i, (labels, imgs) in enumerate(data_loader):
        labels = labels.to(device)

        imgs = imgs.to(device)

        # zerando gradientes da ultima atualizacao
        optimizer.zero_grad()

        # forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

        # logging
        running_loss += float(loss)

        n_iter += i
        tboard_writer.add_scalar("loss/train/batch/", float(loss), n_iter)
        tboard_writer.add_scalars(
            main_tag="metrics/train/batch",
            tag_scalar_dict=metrics(
                torch.max(torch.softmax(outputs, dim=1), dim=1)[1], labels
            ),
            global_step=n_iter,
        )

    tboard_writer.add_scalar("loss/train/epoch", running_loss, n_iter)
    tboard_writer.add_scalars(
        main_tag="metrics/train/epoch",
        tag_scalar_dict=metrics.compute(),
        global_step=n_iter,
    )

    metrics.reset()

    return running_loss, n_iter


def eval(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: dl2.DataLoader2,
    device: torch.DeviceObjType,
    metrics: tmetrics.MetricCollection,
    tboard_writer: tboard.SummaryWriter,
    n_iter: int,
) -> float:
    # modelo em modo de validacao
    model.eval()

    # equivalente a nograd
    running_loss = 0.0
    with torch.inference_mode():
        for i, (labels, imgs) in enumerate(data_loader):
            labels = labels.to(device)
            imgs = imgs.to(device)

            # forward pass
            outputs = model(imgs)

            # erro
            loss = criterion(outputs, labels)
            running_loss += float(loss)

            n_iter += i
            tboard_writer.add_scalar("loss/eval/batch/", float(loss), n_iter)
            tboard_writer.add_scalars(
                main_tag="metrics/eval/batch",
                tag_scalar_dict=metrics(
                    torch.max(torch.softmax(outputs, dim=1), dim=1)[1], labels
                ),
                global_step=n_iter,
            )

    tboard_writer.add_scalar("loss/eval/epoch", running_loss, n_iter)
    tboard_writer.add_scalars(
        main_tag="metrics/eval/epoch",
        tag_scalar_dict=metrics.compute(),
        global_step=n_iter,
    )
    metrics.reset()

    return running_loss, n_iter


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: dl2.DataLoader2,
    val_loader: dl2.DataLoader2,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.DeviceObjType,
    N_epochs: int,
    train_metrics: tmetrics.MetricCollection,
    val_metrics: tmetrics.MetricCollection,
    writer: tboard.SummaryWriter,
):
    train_loss = eval_loss = lr = ttime = 0.0

    pbar = trange(
        N_epochs,
        desc="Epocas",
        unit="epochs",
        postfix={
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "learning_rate": lr,
            "train_batch_time": ttime,
        },
        position=0,
    )
    train_i = val_i = 0
    for _ in pbar:
        start_time = time()
        train_loss, train_i = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            metrics=train_metrics,
            tboard_writer=writer,
            n_iter=train_i,
        )
        ttime = time() - start_time

        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()

        eval_loss, val_i = eval(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device,
            metrics=val_metrics,
            tboard_writer=writer,
            n_iter=val_i,
        )

        pbar.set_postfix(
            {
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "learning_rate": lr,
                "train_batch_time": ttime,
            }
        )

    train_loader.shutdown()
    val_loader.shutdown()

    writer.close()
