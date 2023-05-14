#!

import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.utils.tensorboard as tboard
import torchdata.dataloader2 as dl2
import torchmetrics as tmetrics
import torchmetrics.classification as cmetrics
from torchvision.models.resnet import ResNet18_Weights, resnet18

from utils.constants import *
from utils.datapipes import file_pipe, image_encoder_to_tensor, path_to_label
from utils.trainingLoops import train


def create_model(device, n_classes):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # mudando tamanho de saida da ultima camada
    new_input = model.fc.in_features

    last_layer = nn.Sequential(nn.Linear(in_features=new_input, out_features=n_classes))
    model.fc = last_layer

    model = model.to(device)

    return model


def combined(example):
    weights = ResNet18_Weights.DEFAULT
    preproc_transforms = weights.transforms()
    return path_to_label(example[0]), preproc_transforms(
        image_encoder_to_tensor(example[1])
    )


def worker_init_fn(datapipe, worker_info):
    warnings.filterwarnings(action="ignore")
    torch.set_warn_always(False)
    os.environ["MKL_VERBOSE"] = "0"
    return datapipe


def main():
    torch.set_warn_always(False)

    os.environ["MKL_VERBOSE"] = "0"

    ## criando pipelines de treinamento
    # instanciando os pesos do modelo e a funcao de pre-processamento

    # criando pipelines de treinamento e validacao
    train_pipe, val_pipe = file_pipe(
        file=str(DATA_PATH),
        N_images=N_IMAGENS,
        seed=SEED,
        split=SPLIT,
        transform=combined,
        batch_size=BATCH_SIZE,
        cache_size=MAX_CACHE_SIZE,
    )

    reading_service = dl2.MultiProcessingReadingService(
        num_workers=NUM_WORKERS, worker_init_fn=worker_init_fn
    )
    train_loader = dl2.DataLoader2(datapipe=train_pipe, reading_service=reading_service)
    val_loader = dl2.DataLoader2(datapipe=val_pipe, reading_service=reading_service)

    ## criando modelo
    freq = get_class_freq(DATA_PATH)

    model = create_model(DEVICE, N_CLASSES)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params=params, lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=LR_STEP, gamma=GAMMA
    )

    ## loss function
    class_weights = torch.tensor(list(freq.negative_logprob))
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

    ## metrics
    train_metrics = (
        tmetrics.MetricCollection(
            {
                "acc": cmetrics.Accuracy(task="multiclass", num_classes=N_CLASSES),
                "fbeta": cmetrics.FBetaScore(
                    task="multiclass", beta=F_BETA, num_classes=N_CLASSES
                ),
            },
        )
        .train()
        .to(DEVICE)
    )
    val_metrics = (
        tmetrics.MetricCollection(
            {
                "acc": cmetrics.Accuracy(task="multiclass", num_classes=N_CLASSES),
                "fbeta": cmetrics.FBetaScore(
                    task="multiclass", beta=F_BETA, num_classes=N_CLASSES
                ),
            }
        )
        .eval()
        .to(DEVICE)
    )

    # tensorboard writer
    exp_name = datetime.now().strftime("%m:%d-%H:%M")
    # writes para vermos estatisticas no tensorboard
    writer = tboard.SummaryWriter(log_dir=(LOG_PATH / exp_name))

    try:
        ## traning loop
        train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_scheduler=lr_scheduler,
            device=DEVICE,
            N_epochs=N_EPOCHS,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            writer=writer,
        )
    finally:
        train_loader.shutdown()
        val_loader.shutdown()
        writer.close()

    ## salvando modelo
    # jit nao tem suporte para device "mps" do mac, retornamos o modelo para o cpu
    model.to(torch.device("cpu"))
    trained_model = torch.jit.script(model.state_dict())
    trained_model.save(MODELS_PATH)


if __name__ == "__main__":
    main()
