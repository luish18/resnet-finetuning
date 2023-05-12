import os
from pathlib import Path
import torch
from .utils import get_class_freq


URL = os.getenv("URL")
DATA_PATH = Path("../data/")
MODELS_PATH = Path("../models/model.pt")
LOG_PATH = Path("../logs/runs")

INPUT_SHAPE = (3, 224, 244)
BATCH_SIZE = 32
SEED = 42
SPLIT = 0.8
N_EPOCHS = 100


LR = 1e-2
LR_STEP = 5
GAMMA = 0.1
F_BETA = 0.5

classes = get_class_freq(DATA_PATH)

N_IMAGENS = classes["frequencia"].sum()
N_CLASSES = len(classes["classe"])
NUM_WORKERS = 10


DEVICE = torch.device("mps")

freq = get_class_freq(DATA_PATH)

# mapeamento de cada label para um inteiro
label_mapping = {
    classe: torch.tensor(valor)
    for classe, valor in zip(freq.classe.values, freq.index.values)
}