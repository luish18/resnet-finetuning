import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_class_freq(path: Path) -> pd.DataFrame:
    """Obtém as classes e quantidade de imagens em cada uma delas

    Args:
        path (Path): path para as pastas com as imagens de cada classe

    Returns:
        dict[str, int]: dicionário com cada uma das classes e sua frequência no dataset
    """
    class_paths = path.glob("*/**")
    class_freq: dict[str, int] = {}

    for folder in class_paths:
        class_len: int = len(list(folder.glob("*jpg"))) + len(
            list(folder.glob("*.png"))
        )

        class_freq[folder.name] = class_len

    freq = (
        pd.DataFrame.from_dict(class_freq, orient="index")
        .reset_index()
        .rename({0: "frequencia", "index": "classe"}, axis=1)
    )
    total = freq["frequencia"].sum()
    freq["probs"] = freq["frequencia"] / total
    freq["negative_logprob"] = -np.log2(freq["probs"])

    return freq


URL = os.getenv("URL")
DATA_PATH = Path("/Users/luishf/Documents/GitHub/resnet-finetuning/data")
MODELS_PATH = Path("/Users/luishf/Documents/GitHub/resnet-finetuning/models/model.pt")
LOG_PATH = Path("/Users/luishf/Documents/GitHub/resnet-finetuning/logs/runs")

INPUT_SHAPE = (3, 224, 244)
BATCH_SIZE = 64
SEED = 42
SPLIT = 0.8
N_EPOCHS = 100
MAX_CACHE_SIZE = 256


LR = 1e-3
LR_STEP = 10
GAMMA = 0.1
F_BETA = 1.0

classes = get_class_freq(DATA_PATH)

N_IMAGENS = classes.frequencia.sum()
N_CLASSES = len(classes["classe"])
NUM_WORKERS = 4


DEVICE = torch.device("mps")

freq = get_class_freq(DATA_PATH)

# mapeamento de cada label para um inteiro
label_mapping = {
    classe: torch.tensor(valor)
    for classe, valor in zip(freq.classe.values, freq.index.values)
}
