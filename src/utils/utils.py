from pathlib import Path
import pandas as pd
import numpy as np

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
        class_len: int = len(list(folder.glob("*.jpg"))) + len(
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