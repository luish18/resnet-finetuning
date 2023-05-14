from io import BytesIO
from pathlib import Path

import torchdata.datapipes.iter as idp
import torchvision.transforms as TF
from PIL import Image

from .constants import label_mapping


def path_to_label(item):
    return label_mapping[Path(item).parent.name]


def path_func(path):
    temp_dir = Path("../temp/")
    return str(temp_dir / Path(path).name)


def filter_imgs(file: tuple[str, Image.Image]):
    return file.endswith((".jpg", ".png"))


def image_encoder_to_tensor(example):
    return TF.PILToTensor()(Image.open(BytesIO(example)))


def file_pipe(
    file: Path | str,
    N_images: int,
    seed: int,
    split: float,
    transform: TF.Compose,
    batch_size: int = 32,
    cache_size: int = None,
) -> tuple[idp.IterDataPipe, idp.IterDataPipe]:
    IMAGEM_IDX = 1
    LABEL_IDX = 0

    # link para o arquivo no gdrive
    files = idp.FileLister(file, recursive=True, masks=["**.jpg", "**.png"])

    data_pipe = (
        files.open_files(mode="rb")
        .read_from_stream()
        .map(transform)
        .shuffle()
        .sharding_filter()
        .batch(batch_size=batch_size, drop_last=True)
        .prefetch(N_images)
        .collate()
        .in_memory_cache(cache_size)
    )

    split_dict = {"train": split, "valid": 1 - split}
    train, valid = data_pipe.random_split(
        weights=split_dict, total_length=N_images, seed=seed
    )

    return train, valid


def url_pipe(
    url: str,
    N_images: int,
    seed: int,
    split: float,
    transform: TF.Compose,
    batch_size: int = 32,
    cache_size: int = 1024,
) -> tuple[idp.IterDataPipe, idp.IterDataPipe]:
    IMAGEM_IDX = 1
    LABEL_IDX = 0

    # link para o arquivo no gdrive
    url_wrapper = idp.IterableWrapper([url])

    # cache do dataset em disco
    disk_cache = (
        url_wrapper.on_disk_cache(filepath_fn=path_func)
        .read_from_gdrive()
        .end_caching(same_filepath_fn=True)
    )

    data_pipe = (
        disk_cache.open_files(mode="rb")
        .load_from_zip()
        .filter(filter_fn=filter_imgs, input_col=LABEL_IDX)
        .map(path_to_label, input_col=LABEL_IDX, output_col=LABEL_IDX)
        .read_from_stream()
        .map(image_encoder_to_tensor, input_col=IMAGEM_IDX, output_col=IMAGEM_IDX)
    )

    if transform:
        data_pipe = data_pipe.map(
            transform, input_col=IMAGEM_IDX, output_col=IMAGEM_IDX
        )

    data_pipe = (
        data_pipe.shuffle()
        .in_memory_cache(cache_size)
        .prefetch(batch_size)
        .batch(batch_size=batch_size, drop_last=True)
        .collate()
    )

    split_dict = {"train": split, "valid": 1 - split}
    train, valid = data_pipe.random_split(
        weights=split_dict, total_length=N_images, seed=seed
    )

    return train, valid
