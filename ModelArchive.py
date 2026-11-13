"""Импорт и экспорт архитектуры с весами в ZIP-архиве."""

import io
import json
import os
import tempfile
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn

ACTIVATIONS = (
    "ReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "ELU",
    "SiLU",
    "Softmax",
)


def build_network(layers):
    """Проверяет описание слоёв и собирает сеть PyTorch."""
    if not isinstance(layers, list) or not layers:
        raise ValueError("Add at least one Linear layer")
    modules = []
    previous_width = None
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Invalid layer description")
        if layer.get("type") == "linear":
            in_features = layer.get("in_features")
            out_features = layer.get("out_features")
            if any(
                type(size) is not int or not 1 <= size <= 100000
                for size in (in_features, out_features)
            ):
                raise ValueError("Invalid Linear dimensions")
            if previous_width is not None and in_features != previous_width:
                raise ValueError("Linear layer dimensions do not match")
            modules.append(nn.Linear(in_features, out_features))
            previous_width = out_features
        elif (
            layer.get("type") == "activation"
            and layer.get("name") in ACTIVATIONS
        ):
            name = layer["name"]
            modules.append(
                nn.Softmax(dim=1) if name == "Softmax" else getattr(nn, name)()
            )
        else:
            raise ValueError("Unsupported layer")
    if previous_width is None:
        raise ValueError("Add at least one Linear layer")
    return nn.Sequential(*modules)


def zip_path(path):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Only .zip model files are supported")
    return path


def load_model(path, task):
    """Проверяет архив и возвращает новую сеть с её весами."""
    with ZipFile(zip_path(path)) as archive:
        for name in ("metadata.json", "weights.pt"):
            if archive.namelist().count(name) != 1:
                raise ValueError(
                    "The ZIP must contain metadata.json and weights.pt"
                )
        metadata = json.loads(archive.read("metadata.json"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("format_version") != 1
            or metadata.get("task") != task
        ):
            raise ValueError("Unsupported model format or task type")
        # Читаем state_dict и переносим веса на CPU для совместимости.
        weights = torch.load(
            io.BytesIO(archive.read("weights.pt")),
            map_location="cpu",
            weights_only=True,
        )
        # Проверяем размеры без выделения памяти под случайные веса.
        with torch.device("meta"):
            network = build_network(metadata.get("layers"))
        network.load_state_dict(weights, strict=True, assign=True)
    return metadata["layers"], network


def save_model(path, task, layers, network):
    path = zip_path(path)
    metadata = {"format_version": 1, "task": task, "layers": layers}
    weights = io.BytesIO()
    torch.save(network.state_dict(), weights)
    # Заменяем конечный файл только после записи полного архива.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".zip", delete=False
        ) as file:
            temporary = Path(file.name)
        with ZipFile(temporary, "w") as archive:
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            archive.writestr("weights.pt", weights.getvalue())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
