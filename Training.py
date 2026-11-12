"""Sequential models, dataset validation and portable training checkpoints."""

import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    precision_recall_fscore_support, r2_score,
)
from torch import nn


ACTIVATIONS = ("ReLU", "GELU", "Sigmoid", "Tanh", "LeakyReLU",
               "ELU", "SiLU", "Softmax")
SPLITS = ("Train", "Validation", "Test")
MAX_PARAMETERS = 10_000_000


def validate_architecture(layers):
    if not isinstance(layers, list) or not layers or len(layers) > 256:
        raise ValueError("Add between 1 and 256 layers, including a Linear layer.")
    width = None
    parameters = 0
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("Invalid layer description.")
        if layer.get("type") == "linear":
            sizes = [layer.get("in_features"), layer.get("out_features")]
            if any(type(size) is not int or not 1 <= size <= 100000 for size in sizes):
                raise ValueError("Linear dimensions must be between 1 and 100000.")
            input_size, output_size = sizes
            if width is not None and width != input_size:
                raise ValueError("Adjacent Linear dimensions do not match.")
            width = output_size
            parameters += (input_size + 1) * output_size
        elif layer.get("type") == "activation":
            if layer.get("activation") not in ACTIVATIONS:
                raise ValueError("Unknown activation.")
        else:
            raise ValueError("Unknown layer type.")
    if width is None:
        raise ValueError("Add at least one Linear layer.")
    if parameters > MAX_PARAMETERS:
        raise ValueError("This model exceeds the 10 million parameter limit.")


def build_sequential(layers):
    validate_architecture(layers)
    modules = []
    for layer in layers:
        if layer["type"] == "linear":
            module = nn.Linear(layer["in_features"], layer["out_features"])
        elif layer["activation"] == "Softmax":
            module = nn.Softmax(dim=1)
        else:
            module = getattr(nn, layer["activation"])()
        modules.append(module)
    return nn.Sequential(*modules)


def json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Class labels and feature names must be strings or finite numbers.")


def prepare_datasets(inputs, task, classes=None, feature_columns=None):
    """Fit the label mapping on Train and use it unchanged for optional splits."""
    if task not in ("regression", "classification"):
        raise ValueError("Unknown model task.")
    if len(inputs) != 6 or inputs[0] is None or inputs[1] is None:
        raise ValueError("Connect Train Features and Train Target.")
    train_x, train_y = inputs[:2]
    if not isinstance(train_x, pd.DataFrame) or not isinstance(train_y, pd.Series):
        raise ValueError("Features must be a DataFrame and Target must be a Series.")
    columns = list(train_x.columns)
    if not train_x.columns.is_unique or not columns:
        raise ValueError("Feature names must be unique; at least one feature is required.")
    if feature_columns is not None and columns != feature_columns:
        raise ValueError("Train feature names/order differ from the saved model.")
    if task == "classification":
        if train_y.isna().any():
            raise ValueError("Train Target contains missing labels.")
        if classes is None:
            classes = [json_scalar(value) for value in pd.unique(train_y)]
        if len(classes) < 2:
            raise ValueError("Classification requires at least two Train classes.")
    prepared = {}
    for index, split in enumerate(SPLITS):
        features, target = inputs[2 * index:2 * index + 2]
        if features is None and target is None:
            continue
        if features is None or target is None:
            raise ValueError(f"Connect both {split} Features and {split} Target.")
        if not isinstance(features, pd.DataFrame) or not isinstance(target, pd.Series):
            raise ValueError(f"{split}: expected DataFrame features and Series target.")
        if not len(features) or len(features) != len(target):
            raise ValueError(f"{split}: Features/Target must have equal, nonzero lengths.")
        if not features.index.equals(target.index):
            raise ValueError(f"{split}: Features/Target indexes do not match.")
        if not features.columns.is_unique or set(features.columns) != set(columns):
            raise ValueError(f"{split}: feature columns differ from Train.")
        try:
            x = features.loc[:, columns].to_numpy(dtype=np.float32, copy=True)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{split}: encode nonnumeric features before training.") from error
        if not np.isfinite(x).all():
            raise ValueError(f"{split}: Features contain NaN or infinity.")
        if task == "classification":
            mapping = {label: i for i, label in enumerate(classes)}
            encoded = target.map(mapping)
            if encoded.isna().any():
                raise ValueError(f"{split}: Target contains missing or unknown Train classes.")
            y = torch.tensor(encoded.to_numpy(dtype=np.int64), dtype=torch.long)
        else:
            try:
                values = target.to_numpy(dtype=np.float32, copy=True)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{split}: regression Target must be numeric.") from error
            if not np.isfinite(values).all():
                raise ValueError(f"{split}: Target contains NaN or infinity.")
            y = torch.tensor(values).reshape(-1, 1)
        prepared[split] = (torch.from_numpy(x), y)
    return prepared, classes, [json_scalar(column) for column in columns]


class TrainingSession:
    """Own weights and optimizer between runs; only one worker uses a session."""

    def __init__(self, task, layers):
        validate_architecture(layers)
        self.task = task
        self.layers = layers
        self.model = None
        self.optimizer = None
        self.classes = None
        self.feature_columns = None
        self.epochs = 0
        self.lr = 1e-3
        self.steps = 100
        self.last_metrics = {}

    def prepare(self, inputs, lr):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be positive and finite.")
        prepared, classes, columns = prepare_datasets(
            inputs, self.task, self.classes, self.feature_columns,
        )
        first = next(layer for layer in self.layers if layer["type"] == "linear")
        last = next(layer for layer in reversed(self.layers) if layer["type"] == "linear")
        if first["in_features"] != len(columns):
            raise ValueError(f"First Linear layer needs {len(columns)} input features.")
        outputs = len(classes) if self.task == "classification" else 1
        if last["out_features"] != outputs:
            raise ValueError(f"Last Linear layer needs {outputs} output features.")
        if self.task == "classification" and self.layers[-1]["type"] != "linear":
            raise ValueError("CrossEntropy requires a final Linear layer (raw logits).")
        if self.model is None:
            self.model = build_sequential(self.layers)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.classes = classes
        self.feature_columns = columns
        self.lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return prepared

    def criterion(self):
        return nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()

    def train_epoch(self, datasets, stop_requested, batch_size=256):
        self.model.train()
        x, y = datasets["Train"]
        order = torch.randperm(len(x))
        criterion = self.criterion()
        for start in range(0, len(x), batch_size):
            if stop_requested():
                return False
            indices = order[start:start + batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            loss = criterion(self.model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError("Loss is not finite. Check the data and learning rate.")
            loss.backward()
            self.optimizer.step()
        self.epochs += 1
        return True

    @torch.no_grad()
    def evaluate(self, datasets):
        self.model.eval()
        metrics = {}
        for split, (x, y) in datasets.items():
            predictions = []
            loss_sum = 0.0
            for start in range(0, len(x), 256):
                output = self.model(x[start:start + 256])
                if not torch.isfinite(output).all():
                    raise ValueError("Model predictions are not finite.")
                loss_sum += self.criterion()(output, y[start:start + 256]).item() * len(output)
                predictions.append(output.argmax(dim=1) if self.task == "classification"
                                   else output.reshape(-1))
            predicted = torch.cat(predictions).numpy()
            expected = y.reshape(-1).numpy()
            row = {"Loss": loss_sum / len(x)}
            if self.task == "classification":
                precision, recall, f1, _ = precision_recall_fscore_support(
                    expected, predicted, labels=list(range(len(self.classes))),
                    average="macro", zero_division=0,
                )
                row.update(Accuracy=accuracy_score(expected, predicted),
                           Precision=precision, Recall=recall, F1=f1)
            else:
                row.update(RMSE=math.sqrt(mean_squared_error(expected, predicted)),
                           MAE=mean_absolute_error(expected, predicted),
                           R2=r2_score(expected, predicted) if len(expected) > 1 else None)
            metrics[split] = row
        self.last_metrics = metrics
        return metrics

    def export_zip(self, filename):
        if self.model is None or self.optimizer is None:
            raise ValueError("Train or import a model before exporting.")
        metadata = {
            "format": "weave-sequential", "version": 1, "task": self.task,
            "layers": self.layers, "classes": self.classes,
            "feature_columns": self.feature_columns, "epochs": self.epochs,
            "lr": self.lr, "steps": self.steps, "metrics": self.last_metrics,
        }
        weights = io.BytesIO()
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, weights)
        template = self.model_template()
        destination = Path(filename)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = stream.name
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("metadata.json", json.dumps(metadata, allow_nan=False))
                archive.writestr("weights.pt", weights.getvalue())
                archive.writestr("model.py", template)
            os.replace(temporary, destination)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def model_template(self):
        lines = ["from torch import nn", "", "", "def create_model():",
                 "    return nn.Sequential("]
        for layer in self.layers:
            if layer["type"] == "linear":
                expression = f'nn.Linear({layer["in_features"]}, {layer["out_features"]})'
            elif layer["activation"] == "Softmax":
                expression = "nn.Softmax(dim=1)"
            else:
                expression = f'nn.{layer["activation"]}()'
            lines.append(f"        {expression},")
        lines.extend(["    )", ""])
        return "\n".join(lines)

    @classmethod
    def import_zip(cls, filename, expected_task):
        try:
            with zipfile.ZipFile(filename) as archive:
                for name, limit in (("metadata.json", 1_000_000), ("weights.pt", 256_000_000)):
                    if archive.getinfo(name).file_size > limit:
                        raise ValueError("Model archive is too large.")
                metadata = json.loads(archive.read("metadata.json"))
                if (metadata.get("format") != "weave-sequential"
                        or metadata.get("version") != 1):
                    raise ValueError("Unsupported model archive format.")
                if metadata["task"] != expected_task:
                    raise ValueError("The archive task does not match this node.")
                session = cls(metadata["task"], metadata["layers"])
                session.model = build_sequential(session.layers)
                weights = torch.load(io.BytesIO(archive.read("weights.pt")),
                                     map_location="cpu", weights_only=True)
            session.lr = float(metadata["lr"])
            session.steps = int(metadata["steps"])
            session.epochs = int(metadata["epochs"])
            if (not math.isfinite(session.lr) or session.lr <= 0
                    or not 1 <= session.steps <= 1_000_000 or session.epochs < 0):
                raise ValueError("Invalid training settings in archive.")
            session.classes = metadata["classes"]
            session.feature_columns = metadata["feature_columns"]
            if not isinstance(session.feature_columns, list) or not session.feature_columns:
                raise ValueError("Missing feature schema in archive.")
            for column in session.feature_columns:
                json_scalar(column)
            if expected_task == "classification":
                if not isinstance(session.classes, list) or len(session.classes) < 2:
                    raise ValueError("Missing class labels in archive.")
                for label in session.classes:
                    json_scalar(label)
                if len(set(session.classes)) != len(session.classes):
                    raise ValueError("Duplicate class labels in archive.")
            first = next(layer for layer in session.layers if layer["type"] == "linear")
            last = next(layer for layer in reversed(session.layers) if layer["type"] == "linear")
            outputs = len(session.classes) if expected_task == "classification" else 1
            if (first["in_features"] != len(session.feature_columns)
                    or len(set(session.feature_columns)) != len(session.feature_columns)
                    or last["out_features"] != outputs):
                raise ValueError("Architecture does not match the saved feature/class schema.")
            if expected_task == "classification" and session.layers[-1]["type"] != "linear":
                raise ValueError("CrossEntropy requires a final Linear layer (raw logits).")
            session.model.load_state_dict(weights["model"], strict=True)
            if any(not torch.isfinite(value).all() for value in session.model.state_dict().values()):
                raise ValueError("Model weights are not finite.")
            session.optimizer = torch.optim.Adam(session.model.parameters(), lr=session.lr)
            session.optimizer.load_state_dict(weights["optimizer"])
            for parameter, state in session.optimizer.state.items():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state and (state[key].shape != parameter.shape
                                         or not torch.isfinite(state[key]).all()):
                        raise ValueError("Invalid optimizer state in archive.")
            session.last_metrics = metadata.get("metrics", {})
            if not isinstance(session.last_metrics, dict):
                raise ValueError("Invalid metrics in archive.")
            for split, row in session.last_metrics.items():
                if split not in SPLITS or not isinstance(row, dict):
                    raise ValueError("Invalid metrics in archive.")
                for value in row.values():
                    if value is not None and (not isinstance(value, (int, float))
                                              or not math.isfinite(value)):
                        raise ValueError("Invalid metrics in archive.")
            session.model.eval()
            return session
        except Exception as error:
            raise ValueError(f"Invalid model archive: {error}") from error
