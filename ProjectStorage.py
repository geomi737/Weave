"""Сохранение и чтение самостоятельных проектов SQLite."""

import io
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PyQt6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
)
from qtpynodeeditor import PortType

from FlowNodes import DatasetNode, Model

APPLICATION_ID = 0x57455645
CONTROL_NAMES = (
    "editor",
    "comboBox",
    "dropna_checkbox",
    "test_pct",
    "val_pct",
    "include_validate",
    "min_frequency_spinbox",
    "max_categories_spinbox",
)


def control_value(control):
    if isinstance(control, QComboBox):
        return control.currentText()
    if isinstance(control, QLineEdit):
        return control.text()
    if isinstance(control, QAbstractButton):
        return control.isChecked()
    if isinstance(control, (QSpinBox, QDoubleSpinBox)):
        return control.value()
    raise TypeError("Unsupported saved control")


def serialize_frame(frame):
    document = json.loads(frame.to_json(orient="table", double_precision=15))
    # Представление float в Python сохраняет полную точность float64.
    for column in frame.select_dtypes(include="floating"):
        for row, value in zip(document["data"], frame[column]):
            row[str(column)] = None if pd.isna(value) else float(value)
    return json.dumps(document, separators=(",", ":"))


def capture_project(scene, training_panel):
    """Собирает граф, исходные таблицы и состояния моделей."""
    graph = scene.__getstate__()
    frames, models = {}, {}
    for saved in graph["nodes"]:
        node = scene.nodes[saved["id"]]
        model = node.model
        settings = {
            name: control_value(getattr(model, name))
            for name in CONTROL_NAMES
            if hasattr(model, name)
        }
        if hasattr(model, "drop_group"):
            settings["drop_mode"] = model.drop_group.checkedId()
        if hasattr(model, "random_state"):
            settings["random_state"] = model.random_state
        if hasattr(model, "split_version"):
            settings["split_version"] = model.split_version
        if isinstance(model, Model):
            settings["layers"] = model.get_architecture()
            card = training_panel.cards[node.id]
            training = {
                "learning_rate": card.lr_spinbox.value(),
                "epochs": card.epochs_spinbox.value(),
                "batch_size": card.batch_spinbox.value(),
                "epoch_history": card.epoch_history.copy(),
                "metrics": card.axes_data,
            }
            if hasattr(card, "threshold_spinbox"):
                training["threshold"] = card.threshold_spinbox.value()
            weights = None
            if card.network is not None:
                buffer = io.BytesIO()
                torch.save(card.network.state_dict(), buffer)
                weights = buffer.getvalue()
            models[node.id] = {
                "task": model.task.name,
                "architecture": card._network_architecture,
                "weights": weights,
                "training": training,
            }
        saved["settings"] = settings
        # Производные таблицы получатся заново при пересчёте графа.
        if (
            isinstance(model, DatasetNode)
            and model.dt is not None
            and not node[PortType.input][0].connections
        ):
            frames[node.id] = serialize_frame(model.dt)
    return graph, frames, models


def write_project(path, graph, frames, models):
    """Записывает снимок проекта одной транзакцией SQLite."""
    path = Path(path)
    existed = path.exists()
    try:
        with closing(sqlite3.connect(path)) as database, database:
            application_id = database.execute(
                "PRAGMA application_id"
            ).fetchone()[0]
            tables = database.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if application_id != APPLICATION_ID and (application_id or tables):
                raise ValueError("This file is not a Weave project")
            # Ошибка записи откатит снимок целиком, включая веса и таблицы.
            database.execute("BEGIN IMMEDIATE")
            database.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            database.execute("PRAGMA user_version = 1")
            database.execute(
                "CREATE TABLE IF NOT EXISTS projects "
                "(id INTEGER PRIMARY KEY CHECK(id=1), "
                "graph_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            database.execute(
                "CREATE TABLE IF NOT EXISTS datasets "
                "(node_id TEXT PRIMARY KEY, frame_json TEXT NOT NULL)"
            )
            database.execute(
                "CREATE TABLE IF NOT EXISTS models "
                "(node_id TEXT PRIMARY KEY, task TEXT NOT NULL, "
                "architecture_json TEXT, "
                "weights BLOB, training_json TEXT NOT NULL)"
            )
            database.execute(
                "INSERT INTO projects VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "graph_json=excluded.graph_json, "
                "updated_at=excluded.updated_at",
                (json.dumps(graph), datetime.now(timezone.utc).isoformat()),
            )
            database.execute("DELETE FROM datasets")
            database.executemany(
                "INSERT INTO datasets VALUES (?, ?)", frames.items()
            )
            database.execute("DELETE FROM models")
            database.executemany(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        node_id,
                        item["task"],
                        json.dumps(item["architecture"]),
                        item["weights"],
                        json.dumps(item["training"]),
                    )
                    for node_id, item in models.items()
                ],
            )
    except Exception:
        if not existed:
            path.unlink(missing_ok=True)
        raise


def read_project(path):
    """Читает снимок; восстановлением виджетов занимается загрузчик."""
    # Чтение не должно создавать новый файл по ошибочно указанному пути.
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as database:
        if (
            database.execute("PRAGMA application_id").fetchone()[0]
            != APPLICATION_ID
            or database.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise ValueError("Unsupported Weave project")
        graph = json.loads(
            database.execute(
                "SELECT graph_json FROM projects WHERE id=1"
            ).fetchone()[0]
        )
        frames = {
            node_id: pd.read_json(
                io.StringIO(data), orient="table", precise_float=True
            )
            for node_id, data in database.execute(
                "SELECT node_id, frame_json FROM datasets"
            )
        }
        models = {
            node_id: {
                "task": task,
                "architecture": json.loads(architecture),
                "weights": weights,
                "training": json.loads(training),
            }
            for (
                node_id,
                task,
                architecture,
                weights,
                training,
            ) in database.execute("SELECT * FROM models")
        }
    return graph, frames, models
