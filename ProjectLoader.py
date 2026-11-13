"""Подготовка сцены до безопасной замены текущего проекта."""

import io
import math
from graphlib import TopologicalSorter

import torch
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
)
from qtpynodeeditor import PortType

from FlowNodes import DatasetNode, DropNode, Model, TVTSplitNode
from FlowScene import WeaveFlowScene
from ModelArchive import build_network
from ProjectStorage import CONTROL_NAMES
from TrainingUI import TrainingPanel


def set_control(control, value):
    # Программная установка значений не должна вызывать обработчики ввода.
    with QSignalBlocker(control):
        if isinstance(control, QComboBox):
            index = control.findText(value)
            if value and index < 0:
                raise ValueError(
                    f"Saved column or category is missing: {value}"
                )
            control.setCurrentIndex(index)
        elif isinstance(control, QLineEdit):
            control.setText(value)
        elif isinstance(control, QAbstractButton):
            if type(value) is not bool:
                raise ValueError("Invalid checkbox value")
            control.setChecked(value)
        elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
            if (
                not isinstance(value, (int, float))
                or not control.minimum() <= value <= control.maximum()
            ):
                raise ValueError("Saved numeric setting is out of range")
            control.setValue(value)
        else:
            raise ValueError("Unsupported saved control")


def apply_settings(model, settings, *, selectors=False):
    """Восстанавливает параметры и выбор столбцов в два этапа."""
    for name, value in settings.items():
        if name in CONTROL_NAMES:
            control = getattr(model, name)
            if isinstance(control, QComboBox) == selectors:
                set_control(control, value)
    # Список столбцов заполнится только после поступления входных таблиц.
    if not selectors:
        if "drop_mode" in settings:
            button = model.drop_group.button(settings["drop_mode"])
            if button is None:
                raise ValueError("Invalid OneHot drop mode")
            with QSignalBlocker(button):
                button.setChecked(True)
        if isinstance(model, Model):
            model.replace_architecture(settings["layers"])
        if isinstance(model, TVTSplitNode):
            model.split_version = settings.get("split_version", 1)
            model.include_val_changed(model.include_validate.isChecked())
        if isinstance(model, DropNode):
            model.dropna(model.dropna_checkbox.isChecked())
    elif "random_state" in settings:
        model.random_state = settings["random_state"]


def restore_training(card, saved):
    if saved["task"] != card.model_type.name:
        raise ValueError("Model task does not match its node")
    training = saved["training"]
    for name, control in (
        ("learning_rate", card.lr_spinbox),
        ("epochs", card.epochs_spinbox),
        ("batch_size", card.batch_spinbox),
    ):
        set_control(control, training[name])
    if "threshold" in training:
        set_control(card.threshold_spinbox, training["threshold"])
    if saved["weights"] is not None:
        weights = torch.load(
            io.BytesIO(saved["weights"]), map_location="cpu", weights_only=True
        )
        with torch.device("meta"):
            network = build_network(saved["architecture"])
        network.load_state_dict(weights, strict=True, assign=True)
        card.validate_network(network)
        card.network = network
        card._network_architecture = saved["architecture"]

    # Восстанавливаем историю сразу, без повторного расчёта каждой эпохи.
    history = training["epoch_history"]
    if not isinstance(history, list) or any(
        type(epoch) is not int or epoch < 1 for epoch in history
    ):
        raise ValueError("Invalid epoch history")
    card.epoch_history = history.copy()
    for row, split in enumerate(card.split_names):
        for column, label in enumerate(card.loss_headers):
            values = [
                float(value) for value in training["metrics"][split][label]
            ]
            if len(values) != len(history):
                raise ValueError("Metric history length does not match epochs")
            card.axes_data[split][label] = values
            card.axes_lines[split][label].set_data(history, values)
            if values:
                text = "—" if math.isnan(values[-1]) else f"{values[-1]:.6f}"
                card.loss_table.item(row, column).setText(text)
    for axes in card.axes.values():
        axes.relim()
        axes.autoscale_view()
    card.canvas.draw_idle()


def prepare_project(snapshot, registry, style):
    """Собирает отдельную сцену и проверяет данные до её показа."""
    graph, frames, models = snapshot
    records = {item["id"]: item for item in graph["nodes"]}
    if len(records) != len(graph["nodes"]):
        raise ValueError("Duplicate node IDs")
    incoming = {node_id: [] for node_id in records}
    parents = {node_id: set() for node_id in records}
    for connection in graph["connections"]:
        source, target = connection["out_id"], connection["in_id"]
        if source not in records or target not in records:
            raise ValueError("Connection refers to a missing node")
        incoming[target].append(connection)
        parents[target].add(source)
    # Источники обрабатываются раньше нод, использующих их результаты.
    order = list(TopologicalSorter(parents).static_order())
    scene = WeaveFlowScene(registry, style=style)
    panel = TrainingPanel()

    def create_card(node):
        if isinstance(node.model, Model):
            panel.add_node(node)

    scene.node_created.connect(create_card)
    scene.node_deleted.connect(panel.remove_node)
    try:
        for saved in graph["nodes"]:
            node = scene.restore_node(saved)
            # data_updated не должен запускать цепочку во время её сборки.
            node.model.blockSignals(True)
            apply_settings(node.model, saved.get("settings", {}))
        # Связи создаются до подачи исходных таблиц в граф.
        for connection in graph["connections"]:
            scene.restore_connection(connection)
        if set(models) != set(panel.cards):
            raise ValueError("Missing or unexpected model records")
        for node_id, frame in frames.items():
            node = scene.nodes[node_id]
            if not isinstance(node.model, DatasetNode) or incoming[node_id]:
                raise ValueError("Dataset does not belong to a source node")
            node.model.dt = frame
            node.model.load_btn.setText("Unload Dataset")

        outputs = {}
        for node_id in order:
            node = scene.nodes[node_id]
            for connection in sorted(
                incoming[node_id], key=lambda item: item["in_index"]
            ):
                key = (connection["out_id"], connection["out_index"])
                node.model.set_in_data(
                    outputs[key], node[PortType.input][connection["in_index"]]
                )
            apply_settings(
                node.model,
                records[node_id].get("settings", {}),
                selectors=True,
            )
            if isinstance(node.model, TVTSplitNode):
                # Повторяем разбиение с сохранённым seed и версией алгоритма.
                node.model.update_output()
            needed = {
                item["out_index"]
                for item in graph["connections"]
                if item["out_id"] == node_id
            }
            # Готовый результат выхода используется всеми его соединениями.
            for port in needed:
                outputs[node_id, port] = node.model.out_data(port)
            if isinstance(node.model, Model):
                restore_training(panel.cards[node_id], models[node_id])
                # Геометрия обновится ниже один раз, без отложенного сигнала.
                node.model._resize_timer.stop()
            node.on_node_size_updated()
        for node in scene.nodes.values():
            node.model.blockSignals(False)
        scene.node_created.disconnect(create_card)
        return scene, panel
    except Exception:
        # Неудачная загрузка освобождает только подготовленную скрытую сцену.
        panel.shutdown()
        scene.clear_scene()
        panel.deleteLater()
        scene.deleteLater()
        raise
