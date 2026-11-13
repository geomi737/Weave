"""Карточки обучения и обмен данными с рабочими потоками."""

from contextlib import contextmanager
from pathlib import Path

import torch
from matplotlib import style
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from torch import nn

from FlowNodes import Model
from FlowTypes import ModelTypes
from ModelArchive import load_model, save_model
from Settings import device_for_backend
from Training import TrainingThread

style.use("dark_background")


class TraininCard(QGroupBox):
    """Управляет обучением одной ноды и отображает её метрики."""

    project_changed = pyqtSignal()

    def __init__(self, model: Model, parent=None):
        super().__init__(model.caption, parent)

        self.model = model
        self.model_type = model.task
        self.compute_backend = "CPU"
        self.training_thread = None
        self.network = None
        self._network_architecture = None
        self._paused = False
        self._closing = False

        layout = QVBoxLayout(self)
        settings_layout = QFormLayout()

        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setDecimals(8)
        self.lr_spinbox.setRange(0.00000001, 1)
        self.lr_spinbox.setValue(0.0001)

        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 100000)
        self.epochs_spinbox.setValue(100)

        self.batch_spinbox = QSpinBox()
        self.batch_spinbox.setRange(1, 100000)
        self.batch_spinbox.setValue(16)

        settings_layout.addRow("Batch Size", self.batch_spinbox)
        settings_layout.addRow("Learning rate", self.lr_spinbox)
        settings_layout.addRow("Epochs", self.epochs_spinbox)
        self.device_label = QLabel("CPU")
        settings_layout.addRow("Compute device", self.device_label)

        if self.model_type == ModelTypes.classification:
            self.threshold_spinbox = QDoubleSpinBox()
            self.threshold_spinbox.setRange(0, 1)
            self.threshold_spinbox.setValue(0.5)

        layout.addLayout(settings_layout)
        layout.addLayout(self.init_metrics())
        layout.addLayout(self.init_buttons())

    def init_buttons(self):
        self.control_training_btn = QPushButton()
        self.control_training_btn.setText("Start Training")
        self.control_training_btn.clicked.connect(self.start_training)

        self.pause_training_btn = QPushButton("Pause")
        self.pause_training_btn.setCheckable(True)
        self.pause_training_btn.setEnabled(False)
        self.pause_training_btn.toggled.connect(self.pause_training)

        self.import_model_btn = QPushButton()
        self.import_model_btn.setText("Import Model")
        self.import_model_btn.clicked.connect(self.import_model)

        self.export_model_btn = QPushButton()
        self.export_model_btn.setText("Export Model")
        self.export_model_btn.clicked.connect(self.export_model)

        self.reset_training_btn = QPushButton("Reset Training")
        self.reset_training_btn.clicked.connect(self.reset_training)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.control_training_btn)
        buttons_layout.addWidget(self.pause_training_btn)
        buttons_layout.addWidget(self.import_model_btn)
        buttons_layout.addWidget(self.export_model_btn)
        buttons_layout.addWidget(self.reset_training_btn)

        return buttons_layout

    def init_metrics(self):
        metrics_layout = QVBoxLayout()

        self.regression_metrics = ["MSE", "RMSE", "MAE", "R2"]
        self.classification_metrics = [
            "Cross-Entropy",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ]

        self.split_names = ("Train", "Validation", "Test")
        self.loss_headers = ["Loss"]

        if self.model_type == ModelTypes.regression:
            self.loss_headers.extend(self.regression_metrics)
        elif self.model_type == ModelTypes.classification:
            self.loss_headers.extend(self.classification_metrics)

        self.loss_table = QTableWidget(
            len(self.split_names), len(self.loss_headers)
        )
        self.loss_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.loss_table.setHorizontalHeaderLabels(self.loss_headers)
        self.loss_table.setVerticalHeaderLabels(self.split_names)
        self.loss_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        for row in range(len(self.split_names)):
            for column in range(len(self.loss_headers)):
                self.loss_table.setItem(row, column, QTableWidgetItem("—"))

        table_height = (
            self.loss_table.horizontalHeader().sizeHint().height()
            + self.loss_table.verticalHeader().length()
            + 2 * self.loss_table.frameWidth()
        )
        self.loss_table.setFixedHeight(table_height)

        metrics_layout.addWidget(self.loss_table)

        columns = 2
        rows = (len(self.loss_headers) + columns - 1) // columns

        self.figure = Figure(
            figsize=(5, 3),
            layout="constrained",
            facecolor="#1b1e27",
        )
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(rows * 250)

        self.axes = {}
        self.axes_data = {
            x: {y: [] for y in self.loss_headers} for x in self.split_names
        }
        self.axes_lines = {split: {} for split in self.split_names}

        for index, label in enumerate(self.loss_headers, start=1):
            axes = self.figure.add_subplot(rows, columns, index)
            axes.set_xlabel("Epoch")
            axes.set_ylabel(label)
            axes.set_facecolor("#1b1e27")
            axes.grid(True, alpha=0.3)
            self.axes[label] = axes

            for split in self.split_names:
                (line,) = self.axes[label].plot([], [], label=split)
                self.axes_lines[split][label] = line

            self.axes[label].legend(
                loc="upper right", ncol=3, frameon=False, fontsize=8
            )

        self.epoch_history = []

        metrics_layout.addWidget(self.canvas)

        return metrics_layout

    @pyqtSlot(int, dict)
    def update_metrics(self, epoch, metrics):
        if self._closing:
            return
        # Игнорируем сигналы потока, оставшиеся после замены модели.
        if (
            isinstance(self.sender(), TrainingThread)
            and self.sender() is not self.training_thread
        ):
            return
        self.epoch_history.append(epoch)
        self.project_changed.emit()

        for column, label in enumerate(self.loss_headers):
            for row, split in enumerate(self.split_names):
                value = metrics.get(split, {}).get(label)

                text = "—" if value is None else f"{value:.6f}"
                self.loss_table.item(row, column).setText(text)

                # NaN оставляет разрыв линии для отсутствующей метрики.
                self.axes_data[split][label].append(
                    float("nan") if value is None else value
                )

                self.axes_lines[split][label].set_data(
                    self.epoch_history,
                    self.axes_data[split][label],
                )

            self.axes[label].relim()
            self.axes[label].autoscale_view()

        self.canvas.draw_idle()

    def start_training(self):
        if self.training_thread is not None or self._closing:
            return
        try:
            self.set_training_active(True)
            datasets = {}
            for index, split in enumerate(self.split_names):
                first = index * 2
                last = first + 2
                features, target = self.model.datasets[first:last]
                if features is None and target is None and split != "Train":
                    continue
                if features is None or target is None:
                    raise ValueError(
                        f"{split}: connect both features and target."
                    )
                if len(features) == 0 or len(features) != len(target):
                    raise ValueError(
                        f"{split}: features and target must have equal, "
                        "nonzero lengths."
                    )
                if not features.index.equals(target.index):
                    raise ValueError(
                        f"{split}: features and target indexes do not match."
                    )
                if list(features.columns) != list(
                    self.model.datasets[0].columns
                ):
                    raise ValueError(
                        f"{split}: feature columns must match Train "
                        "in the same order."
                    )

                x = torch.tensor(features.to_numpy(), dtype=torch.float32)
                y = torch.tensor(target.to_numpy(), dtype=torch.float64)
                if (
                    y.ndim != 1
                    or not torch.isfinite(x).all()
                    or not torch.isfinite(y).all()
                ):
                    raise ValueError(
                        f"{split}: use finite numeric features "
                        "and a one-dimensional target."
                    )
                if self.model_type == ModelTypes.classification:
                    if (y < 0).any() or (y != y.round()).any():
                        raise ValueError(
                            f"{split}: class labels must be integers "
                            "starting at zero."
                        )
                    y = y.long()
                else:
                    y = y.float().unsqueeze(1)
                datasets[split] = (x, y)

            # Виджеты читаются здесь; рабочий поток получает готовые значения.
            device = device_for_backend(self.compute_backend)
            self.ensure_network()
            self.clear_history()
            self.device_label.setText(self.compute_backend)
            self.training_thread = TrainingThread(
                self.network,
                datasets,
                self.model_type,
                self.epochs_spinbox.value(),
                self.batch_spinbox.value(),
                self.lr_spinbox.value(),
                parent=self,
                device=device,
            )
            self.training_thread.metrics_ready.connect(self.update_metrics)
            self.training_thread.failed.connect(self.show_training_error)
            self.training_thread.pause_changed.connect(self.pause_changed)
            self.training_thread.finished.connect(self.training_finished)
            self.pause_training_btn.setEnabled(True)
            self.update_model_actions()
            self.training_thread.start()
            self.project_changed.emit()

        except (ValueError, TypeError, RuntimeError, IndexError) as error:
            self.set_training_active(False)
            self.show_training_error(str(error))

    def set_compute_backend(self, backend):
        self.compute_backend = backend
        if self.training_thread is None:
            self.device_label.setText(backend)

    def set_training_active(self, active):
        for widget in (
            self.control_training_btn,
            self.lr_spinbox,
            self.epochs_spinbox,
            self.batch_spinbox,
        ):
            widget.setEnabled(not active and not self._closing)
        if self.model is not None:
            self.model.set_training_active(active)
        self.update_model_actions()

    def update_model_actions(self):
        available = not self._closing and (
            self.training_thread is None or self._paused
        )
        for button in (
            self.import_model_btn,
            self.export_model_btn,
            self.reset_training_btn,
        ):
            button.setEnabled(available)

    @pyqtSlot(bool)
    def pause_changed(self, paused):
        # Импорт и экспорт открываются после фактической остановки расчёта.
        if self.sender() is not self.training_thread:
            return
        self._paused = paused
        self.pause_training_btn.setEnabled(not self._closing)
        self.update_model_actions()

    @pyqtSlot(bool)
    def pause_training(self, paused):
        if self.training_thread is not None:
            self._paused = False
            self.update_model_actions()
            self.pause_training_btn.setEnabled(False)
            self.training_thread.set_paused(paused)
            if not paused:
                self.project_changed.emit()
        self.pause_training_btn.setText("Resume" if paused else "Pause")

    @pyqtSlot(str)
    def show_training_error(self, message):
        if (
            isinstance(self.sender(), TrainingThread)
            and self.sender() is not self.training_thread
        ):
            return
        if not self._closing:
            QMessageBox.warning(self, "Training error", message)

    @pyqtSlot()
    def training_finished(self):
        if (
            isinstance(self.sender(), TrainingThread)
            and self.sender() is not self.training_thread
        ):
            return
        thread = self.training_thread
        self.training_thread = None
        self._paused = False
        self.pause_training_btn.setChecked(False)
        self.pause_training_btn.setEnabled(False)
        self.set_training_active(False)
        if thread is not None:
            thread.deleteLater()

    def clear_history(self):
        self.epoch_history.clear()
        self.project_changed.emit()
        for split in self.split_names:
            for label in self.loss_headers:
                self.axes_data[split][label].clear()
                self.axes_lines[split][label].set_data([], [])
        for row in range(self.loss_table.rowCount()):
            for column in range(self.loss_table.columnCount()):
                self.loss_table.item(row, column).setText("—")
        self.canvas.draw_idle()

    def ensure_network(self):
        """Сохраняет веса, если архитектура сети не менялась."""
        architecture = self.model.get_architecture()
        if self.network is None or architecture != self._network_architecture:
            network = self.model.build_network()
            self.validate_network(network)
            self.network = network
            self._network_architecture = architecture

    def validate_network(self, network):
        if self.model_type == ModelTypes.classification:
            if isinstance(network[-1], (nn.Softmax, nn.Sigmoid)):
                raise ValueError(
                    "CrossEntropyLoss requires raw logits: "
                    "remove the final Softmax/Sigmoid."
                )

    def can_manage_model(self):
        return not self._closing and (
            self.training_thread is None or self._paused
        )

    def discard_training(self):
        self.stop_training()
        self.training_finished()
        self.clear_history()

    def import_model(self):
        if not self.can_manage_model():
            return
        answer = QMessageBox.warning(
            self,
            "Import model",
            "The current model and training progress will be "
            "reset and replaced "
            "if the import succeeds. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import model", "", "Model archives (*.zip)"
        )
        if not path:
            return
        try:
            architecture, network = load_model(path, self.model_type.name)
            self.validate_network(network)
        except Exception as error:
            QMessageBox.warning(self, "Invalid model format", str(error))
            return
        # Прежний запуск сбрасывается лишь после полной проверки импорта.
        self.discard_training()
        self.model.replace_architecture(architecture)
        self.network = network
        self._network_architecture = architecture
        self.project_changed.emit()

    def export_model(self):
        if not self.can_manage_model():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export model", "model.zip", "Model archives (*.zip)"
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".zip"
        try:
            self.ensure_network()
            save_model(
                path,
                self.model_type.name,
                self._network_architecture,
                self.network,
            )
        except Exception as error:
            QMessageBox.warning(self, "Export error", str(error))

    def reset_training(self):
        if not self.can_manage_model():
            return
        try:
            network = self.model.build_network()
            self.validate_network(network)
        except (ValueError, TypeError, RuntimeError) as error:
            self.show_training_error(str(error))
            return
        self.discard_training()
        self.network = network
        self._network_architecture = self.model.get_architecture()

    def stop_training(self):
        if self.training_thread is not None:
            self.training_thread.requestInterruption()
            self.training_thread.wait()

    @pyqtSlot()
    def detach_model(self):
        # Qt-виджеты ноды к этому моменту уже могут быть уничтожены.
        self.model = None
        self.close()

    def closeEvent(self, event):
        self._closing = True
        self.stop_training()
        super().closeEvent(event)


class TrainingPanel(QScrollArea):
    """Связывает карточки обучения с жизненным циклом нод."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWidgetResizable(True)

        container = QWidget()
        self.cards_layout = QVBoxLayout(container)
        self.cards_layout.addStretch()

        self.setWidget(container)

        self.cards = {}

    def add_node(self, node):
        card = TraininCard(node.model)
        node.model.embedded_widget_container.destroyed.connect(
            card.detach_model
        )

        for widget in (
            card.lr_spinbox,
            card.epochs_spinbox,
            card.canvas,
        ):
            widget.installEventFilter(self)

        self.cards_layout.insertWidget(
            self.cards_layout.count() - 1,
            card,
        )

        self.cards[node.id] = card

        return card

    def remove_node(self, node):
        card = self.cards.pop(node.id, None)
        if card is not None:
            card.detach_model()
            self.cards_layout.removeWidget(card)
            card.deleteLater()

    @contextmanager
    def paused_for_snapshot(self):
        """Приостанавливает потоки на время снимка проекта."""
        paused = []
        try:
            for card in self.cards.values():
                thread = card.training_thread
                if thread is not None:
                    paused.append((card, thread, thread.pause_for_snapshot()))
            yield
        finally:
            # Пользовательская пауза после сохранения должна остаться паузой.
            for card, thread, was_paused in paused:
                if not card._closing and not was_paused and thread.isRunning():
                    thread.set_paused(False)

    def shutdown(self):
        # Сначала просим остановиться все потоки, затем ждём их завершения.
        for card in self.cards.values():
            card._closing = True
            if card.training_thread is not None:
                card.training_thread.requestInterruption()
        for card in self.cards.values():
            card.stop_training()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and isinstance(
            watched,
            (QSpinBox, QDoubleSpinBox, FigureCanvasQTAgg),
        ):
            QApplication.sendEvent(self.viewport(), event)
            return True

        return super().eventFilter(watched, event)
