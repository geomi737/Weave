"""Training cards and cooperative background workers for model nodes."""

import time

import torch
from PyQt6.QtCore import QPointF, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QScrollArea, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from Training import SPLITS, TrainingSession


class TrainingWorker(QThread):
    progress = pyqtSignal(int, object)
    outcome = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, session, inputs, lr, steps, parent=None):
        super().__init__(parent)
        self.session = session
        self.inputs = inputs
        self.lr = lr
        self.steps = steps

    def run(self):
        try:
            datasets = self.session.prepare(self.inputs, self.lr)
            self.session.steps = self.steps
            last_update = 0.0
            for step in range(self.steps):
                completed = self.session.train_epoch(datasets, self.isInterruptionRequested)
                metrics = self.session.evaluate(datasets)
                now = time.monotonic()
                if (now - last_update >= 0.1 or step == self.steps - 1
                        or not completed or self.isInterruptionRequested()):
                    self.progress.emit(self.session.epochs, metrics)
                    last_update = now
                if not completed or self.isInterruptionRequested():
                    self.outcome.emit("Paused")
                    return
            self.outcome.emit("Completed")
        except Exception as error:
            self.failed.emit(str(error))


class LossPlot(QWidget):
    colors = {"Train": "#ad96f5", "Validation": "#72d4a1", "Test": "#efad65"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.points = []

    def add_metrics(self, epoch, metrics):
        self.points.append((epoch, {key: value["Loss"] for key, value in metrics.items()}))
        self.points = self.points[-1000:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1b1e27"))
        painter.setPen(QColor("#e5e7ef"))
        painter.drawText(12, 20, "Loss / Epoch")
        for i, (split, color) in enumerate(self.colors.items()):
            painter.setPen(QColor(color))
            painter.drawText(140 + i * 95, 20, split)
        if not self.points:
            return
        left, top = 65, 35
        width, height = max(1, self.width() - 85), max(1, self.height() - 65)
        max_loss = max(loss for _, row in self.points for loss in row.values()) or 1.0
        min_epoch = self.points[0][0]
        span = max(1, self.points[-1][0] - min_epoch)
        painter.setPen(QColor("#9ba4bb"))
        painter.drawText(4, top + 10, f"{max_loss:.3g}")
        painter.drawText(35, top + height, "0")
        painter.drawText(left, self.height() - 8, str(min_epoch))
        painter.drawText(left + width - 35, self.height() - 8, str(self.points[-1][0]))
        painter.drawLine(left, top, left, top + height)
        painter.drawLine(left, top + height, left + width, top + height)
        for split, color in self.colors.items():
            painter.setPen(QPen(QColor(color), 2))
            previous = None
            for epoch, row in self.points:
                if split not in row:
                    continue
                point = QPointF(left + (epoch - min_epoch) / span * width,
                                top + (1 - row[split] / max_loss) * height)
                if previous is not None:
                    painter.drawLine(previous, point)
                else:
                    painter.drawEllipse(point, 2, 2)
                previous = point


class TrainingCard(QGroupBox):
    idle = pyqtSignal()
    show_node_requested = pyqtSignal()

    def __init__(self, model, parent=None):
        super().__init__(model.caption, parent)
        self.model_node = model
        self.session = None
        self.worker = None
        self.removed = False
        self._outcome = "Ready"
        layout = QVBoxLayout(self)
        settings = QFormLayout()
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setDecimals(8)
        self.lr_spinbox.setRange(0.00000001, 100)
        self.lr_spinbox.setSingleStep(0.0001)
        self.lr_spinbox.setValue(0.001)
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, 1_000_000)
        self.steps_spinbox.setValue(100)
        settings.addRow("Learning rate", self.lr_spinbox)
        settings.addRow("Steps (epochs per run)", self.steps_spinbox)
        layout.addLayout(settings)
        self.status_label = QLabel("Connect Train Features and Train Target.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.epoch_label = QLabel("Completed epochs: 0")
        layout.addWidget(self.epoch_label)
        buttons = QHBoxLayout()
        self.continue_button = QPushButton("Train")
        self.pause_button = QPushButton("Pause")
        self.export_button = QPushButton("Export model")
        self.show_node_button = QPushButton("Show node")
        for button in (self.continue_button, self.pause_button,
                       self.export_button, self.show_node_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.continue_button.clicked.connect(self.start)
        self.pause_button.clicked.connect(self.pause)
        self.export_button.clicked.connect(self.export_model)
        self.show_node_button.clicked.connect(lambda: self.show_node_requested.emit())
        self.plot = LossPlot()
        layout.addWidget(self.plot)
        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setMinimumHeight(145)
        self.metrics_table.setMaximumHeight(170)
        layout.addWidget(self.metrics_table)
        if model.task == "classification":
            layout.addWidget(QLabel("Loss: CrossEntropy. Precision / Recall / F1: macro average."))
        else:
            layout.addWidget(QLabel("Loss: MSE. R2 requires at least two samples."))
        layout.addWidget(QLabel("CPU · Adam · Batch size: 256 · Metrics use all rows of each split."))
        model.inputs_changed.connect(self.update_controls)
        model.architecture_changed.connect(self.architecture_changed)
        model.model_imported.connect(self.load_session)
        self.display_metrics(0, {})
        self.update_controls()

    def update_controls(self):
        active = self.worker is not None
        ready = not self.removed and self.model_node.ready_to_train()
        self.continue_button.setEnabled(ready and not active)
        has_model = self.session is not None and self.session.model is not None
        self.continue_button.setText("Continue training" if has_model else "Train")
        self.pause_button.setEnabled(active)
        self.export_button.setEnabled(not active and self.session is not None
                                      and self.session.model is not None)
        self.lr_spinbox.setEnabled(not active)
        self.steps_spinbox.setEnabled(not active)
        self.show_node_button.setEnabled(not self.removed)
        if not active and not ready and not self.removed:
            self.status_label.setText("Connect the Train pair and complete each optional pair.")

    def architecture_changed(self):
        if self.worker is not None or self.session is None:
            return
        if self.model_node.architecture() != self.session.layers:
            self.session = None
            self.plot.points.clear()
            self.plot.update()
            self.display_metrics(0, {})
            self.status_label.setText("Architecture changed. The next run will build a new model.")
            self.update_controls()

    def load_session(self, session):
        self.session = session
        self.lr_spinbox.setValue(session.lr)
        self.steps_spinbox.setValue(session.steps)
        self.plot.points.clear()
        self.plot.update()
        self.display_metrics(session.epochs, session.last_metrics)
        self.status_label.setText("Imported. Connect matching datasets to continue training.")
        self.update_controls()

    def start(self):
        if self.worker is not None or self.removed or not self.model_node.ready_to_train():
            return
        try:
            if self.session is None:
                self.session = TrainingSession(self.model_node.task, self.model_node.architecture())
            inputs = [value.copy(deep=True) if value is not None else None
                      for value in self.model_node.datasets]
            self.worker = TrainingWorker(self.session, inputs, self.lr_spinbox.value(),
                                         self.steps_spinbox.value(), self)
            self.worker.progress.connect(self.display_metrics)
            self.worker.outcome.connect(self.set_outcome)
            self.worker.failed.connect(self.show_error)
            self.worker.finished.connect(self.worker_finished)
            self._outcome = "Completed"
            self.status_label.setText("Training…")
            self.model_node.set_training_active(True)
            self.update_controls()
            self.worker.start()
        except Exception as error:
            self.worker = None
            self.model_node.set_training_active(False)
            self.status_label.setText(str(error))
            self.update_controls()

    def set_outcome(self, outcome):
        self._outcome = outcome

    def show_error(self, message):
        self._outcome = f"Error: {message}"

    def worker_finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        self.model_node.set_training_active(False)
        self.status_label.setText(self._outcome + (" · Node removed" if self.removed else ""))
        self.update_controls()
        self.idle.emit()

    def pause(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status_label.setText("Pausing after the current batch…")
            self.pause_button.setEnabled(False)

    def display_metrics(self, epochs, metrics):
        self.epoch_label.setText(f"Completed epochs: {epochs}")
        columns = (["Loss", "Accuracy", "Precision", "Recall", "F1"]
                   if self.model_node.task == "classification" else ["Loss", "RMSE", "MAE", "R2"])
        self.metrics_table.setColumnCount(len(columns))
        self.metrics_table.setRowCount(3)
        self.metrics_table.setHorizontalHeaderLabels(columns)
        self.metrics_table.setVerticalHeaderLabels(SPLITS)
        for row, split in enumerate(SPLITS):
            for column, metric in enumerate(columns):
                value = metrics.get(split, {}).get(metric)
                self.metrics_table.setItem(row, column, QTableWidgetItem(
                    "—" if value is None else f"{value:.6g}",
                ))
        if metrics:
            self.plot.add_metrics(epochs, metrics)

    def export_model(self):
        if self.worker is not None or self.session is None or self.session.model is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export model", f"{self.model_node.caption}.zip", "Model archives (*.zip)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".zip"):
            filename += ".zip"
        try:
            self.session.lr = self.lr_spinbox.value()
            self.session.steps = self.steps_spinbox.value()
            self.session.export_zip(filename)
            self.status_label.setText("Model exported.")
        except Exception as error:
            self.status_label.setText(f"Export failed: {error}")

    def node_removed(self):
        self.removed = True
        self.model_node.detached = True
        self.pause()
        self.status_label.setText("Node removed. The trained model can still be exported.")
        self.update_controls()


class TrainingPanel(QScrollArea):
    all_idle = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # CPU training leaves resources available for Qt and other applications.
        torch.set_num_threads(1)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        content.setObjectName("trainingContent")
        self.cards_layout = QVBoxLayout(content)
        self.empty_label = QLabel("Add a Regression Model or Classification Model in Node Edit.")
        self.cards_layout.addWidget(self.empty_label)
        self.cards_layout.addStretch()
        self.setWidget(content)
        self.cards = {}
        self.counts = {"regression": 0, "classification": 0}

    def add_node(self, node):
        model = node.model
        self.counts[model.task] += 1
        model.caption = f"{model.name} {self.counts[model.task]}"
        model.embedded_widget_size_updated.emit()
        card = TrainingCard(model)
        self.cards[node.id] = card
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        card.idle.connect(self.check_idle)
        self.empty_label.hide()
        return card

    def remove_node(self, node):
        card = self.cards.get(node.id)
        if card:
            card.node_removed()

    def running(self):
        return any(card.worker is not None for card in self.cards.values())

    def pause_all(self):
        for card in self.cards.values():
            card.pause()

    def check_idle(self):
        if not self.running():
            self.all_idle.emit()
