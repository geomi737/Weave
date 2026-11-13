"""Фоновое обучение, пауза и расчёт метрик нейросети."""

from threading import Condition

import torch
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.utils import gen_batches
from torch import nn, optim

from FlowTypes import ModelTypes


class TrainingInterrupted(Exception):
    """Прерывает обучение на ближайшей границе батча."""


class TrainingThread(QThread):
    """Обучает сеть без обращения к виджетам интерфейса."""

    metrics_ready = pyqtSignal(int, dict)
    failed = pyqtSignal(str)
    pause_changed = pyqtSignal(bool)

    def __init__(
        self,
        network,
        datasets,
        model_type,
        epochs,
        batch_size,
        learning_rate,
        parent=None,
        device="cpu",
    ):
        super().__init__(parent)
        self.network = network
        self.datasets = datasets
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self._pause_condition = Condition()
        self._paused = False
        self._pause_acknowledged = False

    def set_paused(self, paused):
        with self._pause_condition:
            self._paused = paused
            # Пробуждаем ожидающий поток при продолжении обучения.
            self._pause_condition.notify_all()

    def pause_for_snapshot(self):
        """Ждёт фактической паузы перед чтением весов."""
        with self._pause_condition:
            was_paused = self._paused
            self._paused = True
            while self.isRunning() and not self._pause_acknowledged:
                self._pause_condition.wait(timeout=0.1)
            return was_paused

    def check_interruption(self):
        """Проверяет остановку и ждёт продолжения без активного опроса."""
        with self._pause_condition:
            waiting = self._paused
            if waiting:
                # Подтверждаем паузу только после завершения текущего батча.
                self._pause_acknowledged = True
                self._pause_condition.notify_all()
                self.pause_changed.emit(True)
            while self._paused and not self.isInterruptionRequested():
                self._pause_condition.wait(timeout=0.1)
            if waiting:
                self._pause_acknowledged = False
                self.pause_changed.emit(False)
        if self.isInterruptionRequested():
            raise TrainingInterrupted()

    def run(self):
        try:
            # Переносим параметры до создания связанного с ними оптимизатора.
            self.network.to(self.device)
            lossf = (
                nn.CrossEntropyLoss()
                if self.model_type == ModelTypes.classification
                else nn.MSELoss()
            )
            optimizer = optim.AdamW(
                self.network.parameters(), lr=self.learning_rate
            )
            train_features, train_target = self.datasets["Train"]
            for epoch in range(1, self.epochs + 1):
                self.check_interruption()
                self.network.train()
                for batch in gen_batches(len(train_features), self.batch_size):
                    self.check_interruption()
                    # Градиенты накапливаются, поэтому очищаем их перед шагом.
                    optimizer.zero_grad()
                    # Передаём на GPU только текущий батч для экономии памяти.
                    features = train_features[batch].to(self.device)
                    target = train_target[batch].to(self.device)
                    prediction = self.network(features)
                    self.validate_prediction(prediction, target)
                    train_loss = lossf(prediction, target)
                    train_loss.backward()
                    optimizer.step()

                self.network.eval()
                # Проверка качества не должна изменять веса или градиенты.
                with torch.no_grad():
                    metrics = {
                        split: self.evaluate_split(
                            x, y, lossf, self.batch_size
                        )
                        for split, (x, y) in self.datasets.items()
                    }
                self.check_interruption()
                self.metrics_ready.emit(epoch, metrics)
        except TrainingInterrupted:
            pass
        except Exception as error:
            self.failed.emit(str(error))

    def validate_prediction(self, prediction, target):
        if self.model_type == ModelTypes.classification:
            if prediction.ndim != 2 or prediction.shape[1] < 2:
                raise ValueError(
                    "Classification requires at least two output neurons, "
                    "one per class."
                )
            if (target >= prediction.shape[1]).any():
                raise ValueError(
                    "A class label exceeds the number of output neurons."
                )
        elif prediction.shape != target.shape:
            raise ValueError(
                "Regression requires one output neuron per sample."
            )

    def evaluate_split(self, features, target, lossf, batch_size):
        """Считает метрики всей выборки без изменения весов."""
        predictions = []
        total_loss = 0.0
        for batch in gen_batches(len(features), batch_size):
            self.check_interruption()
            batch_target = target[batch].to(self.device)
            prediction = self.network(features[batch].to(self.device))
            self.validate_prediction(prediction, batch_target)
            # Последний батч может быть короче, учитываем его размер.
            total_loss += lossf(prediction, batch_target).item() * len(
                batch_target
            )
            # Метрики sklearn считаются на CPU без хранения графа градиентов.
            predictions.append(prediction.detach().cpu())

        prediction = torch.cat(predictions)
        if self.model_type == ModelTypes.classification:
            # LogLoss нужны вероятности, остальным метрикам номера классов.
            probabilities = prediction.softmax(dim=1).detach().cpu().numpy()
            scores = self.get_classification_scores(
                target.cpu().numpy(),
                prediction.argmax(dim=1).cpu().numpy(),
                probabilities,
            )
        else:
            scores = self.get_regression_scores(
                target.cpu().numpy().ravel(), prediction.cpu().numpy().ravel()
            )
        scores["Loss"] = total_loss / len(features)
        return scores

    def get_classification_scores(self, y_true, y_pred, probabilities):
        labels = list(range(probabilities.shape[1]))
        return {
            "Cross-Entropy": log_loss(y_true, probabilities, labels=labels),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "F1": f1_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
        }

    def get_regression_scores(self, y_true, y_pred):
        metrics = {}
        for metric in ("MSE", "RMSE", "MAE", "R2"):
            match metric:
                case "MSE":
                    metrics["MSE"] = mean_squared_error(y_true, y_pred)
                case "RMSE":
                    metrics["RMSE"] = root_mean_squared_error(y_true, y_pred)
                case "MAE":
                    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
                case "R2":
                    metrics["R2"] = r2_score(y_true, y_pred)

        return metrics
