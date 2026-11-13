import threading
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PyQt6.QtCore import QEventLoop, QThread, QTimer, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMessageBox
from sklearn.metrics import f1_score, log_loss
from torch import nn

from FlowTypes import ModelTypes
from Training import TrainingThread
from TrainingUI import TraininCard


def close_window(window):
    with patch(
        "main.QtWidgets.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Discard,
    ):
        window.close()


class RecordingNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2, 3))
        self.calls = []
        self.threads = []

    def forward(self, features):
        self.threads.append(QThread.currentThread())
        self.calls.append(
            (self.training, torch.is_grad_enabled(), len(features))
        )
        return super().forward(features)


class TrainingModel:
    caption = "Classification"
    task = ModelTypes.classification

    def __init__(self):
        self.datasets = []
        self.active = False
        for size in (11, 2, 3):
            self.datasets.extend(
                [
                    pd.DataFrame(np.arange(size * 2).reshape(size, 2) / 10),
                    pd.Series(np.arange(size) % 3),
                ]
            )

    def get_architecture(self):
        return [
            {
                "type": "linear",
                "in_features": 2,
                "out_features": 3
                if self.task == ModelTypes.classification
                else 1,
            }
        ]

    def build_network(self):
        self.network = RecordingNetwork()
        self.initial_weights = self.network[0].weight.detach().clone()
        return self.network

    def set_training_active(self, active):
        self.active = active


class RecordingCard(TraininCard):
    def __init__(self, model):
        super().__init__(model)
        self.update_threads = []

    @pyqtSlot(int, dict)
    def update_metrics(self, epoch, metrics):
        self.update_threads.append(QThread.currentThread())
        super().update_metrics(epoch, metrics)


class TrainingUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(1)
        self.model = TrainingModel()
        self.card = RecordingCard(self.model)
        self.card.batch_spinbox.setValue(4)
        self.card.epochs_spinbox.setValue(3)
        self.addCleanup(self.card.close)

    def wait_for_training(self):
        if self.card.training_thread is None:
            return
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        self.card.training_thread.finished.connect(loop.quit)
        timer.start(10000)
        loop.exec()
        self.assertTrue(timer.isActive(), "Training timed out")
        timer.stop()
        self.assertIsNone(self.card.training_thread)

    def run_training(self):
        with patch("TrainingUI.QMessageBox.warning") as warning:
            self.card.start_training()
            self.wait_for_training()
        warning.assert_not_called()

    def test_all_train_batches_run_each_epoch_with_short_optional_splits(self):
        self.run_training()
        self.assertTrue(self.model.network.threads)
        self.assertTrue(
            all(
                thread != self.app.thread()
                for thread in self.model.network.threads
            )
        )
        self.assertEqual(self.card.update_threads, [self.app.thread()] * 3)
        calls = self.model.network.calls
        self.assertEqual(
            [size for training, grad, size in calls if training],
            [4, 4, 3] * 3,
        )
        self.assertTrue(
            all(not grad for training, grad, _ in calls if not training)
        )
        self.assertEqual(self.card.epoch_history, [1, 2, 3])
        self.assertFalse(
            torch.equal(
                self.model.initial_weights, self.model.network[0].weight
            )
        )
        self.assertFalse(self.model.active)
        self.assertTrue(self.card.control_training_btn.isEnabled())
        for split in self.card.split_names:
            self.assertEqual(len(self.card.axes_data[split]["Accuracy"]), 3)
            self.assertEqual(
                list(self.card.axes_lines[split]["Accuracy"].get_xdata()),
                [1, 2, 3],
            )
        with torch.no_grad():
            for row, split in enumerate(self.card.split_names):
                first = row * 2
                last = first + 2
                x, y = self.model.datasets[first:last]
                logits = self.model.network(
                    torch.tensor(x.to_numpy(), dtype=torch.float32)
                )
                expected_loss = nn.functional.cross_entropy(
                    logits, torch.tensor(y.to_numpy(), dtype=torch.long)
                ).item()
                self.assertAlmostEqual(
                    self.card.axes_data[split]["Loss"][-1],
                    expected_loss,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(self.card.loss_table.item(row, 0).text()),
                    expected_loss,
                    places=6,
                )

    def test_scores_use_probabilities_and_support_absent_classes(self):
        probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        true = np.array([0, 0])
        predicted = probabilities.argmax(axis=1)
        worker = TrainingThread(
            None, {}, ModelTypes.classification, 1, 4, 0.001
        )
        scores = worker.get_classification_scores(
            true, predicted, probabilities
        )
        self.assertAlmostEqual(
            scores["Cross-Entropy"],
            log_loss(true, probabilities, labels=[0, 1, 2]),
        )
        self.assertAlmostEqual(
            scores["F1"],
            f1_score(
                true,
                predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            ),
        )

    def test_new_run_resets_curves_and_handles_missing_optional_splits(self):
        self.run_training()
        self.model.datasets[2:] = [None] * 4
        self.card.epochs_spinbox.setValue(2)
        self.run_training()
        self.assertEqual(self.card.epoch_history, [1, 2])
        self.assertEqual(len(self.card.axes_data["Train"]["Loss"]), 2)
        self.assertTrue(
            np.isnan(self.card.axes_data["Validation"]["Loss"]).all()
        )
        self.assertEqual(self.card.loss_table.item(1, 0).text(), "—")

    def test_rejects_fractional_labels_and_restores_controls(self):
        self.model.datasets[1] = pd.Series([0.5] * 11)
        with patch("TrainingUI.QMessageBox.warning") as warning:
            self.card.start_training()
        warning.assert_called_once()
        self.assertIn("integers", warning.call_args.args[2])
        self.assertTrue(self.card.control_training_btn.isEnabled())
        self.assertFalse(self.model.active)

    def test_evaluation_does_not_change_parameters_or_gradients(self):
        self.card.network = self.model.build_network()
        self.card.network.eval()
        x, y = self.model.datasets[:2]
        weights = [p.detach().clone() for p in self.card.network.parameters()]
        worker = TrainingThread(
            self.card.network, {}, ModelTypes.classification, 1, 4, 0.001
        )
        with torch.no_grad():
            worker.evaluate_split(
                torch.tensor(x.to_numpy(), dtype=torch.float32),
                torch.tensor(y.to_numpy(), dtype=torch.long),
                nn.CrossEntropyLoss(),
                4,
            )
        for before, after in zip(weights, self.card.network.parameters()):
            self.assertTrue(torch.equal(before, after))
            self.assertIsNone(after.grad)

    def test_worker_error_restores_controls_and_reports_in_gui_thread(self):
        self.model.datasets[1] = pd.Series([3] * 11)
        error_threads = []
        with patch(
            "TrainingUI.QMessageBox.warning",
            side_effect=lambda *args: error_threads.append(
                QThread.currentThread()
            ),
        ) as warning:
            self.card.start_training()
            self.wait_for_training()
        warning.assert_called_once()
        self.assertEqual(error_threads, [self.app.thread()])
        self.assertIn("output neurons", warning.call_args.args[2])
        self.assertFalse(self.model.active)
        self.assertTrue(self.card.control_training_btn.isEnabled())

    def test_gui_timer_runs_while_worker_waits_and_duplicate_start_is_ignored(
        self,
    ):
        entered = threading.Event()
        release = threading.Event()
        original_build = self.model.build_network

        def build_network():
            network = original_build()
            original_forward = network.forward

            def forward(features):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("GUI did not release worker")
                return original_forward(features)

            network.forward = forward
            return network

        self.model.build_network = build_network
        self.card.start_training()
        thread = self.card.training_thread
        self.card.start_training()
        self.assertIs(self.card.training_thread, thread)
        self.assertFalse(self.card.control_training_btn.isEnabled())
        self.assertFalse(self.card.batch_spinbox.isEnabled())
        gui_ticks = []
        timer = QTimer()

        def tick():
            if entered.is_set():
                gui_ticks.append(QThread.currentThread())
                release.set()
                timer.stop()

        timer.timeout.connect(tick)
        timer.start(1)
        try:
            self.wait_for_training()
        finally:
            release.set()
            timer.stop()
        self.assertEqual(gui_ticks, [self.app.thread()])

    def test_main_window_close_interrupts_active_training(self):
        from main import MainWindow

        self.card.epochs_spinbox.setValue(100000)
        self.card.start_training()
        thread = self.card.training_thread
        window = MainWindow()
        window.training_panel.cards["test"] = self.card
        close_window(window)
        self.assertFalse(thread.isRunning())
        self.assertTrue(
            thread.isInterruptionRequested() or thread.isFinished()
        )
        self.app.processEvents()
        self.assertIsNone(self.card.training_thread)
        self.assertLess(len(self.card.epoch_history), 100000)

    def test_pause_keeps_weights_and_history_then_resumes_same_run(self):
        checkpoint = threading.Event()
        original_check = TrainingThread.check_interruption

        def check(thread):
            with thread._pause_condition:
                if thread._paused:
                    checkpoint.set()
            original_check(thread)

        with patch.object(TrainingThread, "check_interruption", check):
            self.card.start_training()
            thread = self.card.training_thread
            self.card.pause_training_btn.click()
            self.assertEqual(self.card.pause_training_btn.text(), "Resume")
            loop = QEventLoop()
            timer = QTimer()
            timeout = QTimer()
            timeout.setSingleShot(True)
            timeout.timeout.connect(loop.quit)
            timer.timeout.connect(
                lambda: loop.quit() if checkpoint.is_set() else None
            )
            timer.start(1)
            timeout.start(5000)
            loop.exec()
            timer.stop()
            timeout.stop()
            self.assertTrue(checkpoint.is_set())
            weights = [
                p.detach().clone() for p in self.card.network.parameters()
            ]
            history = self.card.epoch_history.copy()
            QTimer.singleShot(50, loop.quit)
            loop.exec()
            self.assertEqual(self.card.epoch_history, history)
            for before, after in zip(weights, self.card.network.parameters()):
                self.assertTrue(torch.equal(before, after))
            self.card.pause_training_btn.click()
            self.assertIs(self.card.training_thread, thread)
            self.assertEqual(self.card.pause_training_btn.text(), "Pause")
            self.wait_for_training()
        self.assertEqual(self.card.epoch_history, [1, 2, 3])
        self.assertFalse(self.card.pause_training_btn.isEnabled())

    def test_close_while_paused_stops_thread(self):
        self.card.epochs_spinbox.setValue(100000)
        self.card.start_training()
        thread = self.card.training_thread
        self.card.pause_training_btn.click()
        self.card.close()
        self.assertFalse(thread.isRunning())
        self.app.processEvents()
        self.assertIsNone(self.card.training_thread)

    def test_deleting_model_removes_card_and_stops_paused_training(self):
        from FlowNodes import ClassificationModel
        from main import MainWindow

        window = MainWindow()
        self.addCleanup(close_window, window)
        node = window.flowScene.create_node(ClassificationModel)
        node.model.add_linear_layer()
        layer = node.model.layers[0]
        layer["in_spinbox"].setValue(2)
        layer["out_spinbox"].setValue(3)
        self.app.processEvents()
        node.model.datasets = self.model.datasets
        card = window.training_panel.cards[node.id]
        card.epochs_spinbox.setValue(100000)
        card.start_training()
        thread = card.training_thread
        card.pause_training_btn.click()
        window.flowScene.remove_node(node)
        self.assertNotIn(node.id, window.training_panel.cards)
        self.assertFalse(thread.isRunning())
        self.assertIsNone(card.model)
        # A stale reference cannot start training on the deleted node.
        card.start_training()
        self.app.processEvents()

    def test_destroyed_model_widget_detaches_remaining_card(self):
        from PyQt6 import sip

        from FlowNodes import ClassificationModel
        from main import MainWindow

        window = MainWindow()
        self.addCleanup(close_window, window)
        node = window.flowScene.create_node(ClassificationModel)
        card = window.training_panel.cards[node.id]
        sip.delete(node.model.embedded_widget_container)
        self.assertIsNone(card.model)
        self.assertTrue(card._closing)
        card.start_training()
        card.training_finished()
        window.flowScene.remove_node(node)

    def test_regression_still_trains_and_displays_metrics(self):
        self.model.task = ModelTypes.regression
        self.model.build_network = lambda: nn.Sequential(nn.Linear(2, 1))
        self.card = RecordingCard(self.model)
        self.addCleanup(self.card.close)
        self.card.epochs_spinbox.setValue(2)
        self.card.batch_spinbox.setValue(4)
        self.run_training()
        self.assertEqual(self.card.epoch_history, [1, 2])
        for split in self.card.split_names:
            self.assertTrue(
                np.isfinite(self.card.axes_data[split]["MSE"]).all()
            )
            self.assertEqual(len(self.card.axes_data[split]["MSE"]), 2)
        self.assertNotEqual(self.card.loss_table.item(0, 1).text(), "—")
