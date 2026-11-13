import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
import torch
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from FlowNodes import ClassificationModel
from main import MainWindow
from ModelArchive import build_network, load_model, save_model

ARCHITECTURE = [
    {"type": "linear", "in_features": 2, "out_features": 4},
    {"type": "activation", "name": "ReLU"},
    {"type": "linear", "in_features": 4, "out_features": 2},
]


def close_window(window):
    with patch(
        "main.QtWidgets.QMessageBox.question",
        return_value=QMessageBox.StandardButton.Discard,
    ):
        window.close()


class ModelArchiveTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.zip"
        self.network = build_network(ARCHITECTURE)

    def test_roundtrip_preserves_architecture_weights_and_predictions(self):
        save_model(self.path, "classification", ARCHITECTURE, self.network)
        with ZipFile(self.path) as archive:
            self.assertEqual(
                set(archive.namelist()), {"metadata.json", "weights.pt"}
            )
        architecture, network = load_model(self.path, "classification")
        self.assertEqual(architecture, ARCHITECTURE)
        x = torch.randn(5, 2)
        self.assertTrue(torch.equal(self.network(x), network(x)))

    def test_rejects_wrong_extension_task_missing_files_and_bad_weights(self):
        save_model(self.path, "classification", ARCHITECTURE, self.network)
        wrong_extension = self.path.with_suffix(".pt")
        wrong_extension.write_bytes(self.path.read_bytes())
        with self.assertRaises(ValueError):
            load_model(wrong_extension, "classification")
        with self.assertRaises(ValueError):
            load_model(self.path, "regression")
        with ZipFile(self.path, "w") as archive:
            archive.writestr("metadata.json", "{}")
        with self.assertRaises(ValueError):
            load_model(self.path, "classification")
        weights = io.BytesIO()
        torch.save({"0.weight": torch.zeros(1)}, weights)
        with ZipFile(self.path, "w") as archive:
            archive.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "format_version": 1,
                        "task": "classification",
                        "layers": ARCHITECTURE,
                    }
                ),
            )
            archive.writestr("weights.pt", weights.getvalue())
        with self.assertRaises(RuntimeError):
            load_model(self.path, "classification")


class ModelArchiveUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.zip"
        self.window = MainWindow()
        self.addCleanup(self.window.flowScene.clear_scene)
        self.addCleanup(close_window, self.window)
        self.node = self.window.flowScene.create_node(ClassificationModel)
        self.node.model.replace_architecture(ARCHITECTURE)
        self.card = self.window.training_panel.cards[self.node.id]
        self.app.processEvents()
        self.node.model.datasets[:2] = [
            pd.DataFrame(
                {"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]}
            ),
            pd.Series([0, 1, 0, 1]),
        ]
        self.card.ensure_network()

    def wait_until(self, predicate):
        if predicate():
            return
        loop = QEventLoop()
        poll = QTimer()
        timeout = QTimer()
        timeout.setSingleShot(True)
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        timeout.timeout.connect(loop.quit)
        poll.start(5)
        timeout.start(10000)
        loop.exec()
        poll.stop()
        timeout.stop()
        self.assertTrue(predicate(), "Timed out waiting for training state")

    def pause(self):
        self.card.epochs_spinbox.setValue(100000)
        self.card.start_training()
        self.card.pause_training_btn.click()
        self.wait_until(lambda: self.card._paused)
        self.assertTrue(self.card.import_model_btn.isEnabled())
        self.assertTrue(self.card.export_model_btn.isEnabled())
        self.assertTrue(self.card.reset_training_btn.isEnabled())
        self.assertFalse(self.node.model.add_linear_button.isEnabled())

    def import_path(self, path):
        with patch(
            "TrainingUI.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Yes,
        ) as warning:
            with patch(
                "TrainingUI.QFileDialog.getOpenFileName",
                return_value=(str(path), ""),
            ):
                self.card.import_model()
        return warning

    def test_paused_export_saves_current_weights_and_keeps_run(self):
        self.pause()
        thread = self.card.training_thread
        weights = {
            k: v.clone() for k, v in self.card.network.state_dict().items()
        }
        history = self.card.epoch_history.copy()
        with patch(
            "TrainingUI.QFileDialog.getSaveFileName",
            return_value=(str(self.path), ""),
        ):
            self.card.export_model_btn.click()
        _, restored = load_model(self.path, "classification")
        for key, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))
        self.assertIs(self.card.training_thread, thread)
        self.assertTrue(self.card._paused)
        self.assertEqual(self.card.epoch_history, history)

    def test_invalid_import_keeps_paused_run_architecture_and_weights(self):
        self.pause()
        thread = self.card.training_thread
        network = self.card.network
        architecture = self.node.model.get_architecture()
        self.card.update_metrics(7, {"Train": {"Loss": 1.0}})
        self.path.write_bytes(b"not a zip")
        warning = self.import_path(self.path)
        self.assertEqual(warning.call_count, 2)
        self.assertIs(self.card.training_thread, thread)
        self.assertIs(self.card.network, network)
        self.assertEqual(self.node.model.get_architecture(), architecture)
        self.assertEqual(self.card.epoch_history, [7])
        self.assertTrue(self.card._paused)

    def test_import_replaces_paused_run_and_keeps_weights_on_start(
        self,
    ):
        architecture = [
            {"type": "linear", "in_features": 2, "out_features": 2}
        ]
        imported = build_network(architecture)
        with torch.no_grad():
            for parameter in imported.parameters():
                parameter.fill_(0.75)
        save_model(self.path, "classification", architecture, imported)
        self.pause()
        old_thread = self.card.training_thread
        self.card.update_metrics(7, {"Train": {"Loss": 1.0}})
        self.import_path(self.path)
        self.assertFalse(old_thread.isRunning())
        self.assertIsNone(self.card.training_thread)
        self.assertEqual(self.node.model.get_architecture(), architecture)
        self.assertEqual(self.card.epoch_history, [])
        self.assertEqual(self.card.loss_table.item(0, 0).text(), "—")
        self.assertTrue(self.node.model.add_linear_button.isEnabled())
        self.app.processEvents()  # Deliver any queued events from the old run.
        self.assertEqual(self.card.epoch_history, [])
        captured = []

        def inspect_start(worker):
            captured.extend(
                p.detach().clone() for p in worker.network.parameters()
            )

        with patch("Training.TrainingThread.run", inspect_start):
            self.card.start_training()
            self.wait_until(lambda: self.card.training_thread is None)
        for actual, expected in zip(captured, imported.parameters()):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(captured), len(list(imported.parameters())))

    def test_reset_paused_run_reinitializes_weights_and_unlocks_architecture(
        self,
    ):
        self.pause()
        old_thread = self.card.training_thread
        with torch.no_grad():
            for parameter in self.card.network.parameters():
                parameter.fill_(42)
        self.card.update_metrics(7, {"Train": {"Loss": 1.0}})
        self.card.reset_training_btn.click()
        self.assertFalse(old_thread.isRunning())
        self.assertIsNone(self.card.training_thread)
        self.assertEqual(self.card.epoch_history, [])
        self.assertTrue(self.node.model.add_linear_button.isEnabled())
        self.assertFalse(self.card.pause_training_btn.isEnabled())
        self.assertTrue(
            all(not torch.all(p == 42) for p in self.card.network.parameters())
        )
        self.assertEqual(self.node.model.get_architecture(), ARCHITECTURE)
        self.app.processEvents()
        self.assertEqual(self.card.epoch_history, [])

    def test_import_confirmation_and_file_dialog_cancellation_keep_model(self):
        network = self.card.network
        with patch(
            "TrainingUI.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.No,
        ):
            with patch("TrainingUI.QFileDialog.getOpenFileName") as dialog:
                self.card.import_model()
            dialog.assert_not_called()
        self.assertIs(self.card.network, network)
        self.import_path("")
        self.assertIs(self.card.network, network)

    def test_running_actions_disabled_and_wrong_suffix_rejected(
        self,
    ):
        self.card.epochs_spinbox.setValue(100000)
        self.card.start_training()
        self.assertFalse(self.card.import_model_btn.isEnabled())
        self.assertFalse(self.card.export_model_btn.isEnabled())
        self.assertFalse(self.card.reset_training_btn.isEnabled())
        with patch("TrainingUI.QFileDialog.getOpenFileName") as dialog:
            self.card.import_model()
        dialog.assert_not_called()
        self.card.pause_training_btn.click()
        self.wait_until(lambda: self.card._paused)
        self.path.with_suffix(".pt").write_bytes(b"not a zip")
        thread = self.card.training_thread
        warning = self.import_path(self.path.with_suffix(".pt"))
        self.assertEqual(warning.call_count, 2)
        self.assertIs(self.card.training_thread, thread)
