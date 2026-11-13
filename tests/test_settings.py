import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PyQt6.QtCore import QEventLoop, QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from FlowNodes import ClassificationModel
from main import MainWindow
from ModelArchive import load_model, save_model
from Settings import (
    AppSettings,
    SettingsDialog,
    detect_backends,
    device_for_backend,
)


class SettingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "settings.ini"
        self.storage = QSettings(str(self.path), QSettings.Format.IniFormat)

    def settings(self, available):
        with patch("Settings.detect_backends", return_value=available):
            return AppSettings(self.storage)

    def test_detects_cpu_cuda_and_rocm_builds(self):
        with patch("Settings.torch.cuda.is_available", return_value=False):
            self.assertEqual(
                detect_backends(),
                {"CPU": True, "CUDA": False, "ROCm": False},
            )
        with patch("Settings.torch.cuda.is_available", return_value=True):
            with patch("Settings.torch.version.hip", None):
                self.assertTrue(detect_backends()["CUDA"])
                self.assertFalse(detect_backends()["ROCm"])
            with patch("Settings.torch.version.hip", "7.0"):
                self.assertTrue(detect_backends()["ROCm"])
                self.assertFalse(detect_backends()["CUDA"])

    def test_rocm_and_cuda_use_cuda_device_api(self):
        available = {"CPU": True, "CUDA": True, "ROCm": True}
        with patch("Settings.detect_backends", return_value=available):
            self.assertEqual(device_for_backend("CPU"), torch.device("cpu"))
            for backend in ("CUDA", "ROCm"):
                self.assertEqual(
                    device_for_backend(backend), torch.device("cuda:0")
                )
        with patch("Settings.detect_backends", return_value={"CPU": True}):
            with self.assertRaises(ValueError):
                device_for_backend("CUDA")

    def test_unavailable_choices_disabled_and_cancel_keeps_setting(
        self,
    ):
        settings = self.settings({"CPU": True, "CUDA": False, "ROCm": False})
        dialog = SettingsDialog(settings)
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.backend_combo.model().item(0).isEnabled())
        self.assertFalse(dialog.backend_combo.model().item(1).isEnabled())
        self.assertFalse(dialog.backend_combo.model().item(2).isEnabled())
        dialog.reject()
        self.assertFalse(self.storage.contains("compute/backend"))

    def test_accept_persists_choice_and_lost_gpu_falls_back_to_cpu(
        self,
    ):
        settings = self.settings({"CPU": True, "CUDA": True, "ROCm": False})
        with patch(
            "Settings.torch.cuda.get_device_name", return_value="Test GPU"
        ):
            dialog = SettingsDialog(settings)
            self.addCleanup(dialog.close)
            dialog.backend_combo.setCurrentText("CUDA")
            dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        reopened = QSettings(str(self.path), QSettings.Format.IniFormat)
        self.assertEqual(reopened.value("compute/backend"), "CUDA")
        with patch(
            "Settings.detect_backends",
            return_value={"CPU": True, "CUDA": False, "ROCm": False},
        ):
            self.assertEqual(AppSettings(reopened).backend, "CPU")

    def test_settings_apply_to_existing_and_new_cards(self):
        settings = self.settings({"CPU": True, "CUDA": True, "ROCm": False})
        window = MainWindow(settings=settings)
        node = window.flowScene.create_node(ClassificationModel)
        card = window.training_panel.cards[node.id]
        self.assertEqual(card.compute_backend, "CPU")

        def accept_dialog(dialog):
            settings.set_backend("CUDA")
            return QDialog.DialogCode.Accepted

        with patch.object(SettingsDialog, "exec", accept_dialog):
            window.actionSettings.trigger()
        self.assertEqual(card.compute_backend, "CUDA")
        self.assertEqual(card.device_label.text(), "CUDA")
        other = window.flowScene.create_node(ClassificationModel)
        self.assertEqual(
            window.training_panel.cards[other.id].compute_backend, "CUDA"
        )
        self.assertEqual(window.actionSettings.shortcut().toString(), "Ctrl+,")
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Discard,
        ):
            window.close()
        window.flowScene.clear_scene()
        self.app.processEvents()

    def test_changing_selection_keeps_running_workers_device(self):
        from TrainingUI import TraininCard

        model = type("ModelStub", (), {"caption": "Model", "task": None})()
        card = TraininCard(model)
        self.addCleanup(card.close)
        worker = type("Worker", (), {"device": torch.device("cpu")})()
        card.training_thread = worker
        card.set_compute_backend("CUDA")
        self.assertEqual(card.compute_backend, "CUDA")
        self.assertEqual(worker.device, torch.device("cpu"))
        self.assertEqual(card.device_label.text(), "CPU")
        card.training_thread = None


class CUDATrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        torch.set_num_threads(1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
    def test_cuda_training_cpu_export_open_and_cpu_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            settings = AppSettings(
                QSettings(
                    str(directory / "settings.ini"), QSettings.Format.IniFormat
                )
            )
            settings.set_backend("CUDA")
            window = MainWindow(settings=settings)
            try:
                node = window.flowScene.create_node(ClassificationModel)
                node.model.replace_architecture(
                    [
                        {
                            "type": "linear",
                            "in_features": 2,
                            "out_features": 2,
                        }
                    ]
                )
                datasets = [
                    pd.DataFrame({"a": [0.0, 1.0] * 16, "b": [1.0, 0.0] * 16}),
                    pd.Series([0, 1] * 16),
                    None,
                    None,
                    None,
                    None,
                ]
                node.model.datasets = datasets
                card = window.training_panel.cards[node.id]
                card.epochs_spinbox.setValue(2)
                card.batch_spinbox.setValue(8)
                self.app.processEvents()

                def train():
                    loop = QEventLoop()
                    timer = QTimer()
                    timer.setSingleShot(True)
                    timer.timeout.connect(loop.quit)
                    with patch("TrainingUI.QMessageBox.warning") as warning:
                        card.start_training()
                        self.assertIsNotNone(card.training_thread)
                        card.training_thread.finished.connect(loop.quit)
                        timer.start(30000)
                        loop.exec()
                    self.assertTrue(timer.isActive())
                    timer.stop()
                    warning.assert_not_called()
                    self.assertIsNone(card.training_thread)

                train()
                self.assertTrue(
                    all(p.is_cuda for p in card.network.parameters())
                )
                self.assertEqual(card.device_label.text(), "CUDA")
                archive = directory / "gpu.zip"
                save_model(
                    archive,
                    "classification",
                    card._network_architecture,
                    card.network,
                )
                _, imported = load_model(archive, "classification")
                for actual, saved in zip(
                    imported.parameters(), card.network.parameters()
                ):
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertTrue(torch.equal(actual, saved.cpu()))
                project = directory / "gpu.weave"
                self.assertTrue(window.write_project_file(project))
                with patch(
                    "main.QtWidgets.QFileDialog.getOpenFileName",
                    return_value=(str(project), ""),
                ):
                    self.assertTrue(window.open_project())
                card = window.training_panel.cards[node.id]
                card.model.datasets = datasets
                card.set_compute_backend("CPU")
                train()
                self.assertTrue(
                    all(
                        p.device.type == "cpu"
                        for p in card.network.parameters()
                    )
                )
            finally:
                with patch(
                    "main.QtWidgets.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Discard,
                ):
                    window.close()
                window.flowScene.clear_scene()
                self.app.processEvents()
