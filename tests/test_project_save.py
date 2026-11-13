import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PyQt6.QtCore import QEventLoop, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox
from qtpynodeeditor import PortType

from FlowNodes import ClassificationModel, DatasetNode, SplitColumnNode
from main import MainWindow
from ModelArchive import build_network
from ProjectStorage import capture_project, read_project, write_project


class ProjectSaveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "project.weave"
        self.window = MainWindow()
        self.addCleanup(self.cleanup_window)
        self.source = self.window.flowScene.create_node(DatasetNode)
        self.split = self.window.flowScene.create_node(SplitColumnNode)
        self.model = self.window.flowScene.create_node(ClassificationModel)
        frame = pd.DataFrame(
            {"feature": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]}
        )
        self.source.model.dt = frame
        self.connect(self.source, 0, self.split, 0)
        self.split.model.comboBox.setCurrentText("target")
        self.connect(self.split, 0, self.model, 0)
        self.connect(self.split, 1, self.model, 1)
        self.model.model.replace_architecture(
            [
                {"type": "linear", "in_features": 1, "out_features": 2},
            ]
        )
        self.card = self.window.training_panel.cards[self.model.id]
        self.card.lr_spinbox.setValue(0.002)
        self.card.ensure_network()
        self.card.update_metrics(1, {"Train": {"Loss": 0.5, "Accuracy": 0.75}})
        self.source.position = (25, 50)
        self.app.processEvents()

    def connect(self, source, out_port, target, in_port):
        self.window.flowScene.create_connection(
            source[PortType.output][out_port], target[PortType.input][in_port]
        )

    def cleanup_window(self):
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Discard,
        ):
            self.window.close()
        self.window.flowScene.clear_scene()
        self.app.processEvents()

    def save_as(self, path=None):
        with patch(
            "main.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(path or self.path), ""),
        ):
            return self.window.save_project_as()

    def wait_until(self, predicate):
        if predicate():
            return
        loop = QEventLoop()
        poll, timeout = QTimer(), QTimer()
        timeout.setSingleShot(True)
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        timeout.timeout.connect(loop.quit)
        poll.start(5)
        timeout.start(10000)
        loop.exec()
        poll.stop()
        timeout.stop()
        self.assertTrue(predicate())

    def test_save_as_preserves_graph_data_training_and_weights(
        self,
    ):
        self.assertTrue(self.save_as())
        graph, frames, models = read_project(self.path)
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(len(graph["connections"]), 3)
        source = next(
            item for item in graph["nodes"] if item["id"] == self.source.id
        )
        self.assertEqual(source["position"], {"x": 25.0, "y": 50.0})
        split = next(
            item for item in graph["nodes"] if item["id"] == self.split.id
        )
        self.assertEqual(split["settings"]["comboBox"], "target")
        pd.testing.assert_frame_equal(
            frames[self.source.id], self.source.model.dt
        )
        saved = models[self.model.id]
        self.assertEqual(saved["training"]["learning_rate"], 0.002)
        self.assertEqual(saved["training"]["epoch_history"], [1])
        network = build_network(saved["architecture"])
        network.load_state_dict(
            torch.load(io.BytesIO(saved["weights"]), weights_only=True)
        )
        x = torch.tensor([[1.0], [2.0]])
        self.assertTrue(torch.equal(network(x), self.card.network(x)))
        self.assertFalse(self.window.isWindowModified())
        self.assertIn("Saved:", self.window.statusbar.currentMessage())
        self.assertEqual(self.window.project_path, self.path.resolve())

    def test_save_reuses_path_and_updates_existing_rows(self):
        self.save_as()
        self.card.batch_spinbox.setValue(32)
        self.assertTrue(self.window.isWindowModified())
        with patch("main.QtWidgets.QFileDialog.getSaveFileName") as dialog:
            self.window.actionSave.trigger()
        dialog.assert_not_called()
        _, _, models = read_project(self.path)
        self.assertEqual(models[self.model.id]["training"]["batch_size"], 32)
        self.assertFalse(self.window.isWindowModified())

    def test_save_as_cancel_or_failure_preserves_current_path_and_file(self):
        self.save_as()
        original = self.path.read_bytes()
        self.card.batch_spinbox.setValue(48)
        with patch(
            "main.QtWidgets.QFileDialog.getSaveFileName", return_value=("", "")
        ):
            self.assertFalse(self.window.save_project_as())
        with patch(
            "main.write_project", side_effect=PermissionError("read only")
        ):
            with patch("main.QtWidgets.QMessageBox.warning") as warning:
                self.assertFalse(
                    self.save_as(self.path.with_name("other.weave"))
                )
        warning.assert_called_once()
        self.assertEqual(self.window.project_path, self.path.resolve())
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(self.window.isWindowModified())
        self.assertEqual(self.window.statusbar.currentMessage(), "Save failed")

    def test_database_write_failure_rolls_back_previous_snapshot(self):
        self.save_as()
        graph, frames, models = capture_project(
            self.window.flowScene, self.window.training_panel
        )
        graph["nodes"] = []
        models[self.model.id]["training"]["invalid"] = object()
        with self.assertRaises(TypeError):
            write_project(self.path, graph, frames, models)
        restored_graph, restored_frames, restored_models = read_project(
            self.path
        )
        self.assertEqual(len(restored_graph["nodes"]), 3)
        self.assertIn(self.source.id, restored_frames)
        self.assertNotIn("invalid", restored_models[self.model.id]["training"])

    def test_close_cancel_and_cancelled_save_keep_window_open(self):
        self.window.show()
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Cancel,
        ):
            self.assertFalse(self.window.close())
        self.assertTrue(self.window.isVisible())
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Save,
        ):
            with patch(
                "main.QtWidgets.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ):
                self.assertFalse(self.window.close())
        self.assertTrue(self.window.isVisible())

    def test_close_save_writes_project_before_closing(self):
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Save,
        ):
            with patch(
                "main.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(str(self.path), ""),
            ):
                self.assertTrue(self.window.close())
        self.assertTrue(self.path.exists())
        self.assertEqual(len(read_project(self.path)[0]["nodes"]), 3)

    def test_save_and_save_as_shortcuts(self):
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        with patch(
            "main.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(self.path), ""),
        ):
            QTest.keyClick(
                self.window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier
            )
        self.assertTrue(self.path.exists())
        other = self.path.with_name("copy.weave")
        with patch(
            "main.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(other), ""),
        ):
            QTest.keyClick(
                self.window,
                Qt.Key.Key_S,
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.ShiftModifier,
            )
        self.assertTrue(other.exists())
        self.assertEqual(self.window.project_path, other.resolve())

    def test_paused_training_snapshot_stays_paused_and_close_save_stops_it(
        self,
    ):
        self.card.epochs_spinbox.setValue(100000)
        self.card.start_training()
        self.card.pause_training_btn.click()
        self.wait_until(lambda: self.card._paused)
        thread = self.card.training_thread
        self.assertTrue(self.save_as())
        self.assertTrue(thread.isRunning())
        self.assertTrue(self.card.pause_training_btn.isChecked())
        self.assertTrue(self.card._paused)
        self.card.lr_spinbox.setValue(0.003)
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Save,
        ):
            self.assertTrue(self.window.close())
        self.assertFalse(thread.isRunning())

    def test_running_training_resumes_after_snapshot_and_remains_modified(
        self,
    ):
        self.card.epochs_spinbox.setValue(100000)
        self.card.start_training()
        thread = self.card.training_thread
        self.assertTrue(self.save_as())
        self.assertTrue(thread.isRunning())
        with thread._pause_condition:
            self.assertFalse(thread._paused)
        self.assertTrue(self.window.isWindowModified())
        self.assertIn(
            "training continues", self.window.statusbar.currentMessage()
        )

    def test_new_resets_save_path_and_modified_state(self):
        self.save_as()
        with patch(
            "main.QtWidgets.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.window.actionNew.trigger()
        self.assertIsNone(self.window.project_path)
        self.assertFalse(self.window.isWindowModified())
        self.assertFalse(self.window.flowScene.nodes)

    def test_moving_nodes_and_editing_new_layers_marks_modified(
        self,
    ):
        self.save_as()
        self.source.position = (100, 150)
        self.assertTrue(self.window.isWindowModified())
        self.assertTrue(self.window.save_project())
        self.model.model.add_activation_layer()
        self.app.processEvents()
        self.assertTrue(self.window.isWindowModified())
        self.assertTrue(self.window.save_project())
        self.model.model.layers[-1]["activation_combobox"].setCurrentText(
            "Tanh"
        )
        self.assertTrue(self.window.isWindowModified())

    def test_existing_non_project_file_is_not_overwritten(self):
        self.path.write_bytes(b"not a Weave project")
        with patch("main.QtWidgets.QMessageBox.warning"):
            self.assertFalse(self.save_as())
        self.assertEqual(self.path.read_bytes(), b"not a Weave project")
        self.assertIsNone(self.window.project_path)

    def test_source_float64_values_roundtrip_without_rounding(self):
        self.source.model.dt["precise"] = [
            1.2345678901234567,
            6.56,
            0.41,
            float("nan"),
        ]
        self.assertTrue(self.save_as())
        restored = read_project(self.path)[1][self.source.id]
        pd.testing.assert_frame_equal(
            restored, self.source.model.dt, check_exact=True
        )
