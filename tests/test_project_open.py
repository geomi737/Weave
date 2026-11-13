import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox
from qtpynodeeditor import PortType

from FlowNodes import (
    BinaryReplaceNode,
    ClassificationModel,
    DatasetNode,
    OneHotNode,
    SplitColumnNode,
    TVTSplitNode,
)
from main import MainWindow


class ProjectOpenTest(unittest.TestCase):
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
        scene = self.window.flowScene
        self.source = scene.create_node(DatasetNode)
        self.split = scene.create_node(SplitColumnNode)
        self.binary = scene.create_node(BinaryReplaceNode)
        self.tvt = scene.create_node(TVTSplitNode)
        self.ohe = scene.create_node(OneHotNode)
        self.model = scene.create_node(ClassificationModel)
        self.source.model.dt = pd.DataFrame(
            {
                "Agency": ["A", "B"] * 12,
                "Sales": [float(i) for i in range(24)],
                "Claim": ["No", "Yes"] * 12,
            }
        )
        self.connect(self.source, 0, self.split, 0)
        self.split.model.comboBox.setCurrentText("Claim")
        self.connect(self.split, 1, self.binary, 0)
        self.binary.model.comboBox.setCurrentText("Yes")
        self.connect(self.split, 0, self.tvt, 0)
        self.connect(self.binary, 0, self.tvt, 1)
        self.tvt.model.include_validate.setChecked(True)
        self.tvt.model.test_pct.setValue(0.25)
        self.tvt.model.val_pct.setValue(0.2)
        self.tvt.model.update_output()
        for index in range(3):
            self.connect(self.tvt, index * 2, self.ohe, index)
            self.connect(self.ohe, index, self.model, index * 2)
            self.connect(self.tvt, index * 2 + 1, self.model, index * 2 + 1)
        self.ohe.model.drop_group.button(1).click()
        width = self.model.model.datasets[0].shape[1]
        self.model.model.replace_architecture(
            [
                {"type": "linear", "in_features": width, "out_features": 4},
                {"type": "activation", "name": "ReLU"},
                {"type": "linear", "in_features": 4, "out_features": 2},
            ]
        )
        self.card = self.window.training_panel.cards[self.model.id]
        self.card.lr_spinbox.setValue(0.003)
        self.card.ensure_network()
        self.card.update_metrics(1, {"Train": {"Loss": 1.2, "Accuracy": 0.5}})
        self.card.update_metrics(2, {"Train": {"Loss": 0.8, "Accuracy": 0.75}})
        self.source.position = (150, 200)
        self.app.processEvents()
        self.assertTrue(self.window.write_project_file(self.path))

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

    def open_path(self, path=None):
        with patch(
            "main.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(str(path or self.path), ""),
        ):
            return self.window.open_project()

    def test_roundtrip_restores_complete_clean_project(
        self,
    ):
        previous = self.window.flowScene
        seed = self.tvt.model.random_state
        expected = [item.copy() for item in self.model.model.datasets]
        with torch.no_grad():
            prediction = self.card.network(
                torch.tensor(expected[0].to_numpy(), dtype=torch.float32)
            )
        with patch("main.QtWidgets.QMessageBox.warning") as warning:
            self.assertTrue(self.open_path())
        warning.assert_not_called()
        self.assertIsNot(self.window.flowScene, previous)
        nodes = self.window.flowScene.nodes
        self.assertEqual(len(nodes), 6)
        self.assertEqual(
            nodes[self.split.id].model.comboBox.currentText(), "Claim"
        )
        self.assertEqual(
            nodes[self.binary.id].model.comboBox.currentText(), "Yes"
        )
        self.assertTrue(nodes[self.tvt.id].model.include_validate.isChecked())
        self.assertEqual(nodes[self.tvt.id].model.random_state, seed)
        self.assertEqual(nodes[self.ohe.id].model.drop_group.checkedId(), 1)
        pd.testing.assert_frame_equal(
            nodes[self.source.id].model.dt, self.source.model.dt
        )
        for actual, original in zip(
            nodes[self.model.id].model.datasets, expected
        ):
            if isinstance(actual, pd.Series):
                pd.testing.assert_series_equal(actual, original)
            else:
                pd.testing.assert_frame_equal(actual, original)
        card = self.window.training_panel.cards[self.model.id]
        self.assertEqual(card.epoch_history, [1, 2])
        self.assertEqual(card.lr_spinbox.value(), 0.003)
        self.assertEqual(card.loss_table.item(0, 0).text(), "0.800000")
        with torch.no_grad():
            actual = card.network(
                torch.tensor(expected[0].to_numpy(), dtype=torch.float32)
            )
        self.assertTrue(torch.equal(actual, prediction))
        self.app.processEvents()
        self.assertFalse(self.window.isWindowModified())
        self.assertIn("Opened:", self.window.statusbar.currentMessage())
        self.assertTrue(self.window.save_project())

    def test_corrupt_weights_or_graph_preserve_current_project(self):
        previous = self.window.flowScene
        for corruption in ("weights", "graph"):
            with self.subTest(corruption=corruption):
                bad = self.path.with_name(corruption + ".weave")
                bad.write_bytes(self.path.read_bytes())
                with closing(sqlite3.connect(bad)) as database, database:
                    if corruption == "weights":
                        database.execute(
                            "UPDATE models SET weights=?",
                            (b"bad pytorch data",),
                        )
                    else:
                        graph = json.loads(
                            database.execute(
                                "SELECT graph_json FROM projects"
                            ).fetchone()[0]
                        )
                        graph["connections"][0]["out_id"] = "missing-node"
                        database.execute(
                            "UPDATE projects SET graph_json=?",
                            (json.dumps(graph),),
                        )
                with patch("main.QtWidgets.QMessageBox.warning") as warning:
                    self.assertFalse(self.open_path(bad))
                warning.assert_called_once()
                self.assertIs(self.window.flowScene, previous)
                self.assertEqual(self.window.project_path, self.path)
                self.app.processEvents()

    def test_cancel_or_failed_save_keeps_current_project(self):
        self.card.lr_spinbox.setValue(0.004)
        scene = self.window.flowScene
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Cancel,
        ):
            self.assertFalse(self.open_path())
        self.assertIs(self.window.flowScene, scene)
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Save,
        ):
            with patch.object(self.window, "save_project", return_value=False):
                self.assertFalse(self.open_path())
        self.assertIs(self.window.flowScene, scene)
        self.assertEqual(self.card.lr_spinbox.value(), 0.004)
        self.app.processEvents()

    def test_open_shortcut_discard_and_edit_restored_project(
        self,
    ):
        self.card.lr_spinbox.setValue(0.004)
        self.window.show()
        self.window.activateWindow()
        self.app.processEvents()
        with patch(
            "main.QtWidgets.QFileDialog.getOpenFileName",
            return_value=(str(self.path), ""),
        ):
            with patch(
                "main.QtWidgets.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Discard,
            ):
                QTest.keyClick(
                    self.window,
                    Qt.Key.Key_O,
                    Qt.KeyboardModifier.ControlModifier,
                )
        self.assertEqual(
            self.window.training_panel.cards[self.model.id].lr_spinbox.value(),
            0.003,
        )
        self.app.processEvents()
        self.assertTrue(self.window.flowView.isVisible())
        self.window.tabWidget.setCurrentWidget(self.window.trainingTab)
        self.app.processEvents()
        self.assertTrue(self.window.training_panel.isVisible())
        self.assertFalse(self.window.isWindowModified())
        self.window.training_panel.cards[self.model.id].lr_spinbox.setValue(
            0.005
        )
        self.assertTrue(self.window.isWindowModified())

    def test_cancelled_dialog_and_wrong_extension_preserve_current_project(
        self,
    ):
        scene = self.window.flowScene
        with patch(
            "main.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")
        ):
            self.assertFalse(self.window.open_project())
        wrong = self.path.with_suffix(".zip")
        wrong.write_bytes(self.path.read_bytes())
        with patch("main.QtWidgets.QMessageBox.warning"):
            self.assertFalse(self.open_path(wrong))
        self.assertIs(self.window.flowScene, scene)

    def test_save_before_opening_same_file_reads_the_newly_saved_version(self):
        self.card.lr_spinbox.setValue(0.006)
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Save,
        ):
            self.assertTrue(self.open_path())
        self.assertEqual(
            self.window.training_panel.cards[self.model.id].lr_spinbox.value(),
            0.006,
        )

    def test_opening_another_project_stops_old_training(self):
        self.card.epochs_spinbox.setValue(100000)
        self.card.start_training()
        thread = self.card.training_thread
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Discard,
        ):
            self.assertTrue(self.open_path())
        self.assertFalse(thread.isRunning())
        self.app.processEvents()
        self.assertFalse(self.window.isWindowModified())

    def test_regression_weights_and_metrics_restore_alongside_classification(
        self,
    ):
        from FlowNodes import RegressionModel

        node = self.window.flowScene.create_node(RegressionModel)
        node.model.replace_architecture(
            [
                {"type": "linear", "in_features": 1, "out_features": 1},
            ]
        )
        card = self.window.training_panel.cards[node.id]
        card.ensure_network()
        card.update_metrics(1, {"Train": {"Loss": 0.5, "MSE": 0.5, "R2": 0.2}})
        self.app.processEvents()
        self.assertTrue(self.window.save_project())
        x = torch.tensor([[1.0], [2.0]])
        with torch.no_grad():
            expected = card.network(x)
        self.assertTrue(self.open_path())
        restored = self.window.training_panel.cards[node.id]
        self.assertEqual(restored.axes_data["Train"]["MSE"], [0.5])
        self.assertEqual(restored.axes_data["Train"]["R2"], [0.2])
        with torch.no_grad():
            self.assertTrue(torch.equal(restored.network(x), expected))

    def test_older_project_retains_its_original_split_algorithm(self):
        from contextlib import closing

        from sklearn.model_selection import train_test_split

        with closing(sqlite3.connect(self.path)) as database, database:
            graph = json.loads(
                database.execute("SELECT graph_json FROM projects").fetchone()[
                    0
                ]
            )
            settings = next(
                item["settings"]
                for item in graph["nodes"]
                if item["id"] == self.tvt.id
            )
            settings.pop("split_version")
            settings["val_pct"] = 0.3
            database.execute(
                "UPDATE projects SET graph_json=?", (json.dumps(graph),)
            )
        features = self.source.model.dt.drop(columns="Claim")
        target = (self.source.model.dt["Claim"] == "Yes").astype("int64")
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            target,
            test_size=0.25,
            random_state=settings["random_state"],
        )
        train_x, val_x, train_y, val_y = train_test_split(
            train_x,
            train_y,
            test_size=0.3 * 1.25,
            random_state=settings["random_state"],
        )
        self.assertTrue(self.open_path())
        split = self.window.flowScene.nodes[self.tvt.id].model
        self.assertEqual(split.split_version, 1)
        for port, expected in ((0, train_x), (2, val_x), (4, test_x)):
            pd.testing.assert_frame_equal(split.out_data(port).value, expected)

    def test_old_model_lr_setting_is_ignored_and_training_lr_is_restored(self):
        with closing(sqlite3.connect(self.path)) as database, database:
            graph = json.loads(
                database.execute("SELECT graph_json FROM projects").fetchone()[
                    0
                ]
            )
            settings = next(
                item["settings"]
                for item in graph["nodes"]
                if item["id"] == self.model.id
            )
            settings["lr_spinbox"] = 0.7
            database.execute(
                "UPDATE projects SET graph_json=?", (json.dumps(graph),)
            )
        self.assertTrue(self.open_path())
        model = self.window.flowScene.nodes[self.model.id].model
        card = self.window.training_panel.cards[self.model.id]
        self.assertFalse(hasattr(model, "lr_spinbox"))
        self.assertEqual(card.lr_spinbox.value(), 0.003)
