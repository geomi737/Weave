import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PyQt6.QtWidgets import QApplication
from qtpynodeeditor import PortType

from FlowNodes import (
    ClassificationModel, DatasetNode, RegressionModel, SplitColumnNode, TVTSplitNode,
)
from FlowTypes import DataframeData, SeriesData
from Training import TrainingSession, prepare_datasets
from main import MainWindow


def linear(inputs=1, outputs=1):
    return {"type": "linear", "in_features": inputs, "out_features": outputs}


def regression_inputs(optional=True):
    x = pd.DataFrame({"x": np.linspace(-1, 1, 32)})
    y = 2 * x["x"] + 0.5
    return [x, y, x.copy() if optional else None, y.copy() if optional else None,
            x.copy() if optional else None, y.copy() if optional else None]


def classification_inputs():
    x = pd.DataFrame({"x": [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0]})
    y = pd.Series(["left"] * 3 + ["right"] * 3)
    return [x, y, x.copy(), y.copy(), x.copy(), y.copy()]


class TrainingBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_regression_improves_and_reports_all_splits(self):
        torch.manual_seed(3)
        session = TrainingSession("regression", [linear()])
        datasets = session.prepare(regression_inputs(), 0.05)
        before = session.evaluate(datasets)["Train"]["Loss"]
        for _ in range(120):
            self.assertTrue(session.train_epoch(datasets, lambda: False))
        metrics = session.evaluate(datasets)
        self.assertEqual(set(metrics), {"Train", "Validation", "Test"})
        self.assertLess(metrics["Train"]["Loss"], before / 100)
        self.assertGreater(metrics["Test"]["R2"], 0.99)
        self.assertEqual(set(metrics["Train"]), {"Loss", "RMSE", "MAE", "R2"})

    def test_classification_mapping_metrics_and_checkpoint_roundtrip(self):
        torch.manual_seed(4)
        session = TrainingSession("classification", [linear(outputs=2)])
        datasets = session.prepare(classification_inputs(), 0.1)
        for _ in range(40):
            session.train_epoch(datasets, lambda: False)
        metrics = session.evaluate(datasets)
        self.assertEqual(session.classes, ["left", "right"])
        self.assertEqual(metrics["Test"]["Accuracy"], 1.0)
        self.assertEqual(metrics["Test"]["F1"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.zip"
            session.export_zip(path)
            restored = TrainingSession.import_zip(path, "classification")
            torch.testing.assert_close(session.model(datasets["Test"][0]),
                                       restored.model(datasets["Test"][0]), rtol=0, atol=0)
            self.assertEqual(restored.epochs, session.epochs)
            self.assertEqual(restored.classes, session.classes)
            restored_data = restored.prepare(classification_inputs(), 0.02)
            # Exact continuation also depends on the saved Adam moments.
            torch.manual_seed(99)
            session.prepare(classification_inputs(), 0.02)
            session.train_epoch(datasets, lambda: False)
            torch.manual_seed(99)
            restored.train_epoch(restored_data, lambda: False)
            for actual, expected in zip(restored.model.parameters(), session.model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), {"metadata.json", "weights.pt", "model.py"})
            with self.assertRaisesRegex(ValueError, "task"):
                TrainingSession.import_zip(path, "regression")

    def test_optional_datasets_never_update_weights(self):
        states = []
        for optional in (False, True):
            torch.manual_seed(12)
            session = TrainingSession("regression", [linear()])
            inputs = regression_inputs(optional)
            if optional:
                inputs[3] = inputs[3] + 10000
                inputs[5] = inputs[5] - 10000
            datasets = session.prepare(inputs, 0.01)
            for _ in range(3):
                session.train_epoch(datasets, lambda: False)
                session.evaluate(datasets)
            states.append(session.model.state_dict())
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)

    def test_validation_rejects_partial_misaligned_and_unknown_data(self):
        inputs = regression_inputs()
        inputs[3] = None
        with self.assertRaisesRegex(ValueError, "both Validation"):
            prepare_datasets(inputs, "regression")
        inputs = regression_inputs()
        inputs[1] = inputs[1].set_axis(range(1, 33))
        with self.assertRaisesRegex(ValueError, "indexes"):
            prepare_datasets(inputs, "regression")
        inputs = classification_inputs()
        inputs[3] = pd.Series(["unknown"] * 6)
        with self.assertRaisesRegex(ValueError, "unknown"):
            prepare_datasets(inputs, "classification")
        inputs = regression_inputs()
        inputs[0].iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            prepare_datasets(inputs, "regression")

    def test_output_shape_logits_and_single_sample_r2(self):
        session = TrainingSession("classification", [linear(outputs=1)])
        with self.assertRaisesRegex(ValueError, "2 output"):
            session.prepare(classification_inputs(), 0.01)
        session = TrainingSession("classification", [linear(outputs=2),
                                  {"type": "activation", "activation": "Softmax"}])
        with self.assertRaisesRegex(ValueError, "raw logits"):
            session.prepare(classification_inputs(), 0.01)
        session = TrainingSession("regression", [linear()])
        inputs = regression_inputs()
        inputs[4], inputs[5] = inputs[4].iloc[:1], inputs[5].iloc[:1]
        datasets = session.prepare(inputs, 0.01)
        self.assertIsNone(session.evaluate(datasets)["Test"]["R2"])

    def test_bad_archive_and_unbounded_architecture_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "parameter limit"):
            TrainingSession("regression", [linear(100000, 100000)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            path.write_text("not a zip")
            with self.assertRaises(ValueError):
                TrainingSession.import_zip(path, "regression")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("metadata.json", json.dumps({"format": "other"}))
                archive.writestr("weights.pt", b"bad weights")
            with self.assertRaises(ValueError):
                TrainingSession.import_zip(path, "regression")


class TrainingInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.errors = []
        hook = patch("sys.excepthook", lambda kind, value, traceback: self.errors.append(str(value)))
        hook.start()
        self.addCleanup(hook.stop)
        self.window = MainWindow()
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        self.window.training_panel.pause_all()
        self.wait_until(lambda: not self.window.training_panel.running())
        self.window.flowScene.clear_scene()
        self.window.flowScene.selectionChanged.disconnect()
        self.window.close()
        self.app.processEvents()
        self.assertEqual(self.errors, [])

    def wait_until(self, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(predicate(), "Timed out waiting for training worker")

    def make_node(self, kind=RegressionModel):
        node = self.window.flowScene.create_node(kind)
        node.model.add_linear_button.click()
        node.model.layers[0]["in_spinbox"].setValue(1)
        node.model.layers[0]["out_spinbox"].setValue(2 if kind is ClassificationModel else 1)
        inputs = classification_inputs() if kind is ClassificationModel else regression_inputs()
        for port, value in enumerate(inputs):
            data = SeriesData(value) if port % 2 else DataframeData(value)
            node.model.set_in_data(data, node[PortType.input][port])
        card = self.window.training_panel.cards[node.id]
        card.steps_spinbox.setValue(3)
        return node, card

    def test_train_all_distinct_cards_and_six_input_pair_rules(self):
        regression, first = self.make_node()
        classification, second = self.make_node(ClassificationModel)
        self.assertNotEqual(first.title(), second.title())
        self.assertEqual(regression.model.num_ports[PortType.input], 6)
        regression.model.set_in_data(None, regression[PortType.input][3])
        self.assertFalse(regression.model.ready_to_train())
        self.assertTrue(regression.model.train_button.isHidden())
        regression.model.set_in_data(SeriesData(regression_inputs()[3]), regression[PortType.input][3])
        self.assertTrue(regression.model.ready_to_train())
        self.window.train_all_button.click()
        self.assertIs(self.window.tabWidget.currentWidget(), self.window.trainingTab)
        self.wait_until(lambda: not self.window.training_panel.running())
        for card in (first, second):
            self.assertEqual(card.session.epochs, 3)
            self.assertEqual(card.status_label.text(), "Completed")
            self.assertTrue(card.export_button.isEnabled())
            self.assertEqual(set(card.session.last_metrics), {"Train", "Validation", "Test"})
        first.show_node_button.click()
        self.assertIs(self.window.tabWidget.currentWidget(), self.window.nodeEditTab)

    def test_pause_continue_import_and_export(self):
        node, card = self.make_node()
        card.steps_spinbox.setValue(100000)
        node.model.train_button.click()
        self.assertFalse(card.export_button.isEnabled())
        self.assertFalse(node.model.add_linear_button.isEnabled())
        self.wait_until(lambda: card.session.epochs > 0)
        card.pause_button.click()
        self.wait_until(lambda: card.worker is None)
        self.assertEqual(card.status_label.text(), "Paused")
        completed = card.session.epochs
        card.steps_spinbox.setValue(2)
        card.continue_button.click()
        self.wait_until(lambda: card.worker is None)
        self.assertEqual(card.session.epochs, completed + 2)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "regression.zip")
            with patch("TrainingUI.QFileDialog.getSaveFileName", return_value=(path, "")):
                card.export_button.click()
            imported_node, imported_card = self.make_node()
            with patch("FlowNodes.QFileDialog.getOpenFileName", return_value=(path, "")):
                imported_node.model.import_button.click()
            self.assertEqual(imported_card.session.epochs, card.session.epochs)
            self.assertEqual(imported_node.model.architecture(), node.model.architecture())
            self.assertEqual(imported_card.lr_spinbox.value(), card.lr_spinbox.value())
            imported_card.steps_spinbox.setValue(1)
            imported_card.continue_button.click()
            self.wait_until(lambda: imported_card.worker is None)
            self.assertEqual(imported_card.session.epochs, card.session.epochs + 1)

    def test_removed_node_can_finish_safely_and_export(self):
        node, card = self.make_node()
        card.steps_spinbox.setValue(100000)
        node.model.train_button.click()
        self.wait_until(lambda: card.session.epochs > 0)
        self.window.flowScene.remove_node(node)
        self.wait_until(lambda: card.worker is None)
        self.assertTrue(card.removed)
        self.assertTrue(card.export_button.isEnabled())
        self.assertFalse(card.continue_button.isEnabled())

    def test_invalid_training_data_is_reported_and_controls_recover(self):
        node, card = self.make_node()
        node.model.set_in_data(SeriesData(pd.Series([1.0])), node[PortType.input][1])
        node.model.train_button.click()
        self.wait_until(lambda: card.worker is None)
        self.assertIn("equal, nonzero lengths", card.status_label.text())
        self.assertTrue(node.model.add_linear_button.isEnabled())
        self.assertFalse(card.export_button.isEnabled())
        node.model.set_in_data(SeriesData(regression_inputs()[1]), node[PortType.input][1])
        card.continue_button.click()
        self.wait_until(lambda: card.worker is None)
        self.assertEqual(card.status_label.text(), "Completed")

    def test_window_close_pauses_workers_before_finishing(self):
        node, card = self.make_node()
        self.window.show()
        card.steps_spinbox.setValue(100000)
        node.model.train_button.click()
        self.wait_until(lambda: card.session.epochs > 0)
        self.window.close()
        self.wait_until(lambda: not self.window.training_panel.running())
        self.assertFalse(self.window.isVisible())
        self.assertEqual(card.status_label.text(), "Paused")

    def test_csv_to_split_to_model_with_real_connections(self):
        scene = self.window.flowScene
        source = scene.create_node(DatasetNode)
        target_split = scene.create_node(SplitColumnNode)
        dataset_split = scene.create_node(TVTSplitNode)
        model_node = scene.create_node(RegressionModel)
        model_node.model.restore_architecture([linear()])
        dataset_split.model.include_validate.setChecked(True)

        def connect(upstream, downstream, output=0, input_port=0):
            scene.create_connection(upstream[PortType.output][output],
                                    downstream[PortType.input][input_port])

        connect(source, target_split)
        connect(target_split, dataset_split)
        connect(target_split, dataset_split, output=1, input_port=1)
        for port in range(6):
            connect(dataset_split, model_node, output=port, input_port=port)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "data.csv")
            x, y = regression_inputs()[:2]
            pd.DataFrame({"target": y, "x": x["x"]}).to_csv(path, index=False)
            with patch("FlowNodes.QFileDialog.getOpenFileName", return_value=(path, "")):
                source.model.load_btn.click()
        self.assertTrue(model_node.model.ready_to_train())
        card = self.window.training_panel.cards[model_node.id]
        card.steps_spinbox.setValue(2)
        model_node.model.train_button.click()
        self.wait_until(lambda: card.worker is None)
        self.assertEqual(card.status_label.text(), "Completed")
        self.assertEqual(set(card.session.last_metrics), {"Train", "Validation", "Test"})


if __name__ == "__main__":
    unittest.main()
