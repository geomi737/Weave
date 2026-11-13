import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication
from qtpynodeeditor import DataModelRegistry, PortType

from FlowNodes import (
    BinaryReplaceNode,
    DatasetNode,
    DropNode,
    OneHotNode,
    SeriesNode,
    SplitColumnNode,
    TVTSplitNode,
)
from FlowScene import WeaveFlowScene
from FlowTypes import DataframeData, SeriesData
from main import MainWindow


class NodeUpdatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Ошибка в Qt-слоте должна провалить тест, а не оборвать весь запуск.
        self.slot_errors = []
        hook = patch(
            "sys.excepthook",
            lambda kind, value, traceback: self.slot_errors.append(str(value)),
        )
        hook.start()
        self.addCleanup(hook.stop)
        self.addCleanup(lambda: self.assertEqual(self.slot_errors, []))
        registry = DataModelRegistry()
        for model in (
            BinaryReplaceNode,
            DatasetNode,
            DropNode,
            OneHotNode,
            SeriesNode,
            SplitColumnNode,
            TVTSplitNode,
        ):
            registry.register_model(model)
        self.scene = WeaveFlowScene(registry)
        self.addCleanup(self.scene.clear_scene)

    def connect(self, source, target, output=0, input_port=0):
        return self.scene.create_connection(
            source[PortType.output][output], target[PortType.input][input_port]
        )

    def send(self, node, value, port=0):
        data = None
        if value is not None:
            data = (
                SeriesData(value)
                if isinstance(value, pd.Series)
                else DataframeData(value)
            )
        node.model.set_in_data(data, node[PortType.input][port])

    def test_drop_refresh_preserves_column_and_clears_disconnected_sink(self):
        source = self.scene.create_node(DatasetNode)
        drop = self.scene.create_node(DropNode)
        sink = self.scene.create_node(DatasetNode)
        self.connect(drop, sink)
        connection = self.connect(source, drop)
        self.send(source, pd.DataFrame({"a": [1], "b": [2]}))
        drop.model.comboBox.setCurrentIndex(1)
        updates = QSignalSpy(drop.model.data_updated)
        self.send(source, pd.DataFrame({"b": [4, 5], "a": [6, 7]}))
        self.assertEqual(len(updates), 1)
        self.assertEqual(drop.model.comboBox.currentData(), "b")
        pd.testing.assert_frame_equal(
            sink.model.dt, pd.DataFrame({"a": [6, 7]})
        )
        self.scene.delete_connection(connection)
        self.assertIsNone(sink.model.dt)

    def test_empty_or_removed_column_clears_output(self):
        for model in (DropNode, SplitColumnNode):
            with self.subTest(model=model):
                node = self.scene.create_node(model)
                self.send(node, pd.DataFrame({"a": [1], "b": [2]}))
                self.send(node, pd.DataFrame({"b": [3]}))
                self.assertIsNone(node.model.out_data(0))
                self.send(node, pd.DataFrame(index=[0]))
                self.assertIsNone(node.model.out_data(0))

    def test_drop_na_can_remove_every_column(self):
        node = self.scene.create_node(DropNode)
        node.model.dropna_checkbox.setChecked(True)
        self.send(node, pd.DataFrame({"a": [None], "b": [None]}))
        self.assertEqual(node.model.out_data(0).value.shape, (1, 0))
        self.send(node, None)
        self.assertIsNone(node.model.out_data(0))

    def test_split_refresh_with_different_rows_and_column_order(self):
        source = self.scene.create_node(DatasetNode)
        split = self.scene.create_node(SplitColumnNode)
        tvt = self.scene.create_node(TVTSplitNode)
        sink = self.scene.create_node(DatasetNode)
        self.connect(source, split)
        self.connect(split, tvt)
        self.connect(split, tvt, output=1, input_port=1)
        self.connect(tvt, sink)
        self.send(source, pd.DataFrame({"label": [0, 1] * 20, "f": range(40)}))
        self.send(source, pd.DataFrame({"f": range(60), "label": [0, 1] * 30}))
        self.assertEqual(split.model.comboBox.currentData(), "label")
        self.assertEqual(len(sink.model.dt), 48)
        self.assertEqual(sink.model.dt.columns.tolist(), ["f"])
        for x_port in (0, 4):
            X = tvt.model.out_data(x_port).value
            y = tvt.model.out_data(x_port + 1).value
            pd.testing.assert_index_equal(X.index, y.index)
            self.assertEqual(y.tolist(), (X["f"] % 2).tolist())
        self.send(source, None)
        self.assertIsNone(sink.model.dt)

    def test_numeric_column_names_keep_their_original_type(self):
        split = self.scene.create_node(SplitColumnNode)
        self.send(split, pd.DataFrame({10: [1, 2], 20: [3, 4]}))
        split.model.comboBox.setCurrentIndex(1)
        self.send(split, pd.DataFrame({20: [5, 6], 10: [7, 8]}))
        self.assertEqual(split.model.out_data(1).value.name, 20)
        self.assertEqual(split.model.out_data(0).value.columns.tolist(), [10])

    def test_full_preprocessing_chain_recovers_after_rows_and_schema_change(
        self,
    ):
        source = self.scene.create_node(DatasetNode)
        drop = self.scene.create_node(DropNode)
        split = self.scene.create_node(SplitColumnNode)
        tvt = self.scene.create_node(TVTSplitNode)
        onehot = self.scene.create_node(OneHotNode)
        sinks = [self.scene.create_node(DatasetNode) for _ in range(3)]
        self.connect(source, drop)
        self.connect(drop, split)
        self.connect(split, tvt)
        self.connect(split, tvt, output=1, input_port=1)
        for index, sink in enumerate(sinks):
            self.connect(tvt, onehot, output=2 * index, input_port=index)
            self.connect(onehot, sink, output=index)
        tvt.model.include_validate.setChecked(True)
        for count, column in ((40, "city"), (80, "region")):
            self.send(
                source,
                pd.DataFrame(
                    {
                        "unused": range(count),
                        "target": [0, 1] * (count // 2),
                        column: ["a", "b"] * (count // 2),
                        "n": range(count),
                    }
                ),
            )
            self.assertEqual(
                [len(sink.model.dt) for sink in sinks],
                [count // 2, count * 3 // 10, count // 5],
            )
            self.assertEqual(onehot.model.status_label.text(), "Ready")
            for sink in sinks:
                self.assertEqual(
                    set(sink.model.dt.columns),
                    {"n", f"{column}_a", f"{column}_b"},
                )
        self.send(source, None)
        self.assertTrue(all(sink.model.dt is None for sink in sinks))

    def test_regression_split_controls_cache_and_validation_proportions(self):
        node = self.scene.create_node(TVTSplitNode)
        X = pd.DataFrame({"f": range(1000)})
        self.send(node, X)
        self.send(node, X["f"] * 0.37, port=1)
        self.assertEqual(len(node.model.out_data(0).value), 800)
        self.assertIsNone(node.model.out_data(2))
        updates = QSignalSpy(node.model.data_updated)
        node.model.test_pct.setValue(0.25)
        self.assertEqual(len(updates), 6)
        self.assertEqual(len(node.model.out_data(4).value), 250)
        first = node.model.out_data(0).value.copy()
        pd.testing.assert_frame_equal(first, node.model.out_data(0).value)
        node.model.test_pct.setValue(0.2)
        node.model.include_validate.setChecked(True)
        self.assertEqual(
            [len(node.model.out_data(i).value) for i in (0, 2, 4)],
            [500, 300, 200],
        )
        node.model.val_pct.setValue(0.2)
        self.assertEqual(len(node.model.out_data(2).value), 200)

    def test_classification_preserves_class_proportions(self):
        node = self.scene.create_node(TVTSplitNode)
        self.send(node, pd.DataFrame({"f": range(100)}))
        self.send(node, pd.Series([0] * 80 + [1] * 20), port=1)
        self.assertEqual(
            node.model.out_data(5).value.value_counts().to_dict(),
            {0: 16, 1: 4},
        )

    def test_invalid_split_clears_outputs_and_recovers(self):
        node = self.scene.create_node(TVTSplitNode)
        self.send(node, pd.DataFrame({"f": range(20)}))
        self.send(node, pd.Series([0, 1] * 10), port=1)
        node.model.include_validate.setChecked(True)
        node.model.val_pct.setValue(0.9)
        self.assertTrue(all(node.model.out_data(i) is None for i in range(6)))
        self.assertIn("Sum", node.model.status_label.text())
        node.model.val_pct.setValue(0.3)
        self.assertIsNotNone(node.model.out_data(0))
        self.send(node, pd.DataFrame({"f": range(21)}))
        self.assertIsNone(node.model.out_data(0))
        self.send(node, pd.Series([0, 1] * 10 + [0]), port=1)
        self.assertIsNotNone(node.model.out_data(0))
        self.send(node, pd.Series([0] * 20 + [None]), port=1)
        self.assertIsNone(node.model.out_data(0))

    def test_rare_classes_and_tiny_dataset_do_not_crash(self):
        node = self.scene.create_node(TVTSplitNode)
        self.send(node, pd.DataFrame({"f": range(10)}))
        self.send(node, pd.Series([0] * 9 + [1]), port=1)
        self.assertEqual(len(node.model.out_data(4).value), 2)
        self.assertIn("Without stratification", node.model.status_label.text())
        self.send(node, pd.DataFrame({"f": [1]}))
        self.send(node, pd.Series([0]), port=1)
        self.assertIsNone(node.model.out_data(0))

    def test_binary_refresh_preserves_value_and_explains_invalid_input(self):
        source = self.scene.create_node(SeriesNode)
        binary = self.scene.create_node(BinaryReplaceNode)
        sink = self.scene.create_node(SeriesNode)
        self.connect(source, binary)
        self.connect(binary, sink)
        self.send(source, pd.Series(["yes", "no", "yes"]))
        self.send(source, pd.Series(["no", "yes", "no"]))
        self.assertEqual(binary.model.comboBox.currentData(), "yes")
        self.assertEqual(sink.model.dt.tolist(), [0, 1, 0])
        self.assertEqual(str(sink.model.dt.dtype), "int64")
        self.send(source, pd.Series(["a", "b", "c"]))
        self.assertIsNone(sink.model.dt)
        self.assertIn("two", binary.model.status_label.text())
        self.send(source, None)
        self.assertIsNone(source.model.out_data(0))
        self.assertIsNone(sink.model.out_data(0))

    def test_binary_numeric_values_keep_selection_on_refresh(self):
        node = self.scene.create_node(BinaryReplaceNode)
        self.send(node, pd.Series([1, 2, 1]))
        self.send(node, pd.Series([2, 1, 2]))
        self.assertEqual(node.model.comboBox.currentData(), 1)
        self.assertEqual(node.model.out_data(0).value.tolist(), [0, 1, 0])

    def test_csv_errors_are_shown_and_valid_load_still_propagates(self):
        source = self.scene.create_node(DatasetNode)
        sink = self.scene.create_node(DatasetNode)
        self.connect(source, sink)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            for content in (None, b"", b"a,b\n1,2\n3,4,5\n", b"a\n\xff"):
                if content is not None:
                    path.write_bytes(content)
                with (
                    patch(
                        "FlowNodes.QFileDialog.getOpenFileName",
                        return_value=(str(path), ""),
                    ),
                    patch("FlowNodes.QMessageBox.warning") as warning,
                ):
                    source.model.load_btn.click()
                warning.assert_called_once()
                self.assertIsNone(sink.model.dt)
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            with patch(
                "FlowNodes.QFileDialog.getOpenFileName",
                return_value=(str(path), ""),
            ):
                source.model.load_btn.click()
            self.assertEqual(
                sink.model.dt.to_dict("list"), {"a": [1], "b": [2]}
            )
            source.model.load_btn.click()
            self.assertIsNone(sink.model.dt)

    def test_preview_handles_zero_columns_and_nonstring_headers(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.dataset_preview(pd.DataFrame({10: [2]}))
        self.assertEqual(
            window.tableWidget.horizontalHeaderItem(0).text(), "10"
        )
        window.dataset_preview(pd.DataFrame(index=[0]))
        window.dataset_analysis(pd.DataFrame(index=[0]))
        self.assertEqual(window.tableWidget.columnCount(), 0)
        self.assertIn("Columns: 0", window.datasetAnalysis.toPlainText())
        window.dataset_analysis(None)
        self.assertEqual(window.datasetAnalysis.toPlainText(), "")


if __name__ == "__main__":
    unittest.main()
