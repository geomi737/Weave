import unittest
from unittest.mock import patch

import pandas as pd
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication
from qtpynodeeditor import DataModelRegistry, PortType

from FlowNodes import DatasetNode, OneHotNode
from FlowScene import WeaveFlowScene
from FlowTypes import DataframeData


class OneHotNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        registry = DataModelRegistry()
        registry.register_model(OneHotNode)
        registry.register_model(DatasetNode)
        self.scene = WeaveFlowScene(registry)
        self.addCleanup(self.scene.clear_scene)
        self.node = self.scene.create_node(OneHotNode)
        self.model = self.node.model
        self.train = pd.DataFrame(
            {"color": ["red", "blue", "red"], "size": ["S", "M", "S"]},
            index=[10, 20, 30],
        )

    def send(self, index, frame):
        data = DataframeData(frame) if frame is not None else None
        self.model.set_in_data(data, self.node[PortType.input][index])

    def test_optional_splits_share_train_schema_without_leakage(self):
        for optional in ((), (1,), (2,), (1, 2)):
            with self.subTest(optional=optional):
                self.send(0, self.train)
                encoder = self.model.encoder
                with patch.object(encoder, "fit", wraps=encoder.fit) as fit:
                    for index in (1, 2):
                        frame = pd.DataFrame(
                            {"size": ["L"], "color": ["green"]}, index=[99]
                        ) if index in optional else None
                        self.send(index, frame)
                    fit.assert_not_called()
                result = self.model.out_data(0).value
                pd.testing.assert_index_equal(result.index, self.train.index)
                self.assertNotIn("color_green", result.columns)
                for index in (1, 2):
                    output = self.model.out_data(index)
                    if index in optional:
                        pd.testing.assert_index_equal(output.value.columns, result.columns)
                        self.assertEqual(output.value.index.tolist(), [99])
                        self.assertEqual(output.value.to_numpy().sum(), 0)
                    else:
                        self.assertIsNone(output)

    def test_waits_for_train_and_clears_outputs_when_train_is_removed(self):
        self.send(2, self.train)
        self.assertEqual(self.model.outputs, [None, None, None])
        self.send(0, self.train)
        self.assertIsNotNone(self.model.out_data(2))
        updates = QSignalSpy(self.model.data_updated)
        self.send(0, None)
        self.assertEqual(self.model.outputs, [None, None, None])
        self.assertEqual([list(item) for item in updates], [[0], [1], [2]])

    def test_drop_modes_and_frequency_limits(self):
        frame = pd.DataFrame({"letter": ["a"] * 4 + ["b"] * 2 + ["c"]})
        self.send(0, frame)
        self.assertEqual(self.model.out_data(0).value.shape[1], 3)
        self.model.drop_group.button(1).click()
        self.assertEqual(self.model.out_data(0).value.shape[1], 2)
        self.model.drop_group.button(2).click()
        self.assertEqual(self.model.out_data(0).value.shape[1], 3)
        self.model.min_frequency_spinbox.setValue(0.4)
        self.assertEqual(self.model.out_data(0).value.shape[1], 1)
        self.model.drop_group.button(0).click()
        self.model.min_frequency_spinbox.setValue(2)
        self.assertEqual(self.model.encoder.min_frequency, 2)
        self.model.max_categories_spinbox.setValue(2)
        self.assertEqual(self.model.out_data(0).value.shape[1], 2)
        self.model.min_frequency_spinbox.setValue(1.5)
        self.assertIsNone(self.model.out_data(0))
        self.assertIn("min_frequency", self.model.status_label.text())

    def test_bad_optional_schema_does_not_break_other_outputs(self):
        self.send(0, self.train)
        self.send(1, pd.DataFrame({"other": ["a"]}))
        self.send(2, self.train.iloc[:0])
        self.assertIsNotNone(self.model.out_data(0))
        self.assertIsNone(self.model.out_data(1))
        self.assertIn("Validate", self.model.status_label.text())
        empty = self.model.out_data(2).value
        self.assertEqual(len(empty), 0)
        pd.testing.assert_index_equal(empty.columns, self.model.out_data(0).value.columns)
        self.send(1, self.train)
        self.assertIsNotNone(self.model.out_data(1))
        self.assertEqual(self.model.status_label.text(), "Готово")

    def test_train_update_refits_and_propagates_to_dataset(self):
        sink = self.scene.create_node(DatasetNode)
        self.scene.create_connection(
            self.node[PortType.output][0], sink[PortType.input][0]
        )
        source = self.scene.create_node(DatasetNode)
        source.model.dt = self.train
        connection = self.scene.create_connection(
            source[PortType.output][0], self.node[PortType.input][0]
        )
        self.assertIn("color_red", sink.model.dt.columns)
        source.model.dt = pd.DataFrame({"new": ["x", "y"]})
        source.model.data_updated.emit(0)
        self.assertEqual(sink.model.dt.columns.tolist(), ["new_x", "new_y"])
        self.scene.delete_connection(connection)
        self.assertIsNone(sink.model.dt)
        self.assertEqual(self.model.outputs, [None, None, None])

    def test_empty_and_mixed_train_clear_outputs_and_can_recover(self):
        for bad_train in (
            self.train.iloc[:0],
            pd.DataFrame({"mixed": pd.Series(["a", 1], dtype=object)}),
        ):
            with self.subTest(train=bad_train):
                self.send(0, self.train)
                self.send(0, bad_train)
                self.assertIsNone(self.model.out_data(0))
                self.assertIn("Train", self.model.status_label.text())
                self.send(0, self.train)
                self.assertIsNotNone(self.model.out_data(0))

    def test_numeric_columns_pass_through_and_empty_optional_preserves_schema(self):
        train = pd.DataFrame({"number": [1, 2]}, index=[10, 20])
        self.send(0, train)
        self.send(2, train.iloc[:0])
        pd.testing.assert_frame_equal(self.model.out_data(0).value, train)
        pd.testing.assert_frame_equal(self.model.out_data(2).value, train.iloc[:0])

    def test_schema_change_clears_stale_optional_and_recovers(self):
        self.send(0, self.train)
        self.send(2, self.train)
        replacement = pd.DataFrame({"new": ["x", "y"]})
        self.send(0, replacement)
        self.assertIsNotNone(self.model.out_data(0))
        self.assertIsNone(self.model.out_data(2))
        self.assertIn("Test", self.model.status_label.text())
        self.send(2, replacement)
        pd.testing.assert_frame_equal(
            self.model.out_data(0).value, self.model.out_data(2).value
        )

    def test_frequency_one_is_an_integer_count(self):
        self.send(0, self.train)
        self.model.min_frequency_spinbox.setValue(1)
        self.assertEqual(self.model.encoder.min_frequency, 1)
        self.assertIsNotNone(self.model.out_data(0))


if __name__ == "__main__":
    unittest.main()
