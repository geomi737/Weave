import unittest

import pandas as pd
import torch
from PyQt6.QtWidgets import QApplication
from qtpynodeeditor import DataModelRegistry, PortType

from FlowNodes import (
    BinaryReplaceNode,
    DatasetNode,
    OneHotNode,
    SplitColumnNode,
)
from FlowScene import WeaveFlowScene
from FlowTypes import DataframeData, SeriesData


class SchemaRefreshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        registry = DataModelRegistry()
        for model in (
            BinaryReplaceNode,
            DatasetNode,
            OneHotNode,
            SplitColumnNode,
        ):
            registry.register_model(model)
        self.scene = WeaveFlowScene(registry)
        self.addCleanup(self.scene.clear_scene)

    def connect(self, source, output, sink, input_port):
        self.scene.create_connection(
            source[PortType.output][output], sink[PortType.input][input_port]
        )

    def test_changing_split_column_recovers_after_optional_inputs_refresh(
        self,
    ):
        split = self.scene.create_node(SplitColumnNode)
        ohe = self.scene.create_node(OneHotNode)
        sinks = [self.scene.create_node(DatasetNode) for _ in range(3)]
        frame = pd.DataFrame({"Claim": ["yes", "no"], "Agency": ["A", "B"]})
        split.model.set_in_data(DataframeData(frame), split[PortType.input][0])
        self.connect(split, 0, ohe, 0)
        for port, sink in enumerate(sinks):
            self.connect(ohe, port, sink, 0)
            if port:
                ohe.model.set_in_data(
                    split.model.out_data(0), ohe[PortType.input][port]
                )

        for selected in ("Agency", "Claim", "Agency"):
            split.model.comboBox.setCurrentText(selected)
            split.model.update_output()
            self.assertIsNotNone(sinks[0].model.dt)
            self.assertIsNone(sinks[1].model.dt)
            self.assertIsNone(sinks[2].model.dt)
            for port in (1, 2):
                ohe.model.set_in_data(
                    split.model.out_data(0), ohe[PortType.input][port]
                )
                pd.testing.assert_frame_equal(
                    sinks[0].model.dt, sinks[port].model.dt
                )

    def test_numeric_transition_and_optional_column_reordering(self):
        node = self.scene.create_node(OneHotNode)
        for frame in (
            pd.DataFrame({"color": ["a", "b"], "number": [1, 2]}),
            pd.DataFrame({"number": [1, 2]}),
            pd.DataFrame({"color": ["x", "y"], "number": [1, 2]}),
        ):
            node.model.set_in_data(
                DataframeData(frame), node[PortType.input][0]
            )
            node.model.set_in_data(
                DataframeData(frame[frame.columns[::-1]]),
                node[PortType.input][2],
            )
            pd.testing.assert_frame_equal(
                node.model.out_data(0).value, node.model.out_data(2).value
            )

    def test_binary_output_is_numeric_and_keeps_index(self):
        node = self.scene.create_node(BinaryReplaceNode)
        series = pd.Series(
            ["No", "Yes", "No"], index=[10, 20, 30], name="Claim"
        )
        node.model.set_in_data(SeriesData(series), node[PortType.input][0])
        result = node.model.out_data(0).value
        expected = pd.Series(
            [1, 0, 1], index=series.index, name="Claim", dtype="int64"
        )
        pd.testing.assert_series_equal(result, expected)
        tensor = torch.tensor(result.to_numpy(), dtype=torch.long)
        self.assertEqual(tensor.tolist(), [1, 0, 1])

    def test_binary_rejects_missing_targets_instead_of_turning_them_into_zero(
        self,
    ):
        node = self.scene.create_node(BinaryReplaceNode)
        series = pd.Series(["Yes", "No", None], name="Claim")
        node.model.set_in_data(SeriesData(series), node[PortType.input][0])
        self.assertIsNone(node.model.out_data(0))
        self.assertIn("non-missing", node.model.status_label.text())
