import unittest

import pandas as pd
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication
from qtpynodeeditor import DataModelRegistry, PortType

from FlowNodes import DatasetNode
from FlowScene import WeaveFlowScene


class ConnectionLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reconnect_keeps_one_deletion_handler(self):
        registry = DataModelRegistry()
        registry.register_model(DatasetNode)
        scene = WeaveFlowScene(registry)
        self.addCleanup(scene.clear_scene)
        source = scene.create_node(DatasetNode)
        target = scene.create_node(DatasetNode)
        output = source[PortType.output][0]
        input_port = target[PortType.input][0]
        source.model.dt = pd.DataFrame({"value": [1, 2]})
        connection = scene.create_connection(output, input_port)
        deleted = QSignalSpy(scene.connection_deleted)

        for count in range(1, 6):
            connection.propagate_empty_data()
            connection.clear_node(PortType.input)
            connection.required_port = PortType.input
            self.assertEqual(len(deleted), count)
            self.assertIsNone(target.model.dt)

            input_port.add_connection(connection)
            connection.connect_to(input_port)
            scene.connection_created.emit(connection)
            source.on_data_updated(output)
            pd.testing.assert_frame_equal(target.model.dt, source.model.dt)

        scene.delete_connection(connection)
        self.assertEqual(len(deleted), 6)
        self.assertIsNone(target.model.dt)
        self.assertEqual(scene.connections, [])

        replacement = scene.create_connection(output, input_port)
        scene.delete_connection(replacement)
        self.assertEqual(len(deleted), 7)


if __name__ == "__main__":
    unittest.main()
