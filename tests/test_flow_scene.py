import unittest

from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication
from qtpynodeeditor import DataModelRegistry, PortType

from FlowNodes import TextInputNode, TextOutputNode
from FlowScene import WeaveFlowScene


class ConnectionLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_reconnect_keeps_one_deletion_handler(self):
        registry = DataModelRegistry()
        registry.register_model(TextInputNode)
        registry.register_model(TextOutputNode)
        scene = WeaveFlowScene(registry)
        self.addCleanup(scene.clear_scene)
        source = scene.create_node(TextInputNode)
        target = scene.create_node(TextOutputNode)
        output = source[PortType.output][0]
        input_port = target[PortType.input][0]
        source.model.editor.setText("hello")
        connection = scene.create_connection(output, input_port)
        deleted = QSignalSpy(scene.connection_deleted)

        for count in range(1, 6):
            connection.propagate_empty_data()
            connection.clear_node(PortType.input)
            connection.required_port = PortType.input
            self.assertEqual(len(deleted), count)
            self.assertEqual(target.model.label.text(), "No data")

            input_port.add_connection(connection)
            connection.connect_to(input_port)
            scene.connection_created.emit(connection)
            source.on_data_updated(output)
            self.assertEqual(target.model.label.text(), "hello")

        scene.delete_connection(connection)
        self.assertEqual(len(deleted), 6)
        self.assertEqual(target.model.label.text(), "No data")
        self.assertEqual(scene.connections, [])

        replacement = scene.create_connection(output, input_port)
        scene.delete_connection(replacement)
        self.assertEqual(len(deleted), 7)


if __name__ == "__main__":
    unittest.main()
