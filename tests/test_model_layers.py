import unittest

from PyQt6.QtWidgets import QApplication, QPushButton
from qtpynodeeditor import DataModelRegistry

from FlowNodes import Model
from FlowScene import WeaveFlowScene


class ModelLayersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_remove_middle_layer_preserves_other_values_and_resizes_node(self):
        registry = DataModelRegistry()
        registry.register_model(Model)
        scene = WeaveFlowScene(registry)
        self.addCleanup(scene.clear_scene)
        node = scene.create_node(Model)
        model = node.model
        initial_height = node.size.height()

        for _ in range(3):
            model.add_linear_button.click()
            self.app.processEvents()

        first, middle, last = model.layers
        first["out_spinbox"].setValue(32)
        last["in_spinbox"].setValue(32)
        full_height = node.size.height()
        self.assertGreater(full_height, initial_height)

        middle["widget"].findChild(QPushButton).click()
        self.app.processEvents()

        self.assertEqual(model.layers, [first, last])
        self.assertEqual(model.layers_layout.count(), 2)
        self.assertEqual(first["out_spinbox"].value(), 32)
        self.assertEqual(last["in_spinbox"].value(), 32)
        self.assertLess(node.size.height(), full_height)

        for layer in list(model.layers):
            layer["widget"].findChild(QPushButton).click()
            self.app.processEvents()
        self.assertEqual(model.layers, [])
        self.assertEqual(model.layers_layout.count(), 0)
        self.assertEqual(node.size.height(), initial_height)

    def test_mixed_layers_keep_order_and_activation_selection(self):
        registry = DataModelRegistry()
        registry.register_model(Model)
        scene = WeaveFlowScene(registry)
        self.addCleanup(scene.clear_scene)
        node = scene.create_node(Model)
        model = node.model

        model.add_linear_button.click()
        model.add_activation_button.click()
        model.add_linear_button.click()
        self.app.processEvents()
        first, activation, last = model.layers
        combo = activation["activation_combobox"]
        self.assertEqual(combo.currentText(), "ReLU")
        combo.setCurrentText("Softmax")
        full_height = node.size.height()

        first["widget"].findChild(QPushButton).click()
        self.app.processEvents()
        self.assertEqual(model.layers, [activation, last])
        self.assertEqual(combo.currentText(), "Softmax")
        self.assertLess(node.size.height(), full_height)

        activation["widget"].findChild(QPushButton).click()
        self.app.processEvents()
        self.assertEqual(model.layers, [last])
        self.assertEqual(model.layers_layout.count(), 1)
