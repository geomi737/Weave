import os

from qtpy.QtWidgets import QWidget
from FlowTypes import TextData
from qtpynodeeditor import (
    DataModelRegistry,
    FlowScene,
    FlowView,
    NodeData,
    NodeDataModel,
    NodeDataType,
    Port,
    PortType
)

from PyQt6.QtWidgets import QLineEdit, QLabel


class TextInputNode(NodeDataModel):
    name = "Text Input"
    caption = "Text Input"

    num_ports = {
        PortType.input: 0,
        PortType.output: 1
    }

    data_type = TextData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)

        self.editor = QLineEdit()
        self.editor.textChanged.connect(self.on_text_changed)

    def embedded_widget(self) -> QWidget:
        return self.editor

    def out_data(self, port: int) -> NodeData:
        return TextData(self.editor.text())

    def on_text_changed(self):
        self.data_updated.emit(0)


class TextOutputNode(NodeDataModel):
    name = "Text Output"
    caption = "Text Output"

    num_ports = {
        PortType.input: 1,
        PortType.output: 0
    }

    data_type = TextData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)

        self.label = QLabel()
        self.label.setText("No data")

    def embedded_widget(self) -> QWidget:
        return self.label

    def set_in_data(self, node_data: TextData, port: Port):
        if node_data is None:
            self.label.setText("No data")
            return

        self.label.setText(node_data.value)


class TextTransformNode(NodeDataModel):
    name = "Text Transform"
    caption = "Text Transform"

    num_ports = {
        PortType.input: 1,
        PortType.output: 2
    }

    port_caption = {
        PortType.input: {
            0: "Text"
        },
        PortType.output: {
            0: "Uppercase",
            1: "Lowercase"
        }
    }
    port_caption_visible = True
    
    data_type = TextData.data_type
    
    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
