import os
import sys

# Указать до импорта qtpy и qtpynodeeditor.
os.environ["QT_API"] = "pyqt6"

from qtpy.QtWidgets import QApplication, QLabel, QLineEdit
from qtpynodeeditor import (
    DataModelRegistry,
    FlowScene,
    FlowView,
    NodeData,
    NodeDataModel,
    NodeDataType,
    PortType,
)


# Объект, который передаётся между нодами.
class TextData(NodeData):
    data_type = NodeDataType("text", "Текст")

    def __init__(self, value):
        self.value = value


# Источник: поле ввода → выходной порт.
class TextSource(NodeDataModel):
    name = "TextSource"
    caption = "Ввод текста"

    num_ports = {
        PortType.input: 0,
        PortType.output: 1,
    }
    data_type = TextData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)

        self.editor = QLineEdit("Привет!")
        self.editor.textChanged.connect(self.on_changed)

    def embedded_widget(self):
        return self.editor

    def out_data(self, port):
        return TextData(self.editor.text())

    def on_changed(self, text):
        # Изменились данные выходного порта №0.
        self.data_updated.emit(0)


# Приёмник: входной порт → надпись.
class TextDisplay(NodeDataModel):
    name = "TextDisplay"
    caption = "Результат"

    num_ports = {
        PortType.input: 1,
        PortType.output: 0,
    }
    data_type = TextData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)

        self.label = QLabel("Нет данных")
        self.label.setMinimumWidth(180)

    def embedded_widget(self):
        return self.label

    def set_in_data(self, data, port):
        # При отключении соединения может прийти None.
        self.label.setText(
            data.value if data is not None else "Нет данных"
        )


app = QApplication(sys.argv)

# Регистрируем доступные типы нод.
registry = DataModelRegistry()
registry.register_model(TextSource, category="Примеры")
registry.register_model(TextDisplay, category="Примеры")

# Сцена хранит граф, view отображает его.
scene = FlowScene(registry=registry)
view = FlowView(scene)

source = scene.create_node(TextSource)
display = scene.create_node(TextDisplay)

source.position = (-250, 0)
display.position = (100, 0)

# Выход №0 первой ноды → вход №0 второй.
scene.create_connection(
    source[PortType.output][0],
    display[PortType.input][0],
)

view.setWindowTitle("Пример qtpynodeeditor")
view.resize(900, 600)
view.show()

sys.exit(app.exec())