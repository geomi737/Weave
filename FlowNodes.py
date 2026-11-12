import random
from re import L

import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from sklearn.model_selection import train_test_split
from FlowTypes import DataframeData, SeriesData, TextData
from qtpynodeeditor import (
    NodeData,
    NodeDataModel,
    Port,
    PortType
)

from sklearn.preprocessing import OneHotEncoder

from PyQt6.QtWidgets import (QLineEdit,
                             QLabel,
                             QPushButton,
                             QFileDialog,
                             QVBoxLayout,
                             QHBoxLayout,
                             QWidget,
                             QComboBox,
                             QCheckBox,
                             QButtonGroup,
                             QRadioButton,
                             QDoubleSpinBox,
                             QSpinBox)


class ContainerModel(NodeDataModel, verify=False):
    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.embedded_widget_container, self.layout, self._status_layout = self._get_widget_container()
        self.status_label = QLabel()
        self._status_layout.addWidget(self.status_label)

    def _get_widget_container(self):
        embedded_widget_container = QWidget()
        embedded_widget_container.setObjectName("embeddedWidgetContainer")
        embedded_widget_container.setContentsMargins(0, 0, 0, 0)
        global_layout = QVBoxLayout(embedded_widget_container)
        usecase_widget = QWidget()
        layout = QVBoxLayout(usecase_widget)
        _status_widget = QWidget()
        _status_layout = QVBoxLayout(_status_widget)
        global_layout.addWidget(usecase_widget)
        global_layout.addWidget(_status_widget)
        return embedded_widget_container, layout, _status_layout

    def embedded_widget(self) -> QWidget:
        return self.embedded_widget_container


class TextInputNode(ContainerModel):
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


class TextOutputNode(ContainerModel):
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


class DatasetNode(ContainerModel):
    name = "Dataset"
    caption = "Dataset"

    num_ports = {
        PortType.input: 1,
        PortType.output: 1
    }

    data_type = DataframeData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.init_widget()
        self.dt = None

    def init_widget(self):
        self.load_btn = QPushButton()
        self.load_btn.setText("Load Dataset")
        self.load_btn.clicked.connect(self.load_dataset)

        self.layout.addWidget(self.load_btn)

    def embedded_widget(self) -> QWidget:
        return self.embedded_widget_container

    def out_data(self, port: int) -> NodeData:
        if self.dt is None:
            return None
        else:
            return DataframeData(self.dt)

    def set_in_data(self, node_data: DataframeData, port: Port):
        self.dt = node_data.value if node_data is not None else None
        self.load_btn.hide()
        self.data_updated.emit(0)

    def input_connection_deleted(self, connection):
        self.dt = None
        self.load_btn.show()
        self.load_btn.setText("Load Dataset")
        self.data_updated.emit(0)

    def load_dataset(self):
        if self.dt is None:
            path, _ = QFileDialog.getOpenFileName(caption="Choose dataset", directory="", filter="CSV (*.csv)")
            if path:
                self.dt = pd.read_csv(path)
                self.load_btn.setText("Unload Dataset")
                self.data_updated.emit(0)
        else:
            self.dt = None
            self.load_btn.setText("Load Dataset")
            self.data_updated.emit(0)

    def get_dataset(self):
        return self.dt


class DropNode(ContainerModel):
    name = "Drop Column"
    caption = "Drop Column"

    num_ports = {
        PortType.input: 1,
        PortType.output: 1
    }

    data_type = DataframeData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.init_widget()
        self.dt = None

    def init_widget(self):
        self.comboBox = QComboBox()
        self.comboBox.setFixedWidth(100)
        self.comboBox.activated.connect(lambda: self.data_updated.emit(0))
        self.layout.addWidget(self.comboBox)

        self.dropna_checkbox = QCheckBox()
        self.dropna_checkbox.setText("Drop all NA")
        self.dropna_checkbox.toggled.connect(self.dropna)
        self.layout.addWidget(self.dropna_checkbox)

    def set_in_data(self, node_data: DataframeData, port: Port):
        self.dt = node_data.value if node_data is not None else None
        self.comboBox.clear()
        if self.dt is not None:
            for column in self.dt.columns.to_list():
                self.comboBox.addItem(column)

    def update_data(self):
        self.data_updated.emit(0)

    def input_connection_deleted(self, connection):
        self.dt = None
        self.comboBox.clear()

    def out_data(self, port: int) -> NodeData:
        if self.dt is None:
            return None
        else:
            if self.dropna_checkbox.isChecked():
                return DataframeData(self.dt.drop(columns=self.dt.columns[self.dt.isna().any()].to_list()))
            return DataframeData(self.dt.drop(columns=self.comboBox.currentText()))

    def embedded_widget(self) -> QWidget:
        return self.embedded_widget_container

    def dropna(self, state):
        if state is True:
            self.comboBox.setDisabled(True)
        else:
            self.comboBox.setEnabled(True)
        self.data_updated.emit(0)


class SplitColumnNode(ContainerModel):
    name = "Split Column"
    caption = "Split Column"

    num_ports = {
        PortType.input: 1,
        PortType.output: 2
    }

    data_type = {
        PortType.input: {
            0: DataframeData.data_type
        },
        PortType.output: {
            0: DataframeData.data_type,
            1: SeriesData.data_type
        }
    }

    port_caption = {
        PortType.input: {
            0: "Dataframe"
        },
        PortType.output: {
            0: "Features",
            1: "Target"
        }
    }
    port_caption_visible = True

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.init_widget()
        self.dt = None

    def init_widget(self):
        self.comboBox = QComboBox()
        self.comboBox.setFixedWidth(100)
        self.comboBox.activated.connect(self.update_output)
        self.layout.addWidget(self.comboBox)

    def embedded_widget(self) -> QWidget:
        return super().embedded_widget()

    def update_output(self):
        self.data_updated.emit(0)
        self.data_updated.emit(1)

    def set_in_data(self, node_data: DataframeData, port: Port):
        self.dt = node_data.value if node_data is not None else None
        self.comboBox.clear()
        if self.dt is not None:
            for column in self.dt.columns.to_list():
                self.comboBox.addItem(column)
        self.update_output()

    def input_connection_deleted(self, connection):
        self.dt = None
        self.comboBox.clear()
        self.update_output()

    def out_data(self, port: int) -> NodeData:
        if self.dt is None:
            return None
        else:
            if port == 0:
                return DataframeData(self.dt.drop(columns=self.comboBox.currentText()))
            elif port == 1:
                return SeriesData(self.dt[self.comboBox.currentText()])


class SeriesNode(ContainerModel):
    name = "Series View"
    caption = "Series View"

    num_ports = {
        PortType.input: 1,
        PortType.output: 1
    }

    data_type = SeriesData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.dt = None

    def embedded_widget(self) -> QWidget:
        return super().embedded_widget()

    def set_in_data(self, node_data: SeriesData, port: Port):
        self.dt = node_data.value if node_data is not None else None
        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        return SeriesData(self.dt)

    def get_dataset(self):
        return self.dt.to_frame() if self.dt is not None else None


class BinaryReplaceNode(ContainerModel):
    name = "Binary Replace"
    caption = "Binary Replace"

    num_ports = {
        PortType.input: 1,
        PortType.output: 1
    }

    data_type = SeriesData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.init_widget()
        self.series = None

    def init_widget(self):
        self.label = QLabel()
        self.label.setText("Value to 1")
        self.comboBox = QComboBox()
        self.comboBox.activated.connect(lambda: self.data_updated.emit(0))
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.comboBox)

    def set_in_data(self, node_data: SeriesData, port: Port):
        self.series = node_data.value if node_data is not None else None
        self.comboBox.clear()
        if self.series is not None:
            for value in self.series.unique().tolist():
                self.comboBox.addItem(str(value))
        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        if self.series is None:
            return None
        else:
            if self.series.nunique() == 2:
                self.status_label.clear()
                result = self.series.astype(str)
                result = result.where(result != self.comboBox.currentText(), 1)
                result = result.where(result == 1, 0)
                return SeriesData(result)
            else:
                self.status_label.setText("Input contains more than 2 unique items")


class OneHotNode(ContainerModel):
    name = "OneHot"
    caption = "OneHot"

    num_ports = {
        PortType.input: 3,
        PortType.output: 3
    }

    data_type = DataframeData.data_type

    port_caption = {
        PortType.input: {0: "Train", 1: "Validate", 2: "Test"},
        PortType.output: {0: "Train", 1: "Validate", 2: "Test"},
    }
    port_caption_visible = True

    DROP_MODES = (None, "first", "if_binary")

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.datasets = [None, None, None]

        self.encoder = None
        self.init_widget()

    def init_widget(self):
        self.layout.addWidget(QLabel("Drop"))
        self.drop_group = QButtonGroup(self.embedded_widget_container)
        for index, text in enumerate(("No drop", "first", "if_binary")):
            button = QRadioButton(text)
            self.drop_group.addButton(button, index)
            self.layout.addWidget(button)
        self.drop_group.button(0).setChecked(True)
        self.drop_group.idClicked.connect(self.update_data)

        self.layout.addWidget(QLabel("Unknown"))
        self.unknown_group = QButtonGroup(self.embedded_widget_container)

        self.layout.addWidget(QLabel("min_frequency"))
        self.min_frequency_spinbox = QDoubleSpinBox()
        self.min_frequency_spinbox.setDecimals(2)
        self.min_frequency_spinbox.setRange(0, 1)
        self.min_frequency_spinbox.setSingleStep(0.01)

        self.layout.addWidget(self.min_frequency_spinbox)
        self.min_frequency_spinbox.valueChanged.connect(self.update_data)

        self.layout.addWidget(QLabel("max_categories"))
        self.max_categories_spinbox = QSpinBox()
        self.max_categories_spinbox.setRange(0, 10000000)
        self.layout.addWidget(self.max_categories_spinbox)
        self.max_categories_spinbox.valueChanged.connect(self.update_data)

    def set_in_data(self, node_data: DataframeData, port: Port):
        self.datasets[port.index] = node_data.value if node_data is not None else None
        self.update_data()

    def retrain(self):
        if self.datasets[0] is not None:
            self.encoder = OneHotEncoder(drop=self.DROP_MODES[self.drop_group.checkedId()],
                                         handle_unknown="infrequent_if_exist",
                                         max_categories=self.max_categories_spinbox.value() if self.max_categories_spinbox.value() else None,
                                         min_frequency=self.min_frequency_spinbox.value() if self.min_frequency_spinbox.value() else None,
                                         sparse_output=False).set_output(transform="pandas")
            train_obj: pd.DataFrame = self.datasets[0].copy()
            to_transform = train_obj.select_dtypes(exclude=['number']).columns.tolist()
            self.encoder.fit(train_obj[to_transform])

    def out_data(self, port: int) -> NodeData:
        if self.datasets[0] is None or self.datasets[port] is None or self.encoder is None:
            return None
        else:
            to_transform = self.datasets[0].select_dtypes(exclude=['number']).columns.tolist()
            to_return = self.datasets[port].copy()
            result = self.encoder.transform(to_return[to_transform])

            return DataframeData(pd.concat([to_return.drop(columns=to_transform), result], axis=1))

    def update_data(self):
        self.retrain()
        self.data_updated.emit(0)
        self.data_updated.emit(1)
        self.data_updated.emit(2)


class TVTSplitNode(ContainerModel):
    name = "TVT Split"
    caption = "Train Validate Test Split"

    num_ports = {
        PortType.input: 2,
        PortType.output: 6
    }

    data_type = {
        PortType.input: {
            0: DataframeData.data_type,
            1: SeriesData.data_type
        },
        PortType.output: {x: (DataframeData.data_type if x % 2 == 0 else SeriesData.data_type) for x in range(6)}
    }

    port_caption = {
        PortType.input: {
            0: "Features",
            1: "Target"},
        PortType.output: {
            0: "Train Features",
            1: "Train Target",
            2: "Validate Features",
            3: "Validate Target",
            4: "Test Features",
            5: "Test Target"},
    }

    port_caption_visible = True

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.init_widget()
        self.X: DataframeData | None = None
        self.y: SeriesData | None = None

    def init_widget(self):
        self.val_pct = QDoubleSpinBox()
        self.val_label = QLabel("Validate%")
        self.val_label.hide()
        self.val_pct.setValue(0.3)
        self.val_pct.setRange(0.01, 0.99)
        self.val_pct.setDecimals(2)
        self.val_pct.setSingleStep(0.01)
        self.val_pct.hide()
        self.val_pct.valueChanged.connect(self.update_output)

        self.test_pct = QDoubleSpinBox()
        self.test_pct.setValue(0.2)
        self.test_pct.setRange(0.01, 0.99)
        self.test_pct.setDecimals(2)
        self.test_pct.setSingleStep(0.01)
        self.test_pct.valueChanged.connect(self.update_output)

        self.include_validate = QCheckBox()
        self.include_validate.setText("Include validation")
        self.include_validate.stateChanged.connect(self.include_val_changed)

        self.layout.addWidget(self.include_validate)
        self.layout.addWidget(QLabel("Test%"))
        self.layout.addWidget(self.test_pct)
        self.layout.addWidget(self.val_label)
        self.layout.addWidget(self.val_pct)

    def set_in_data(self, node_data: DataframeData | SeriesData, port: Port):
        self.random_state = random.randint(1, 1000000)
        port = port.index
        dt = node_data.value if node_data is not None else None
        if port == 0:
            self.X = dt
        else:
            self.y = dt
        self.update_output()

    def out_data(self, port: int) -> NodeData:
        if self.X is None or self.y is None:
            return None
        elif (self.X.shape[0] != self.y.shape[-1]):
            self.status_label.setText("Shapes of features and target does not match")
        else:
            val_X, val_y = None, None
            train_X, test_X, train_y, test_y = train_test_split(
                self.X, self.y, test_size=self.test_pct.value(), random_state=self.random_state)
            if self.include_validate.isChecked():
                train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=self.val_pct.value(
                ) * (1 + self.test_pct.value()), random_state=self.random_state)
            match port:
                case 0:
                    return DataframeData(train_X)
                case 1:
                    return SeriesData(train_y)
                case 2:
                    return DataframeData(val_X)
                case 3:
                    return SeriesData(val_y)
                case 4:
                    return DataframeData(test_X)
                case 5:
                    return SeriesData(test_y)

    def update_output(self):
        for port in range(6):
            self.data_updated.emit(port)

    def include_val_changed(self, state):
        if state:
            self.val_pct.show()
            self.val_label.show()
        else:
            self.val_pct.hide()
            self.val_label.hide()
        self.update_output()


class Model(ContainerModel):
    name = "Model"
    caption = "Model"

    num_ports = {
        PortType.input: 2,
        PortType.output: 0
    }

    data_type = {
        PortType.input: {
            0: DataframeData.data_type,
            1: SeriesData.data_type
        }
    }

    port_caption = {
        PortType.input: {
            0: "Features",
            1: "Target"
        }
    }

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.layers = []
        self.init_widget()
        self.X = None
        self.y = None

    def set_in_data(self, node_data: DataframeData | SeriesData, port: Port):
        data = node_data.value if node_data else None
        port = port.index
        match port:
            case 0:
                self.X = data
            case 1:
                self.y = data

    def init_widget(self):
        self.lr_spinbox = QDoubleSpinBox(self.embedded_widget_container)
        self.lr_spinbox.setRange(0, 1)
        self.lr_spinbox.setDecimals(8)
        self.lr_spinbox.setValue(1e-4)
        self.add_linear_button = QPushButton("Add linear", self.embedded_widget_container)
        self.add_linear_button.clicked.connect(self.add_linear_layer)
        self.add_activation_button = QPushButton("Add activation", self.embedded_widget_container)
        self.add_activation_button.clicked.connect(self.add_activation_layer)

        self.layout.addWidget(self.lr_spinbox)
        self.layout.addWidget(self.add_linear_button)
        self.layout.addWidget(self.add_activation_button)
        self.layers_layout = QVBoxLayout()
        self.layout.addLayout(self.layers_layout)

    def add_linear_layer(self):
        layer_widget = QWidget()
        layer_layout = QVBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.addWidget(QLabel("Linear layer"))

        row_layout = QHBoxLayout()
        in_spinbox = QSpinBox()
        in_spinbox.setRange(1, 100000)
        in_spinbox.setValue(10)
        out_spinbox = QSpinBox()
        out_spinbox.setRange(1, 100000)
        out_spinbox.setValue(1)

        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet(
            "background-color: #c62828; color: white;"
        )
        row_layout.addWidget(in_spinbox, 1)
        row_layout.addWidget(out_spinbox, 1)
        row_layout.addWidget(delete_button)
        layer_layout.addLayout(row_layout)
        self.layers_layout.addWidget(layer_widget)

        layer = {
            "type": "linear",
            "widget": layer_widget,
            "in_spinbox": in_spinbox,
            "out_spinbox": out_spinbox,
        }
        self.layers.append(layer)
        delete_button.clicked.connect(
            lambda checked=False: self.remove_layer(layer)
        )
        self._update_layers_size()

    def add_activation_layer(self):
        layer_widget = QWidget()
        layer_layout = QVBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.addWidget(QLabel("Activation layer"))

        row_layout = QHBoxLayout()
        activation_combobox = QComboBox()
        activation_combobox.addItems([
            "ReLU", "GELU", "Sigmoid", "Tanh", "LeakyReLU",
            "ELU", "SiLU", "Softmax",
        ])
        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet(
            "background-color: #c62828; color: white;"
        )
        row_layout.addWidget(activation_combobox, 1)
        row_layout.addWidget(delete_button)
        layer_layout.addLayout(row_layout)
        self.layers_layout.addWidget(layer_widget)

        layer = {
            "type": "activation",
            "widget": layer_widget,
            "activation_combobox": activation_combobox,
        }
        self.layers.append(layer)
        delete_button.clicked.connect(
            lambda checked=False: self.remove_layer(layer)
        )
        self._update_layers_size()

    def remove_layer(self, layer):
        self.layers.remove(layer)
        self.layers_layout.removeWidget(layer["widget"])
        layer["widget"].hide()
        layer["widget"].deleteLater()
        self._update_layers_size()

    def _update_layers_size(self):
        # Let Qt recalculate the layout before updating the node geometry.
        QTimer.singleShot(0, self.embedded_widget_size_updated.emit)
