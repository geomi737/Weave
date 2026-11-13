"""Ноды подготовки данных и настройки архитектуры нейросетей."""

import random
from math import ceil

import pandas as pd
from PyQt6.QtCore import QSignalBlocker, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qtpynodeeditor import NodeData, NodeDataModel, Port, PortType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target

from FlowTypes import DataframeData, ModelTypes, SeriesData
from ModelArchive import ACTIVATIONS, build_network


def refill_combo(combo, values):
    """Сохраняет выбор и тип значения при обновлении списка."""
    previous, had_items = combo.currentData(), combo.count() > 0
    # Заполнение списка не должно запускать пересчёт с пустым выбором.
    with QSignalBlocker(combo):
        combo.clear()
        for value in values:
            combo.addItem(str(value), value)
        combo.setCurrentIndex(combo.findData(previous) if had_items else 0)


class ContainerModel(NodeDataModel, verify=False):
    """Общая оболочка виджета ноды с областью статуса."""

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.embedded_widget_container, self.layout, self._status_layout = (
            self._get_widget_container()
        )
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


class DatasetNode(ContainerModel):
    """Загружает CSV или отображает входящий DataFrame."""

    name = "Dataset"
    caption = "Dataset"

    num_ports = {PortType.input: 1, PortType.output: 1}

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
            path, _ = QFileDialog.getOpenFileName(
                caption="Choose dataset", directory="", filter="CSV (*.csv)"
            )
            if path:
                try:
                    self.dt = pd.read_csv(path)
                except (OSError, ValueError, UnicodeError) as error:
                    QMessageBox.warning(
                        self.embedded_widget_container, "CSV error", str(error)
                    )
                    return
                self.load_btn.setText("Unload Dataset")
                self.data_updated.emit(0)
        else:
            self.dt = None
            self.load_btn.setText("Load Dataset")
            self.data_updated.emit(0)

    def get_dataset(self):
        return self.dt


class DropNode(ContainerModel):
    """Удаляет выбранный столбец или столбцы с пропусками."""

    name = "Drop Column"
    caption = "Drop Column"

    num_ports = {PortType.input: 1, PortType.output: 1}

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
        refill_combo(self.comboBox, [] if self.dt is None else self.dt.columns)
        self.update_data()

    def update_data(self):
        self.data_updated.emit(0)

    def input_connection_deleted(self, connection):
        self.set_in_data(None, None)

    def out_data(self, port: int) -> NodeData:
        if self.dt is None:
            return None
        else:
            if self.dropna_checkbox.isChecked():
                return DataframeData(
                    self.dt.drop(
                        columns=self.dt.columns[self.dt.isna().any()].to_list()
                    )
                )
            if self.comboBox.currentIndex() < 0:
                return None
            return DataframeData(
                self.dt.drop(columns=self.comboBox.currentData())
            )

    def embedded_widget(self) -> QWidget:
        return self.embedded_widget_container

    def dropna(self, state):
        if state is True:
            self.comboBox.setDisabled(True)
        else:
            self.comboBox.setEnabled(True)
        self.data_updated.emit(0)


class SplitColumnNode(ContainerModel):
    """Отделяет целевую переменную от таблицы признаков."""

    name = "Split Column"
    caption = "Split Column"

    num_ports = {PortType.input: 1, PortType.output: 2}

    data_type = {
        PortType.input: {0: DataframeData.data_type},
        PortType.output: {0: DataframeData.data_type, 1: SeriesData.data_type},
    }

    port_caption = {
        PortType.input: {0: "Dataframe"},
        PortType.output: {0: "Features", 1: "Target"},
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
        refill_combo(self.comboBox, [] if self.dt is None else self.dt.columns)
        self.update_output()

    def input_connection_deleted(self, connection):
        self.set_in_data(None, None)

    def out_data(self, port: int) -> NodeData:
        if self.dt is None or self.comboBox.currentIndex() < 0:
            return None
        else:
            if port == 0:
                return DataframeData(
                    self.dt.drop(columns=self.comboBox.currentData())
                )
            elif port == 1:
                return SeriesData(self.dt[self.comboBox.currentData()])


class SeriesNode(ContainerModel):
    name = "Series View"
    caption = "Series View"

    num_ports = {PortType.input: 1, PortType.output: 1}

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
        return SeriesData(self.dt) if self.dt is not None else None

    def get_dataset(self):
        return self.dt.to_frame() if self.dt is not None else None


class BinaryReplaceNode(ContainerModel):
    """Кодирует два непустых значения цели числами 0 и 1."""

    name = "Binary Replace"
    caption = "Binary Replace"

    num_ports = {PortType.input: 1, PortType.output: 1}

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
        refill_combo(
            self.comboBox,
            [] if self.series is None else self.series.unique().tolist(),
        )
        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        if self.series is None:
            return None
        if self.series.nunique() != 2 or self.series.isna().any():
            self.status_label.setText(
                "Input must contain exactly two non-missing values"
            )
            return None
        if self.comboBox.currentIndex() < 0:
            return None
        self.status_label.clear()
        result = (self.series == self.comboBox.currentData()).astype("int64")
        return SeriesData(result)


class OneHotNode(ContainerModel):
    """Обучает кодировщик на Train и преобразует другие выборки."""

    name = "OneHot"
    caption = "OneHot"

    num_ports = {PortType.input: 3, PortType.output: 3}

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
        self.min_frequency_spinbox.setRange(0, 10000000)
        self.min_frequency_spinbox.setToolTip(
            "0: disabled; 0–1: fraction; integers >= 1: count"
        )
        self.min_frequency_spinbox.setSingleStep(0.01)

        self.layout.addWidget(self.min_frequency_spinbox)
        self.min_frequency_spinbox.valueChanged.connect(self.update_data)

        self.layout.addWidget(QLabel("max_categories"))
        self.max_categories_spinbox = QSpinBox()
        self.max_categories_spinbox.setRange(0, 10000000)
        self.layout.addWidget(self.max_categories_spinbox)
        self.max_categories_spinbox.valueChanged.connect(self.update_data)

    def set_in_data(self, node_data: DataframeData, port: Port):
        self.datasets[port.index] = (
            node_data.value if node_data is not None else None
        )
        # Проверочные выборки не участвуют в обучении кодировщика.
        self.update_data(retrain=port.index == 0)

    def retrain(self):
        self.encoder = None
        if self.datasets[0] is not None:
            frequency = self.min_frequency_spinbox.value()
            if frequency >= 1:
                if not frequency.is_integer():
                    raise ValueError(
                        "min_frequency must be a fraction below 1 "
                        "or an integer count"
                    )
                # Для sklearn число 1.0 является недопустимой долей.
                frequency = int(frequency)
            self.encoder = OneHotEncoder(
                drop=self.DROP_MODES[self.drop_group.checkedId()],
                handle_unknown="infrequent_if_exist",
                max_categories=self.max_categories_spinbox.value()
                if self.max_categories_spinbox.value()
                else None,
                min_frequency=frequency or None,
                sparse_output=False,
            ).set_output(transform="pandas")
            train_obj: pd.DataFrame = self.datasets[0].copy()
            to_transform = train_obj.select_dtypes(
                exclude=["number"]
            ).columns.tolist()
            if to_transform:
                self.encoder.fit(train_obj[to_transform])

    def out_data(self, port: int) -> NodeData:
        if (
            self.datasets[0] is None
            or self.datasets[port] is None
            or self.encoder is None
        ):
            return None
        else:
            to_transform = (
                self.datasets[0]
                .select_dtypes(exclude=["number"])
                .columns.tolist()
            )
            # Train обновляется раньше других входов при смене столбцов.
            # До их обновления несовместимые выходы должны быть пустыми.
            train_columns = self.datasets[0].columns
            if set(self.datasets[port].columns) != set(train_columns):
                self.status_label.setText(
                    f"{self.port_caption[PortType.input][port]}: "
                    "columns differ from Train"
                )
                return None
            # Одинаковый порядок признаков нужен всем входам модели.
            to_return = (
                self.datasets[port].reindex(columns=train_columns).copy()
            )
            if not to_transform:
                return DataframeData(to_return)
            # Пустая выборка сохраняет схему, но не передаётся в transform.
            if to_return.empty:
                result = pd.DataFrame(
                    index=to_return.index,
                    columns=self.encoder.get_feature_names_out(),
                    dtype=float,
                )
            else:
                result = self.encoder.transform(to_return[to_transform])

            return DataframeData(
                pd.concat(
                    [to_return.drop(columns=to_transform), result], axis=1
                )
            )

    def update_data(self, *_args, retrain=True):
        try:
            if retrain:
                self.retrain()
            self.status_label.setText("Ready")
        except (ValueError, TypeError) as error:
            self.encoder = None
            self.status_label.setText(f"Train: {error}")
        self.data_updated.emit(0)
        self.data_updated.emit(1)
        self.data_updated.emit(2)


class TVTSplitNode(ContainerModel):
    """Хранит согласованное разбиение признаков и цели."""

    name = "TVT Split"
    caption = "Train Validate Test Split"

    num_ports = {PortType.input: 2, PortType.output: 6}

    data_type = {
        PortType.input: {0: DataframeData.data_type, 1: SeriesData.data_type},
        PortType.output: {
            x: (
                DataframeData.data_type if x % 2 == 0 else SeriesData.data_type
            )
            for x in range(6)
        },
    }

    port_caption = {
        PortType.input: {0: "Features", 1: "Target"},
        PortType.output: {
            0: "Train Features",
            1: "Train Target",
            2: "Validate Features",
            3: "Validate Target",
            4: "Test Features",
            5: "Test Target",
        },
    }

    port_caption_visible = True

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.init_widget()
        self.X = None
        self.y = None
        self.outputs = [None] * 6
        self.random_state = random.randint(1, 1000000)
        self.split_version = 2

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
        if not self.signalsBlocked():
            self.split_version = 2
        self.random_state = random.randint(1, 1000000)
        port = port.index
        dt = node_data.value if node_data is not None else None
        if port == 0:
            self.X = dt
        else:
            self.y = dt
        self.update_output()

    def out_data(self, port: int) -> NodeData:
        return self.outputs[port]

    def split_datasets(self):
        if self.X is None or self.y is None:
            return [None] * 6
        # Парные входы приходят последовательно и временно могут различаться.
        if len(self.X) != len(self.y) or not self.X.index.equals(self.y.index):
            raise ValueError(
                "Features and target must have matching rows and indexes"
            )
        if len(self.X) < 2 or self.y.isna().any():
            raise ValueError(
                "Use at least two rows and a target without missing values"
            )
        test_size = self.test_pct.value()
        val_size = (
            self.val_pct.value() if self.include_validate.isChecked() else 0
        )
        if test_size + val_size >= 1:
            raise ValueError(
                "Sum of Test and Validation fractions must be below 1"
            )
        unstratified = False

        # Редкие классы и маленькие выборки могут исключать стратификацию.
        def split(features, target, size):
            nonlocal unstratified
            stratify = None
            if self.split_version >= 2 and type_of_target(target) in (
                "binary",
                "multiclass",
            ):
                counts = target.value_counts()
                test_rows = ceil(len(target) * size)
                if counts.min() >= 2 and min(
                    test_rows, len(target) - test_rows
                ) >= len(counts):
                    stratify = target
                else:
                    unstratified = True
            return train_test_split(
                features,
                target,
                test_size=size,
                random_state=self.random_state,
                stratify=stratify,
            )

        train_X, test_X, train_y, test_y = split(self.X, self.y, test_size)
        val_X, val_y = None, None
        if val_size:
            # Validation задаётся от всей выборки, а отделяется от остатка.
            # Старый алгоритм сохранён для ранее записанных проектов.
            fraction = (
                val_size / (1 - test_size)
                if self.split_version >= 2
                else val_size * (1 + test_size)
            )
            train_X, val_X, train_y, val_y = split(train_X, train_y, fraction)
        self.status_label.setText(
            "Without stratification: too few samples per class"
            if unstratified
            else "Ready"
        )
        return [
            DataframeData(train_X),
            SeriesData(train_y),
            DataframeData(val_X) if val_X is not None else None,
            SeriesData(val_y) if val_y is not None else None,
            DataframeData(test_X),
            SeriesData(test_y),
        ]

    def update_output(self, *_args):
        if _args and not self.signalsBlocked():
            self.split_version = 2
        # Ошибка разбиения не должна оставлять на выходах старые данные.
        self.outputs = [None] * 6
        self.status_label.clear()
        try:
            self.outputs = self.split_datasets()
        except (ValueError, TypeError) as error:
            self.status_label.setText(str(error))
        for port in range(6):
            self.data_updated.emit(port)

    def include_val_changed(self, state):
        if not self.signalsBlocked():
            self.split_version = 2
        if state:
            self.val_pct.show()
            self.val_label.show()
        else:
            self.val_pct.hide()
            self.val_label.hide()
        self.update_output()


class Model(ContainerModel):
    """Хранит архитектуру сети и данные с шести входных портов."""

    name = "Model"
    caption = "Model"
    task = None

    num_ports = {PortType.input: 6, PortType.output: 0}

    data_type = {
        PortType.input: {
            0: DataframeData.data_type,
            1: SeriesData.data_type,
            2: DataframeData.data_type,
            3: SeriesData.data_type,
            4: DataframeData.data_type,
            5: SeriesData.data_type,
        }
    }

    port_caption = {
        PortType.input: {
            0: "Train Features",
            1: "Train Target",
            2: "Validation Features",
            3: "Validation Target",
            4: "Test Features",
            5: "Test Target",
        }
    }
    port_caption_visible = True

    def __init__(self, style=None, parent=None):
        super().__init__(style, parent)
        self.layers = []
        # Несколько изменений слоёв объединяются в один пересчёт геометрии.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(
            self.embedded_widget_size_updated.emit
        )
        self.datasets = [None] * 6
        self.init_widget()

    def set_in_data(self, node_data: DataframeData | SeriesData, port: Port):
        self.datasets[port.index] = (
            node_data.value if node_data is not None else None
        )

    def init_widget(self):
        self.add_linear_button = QPushButton(
            "Add linear", self.embedded_widget_container
        )
        self.add_linear_button.clicked.connect(self.add_linear_layer)
        self.add_activation_button = QPushButton(
            "Add activation", self.embedded_widget_container
        )
        self.add_activation_button.clicked.connect(self.add_activation_layer)

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
        delete_button.setStyleSheet("background-color: #c62828; color: white;")
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
        activation_combobox.addItems(ACTIVATIONS)
        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet("background-color: #c62828; color: white;")
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

    def set_training_active(self, active):
        self.add_linear_button.setEnabled(not active)
        self.add_activation_button.setEnabled(not active)

        for layer in self.layers:
            layer["widget"].setEnabled(not active)

    def remove_layer(self, layer):
        self.layers.remove(layer)
        self.layers_layout.removeWidget(layer["widget"])
        layer["widget"].hide()
        layer["widget"].deleteLater()
        self._update_layers_size()

    def _update_layers_size(self):
        self._resize_timer.start(0)

    def get_architecture(self):
        """Описывает слои обычными словарями для сохранения."""
        return [
            {
                "type": "linear",
                "in_features": layer["in_spinbox"].value(),
                "out_features": layer["out_spinbox"].value(),
            }
            if layer["type"] == "linear"
            else {
                "type": "activation",
                "name": layer["activation_combobox"].currentText(),
            }
            for layer in self.layers
        ]

    def replace_architecture(self, architecture):
        for layer in list(self.layers):
            self.remove_layer(layer)
        for layer in architecture:
            if layer["type"] == "linear":
                self.add_linear_layer()
                self.layers[-1]["in_spinbox"].setValue(layer["in_features"])
                self.layers[-1]["out_spinbox"].setValue(layer["out_features"])
            else:
                self.add_activation_layer()
                self.layers[-1]["activation_combobox"].setCurrentText(
                    layer["name"]
                )

    def build_network(self):
        return build_network(self.get_architecture())


class RegressionModel(Model):
    name = "Regression Model"
    caption = "Regression Model"
    task = ModelTypes.regression


class ClassificationModel(Model):
    name = "Classification Model"
    caption = "Classification Model"
    task = ModelTypes.classification
