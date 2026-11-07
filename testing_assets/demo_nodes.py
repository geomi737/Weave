"""Визуальный эксперимент qtpynodeeditor + PyQt6. Без обучения и обработки CSV.

Запуск: .venv/bin/python demo_nodes.py
Снимок: QT_QPA_PLATFORM=offscreen .venv/bin/python demo_nodes.py --screenshot /tmp/nodes.png
"""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPen
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QVBoxLayout, QWidget,
)
from qtpynodeeditor import (
    DataModelRegistry, FlowScene, FlowView, NodeData, NodeDataModel,
    NodeDataType, PortType,
)
from qtpynodeeditor.style import Style, StyleCollection


def make_graph_style():
    """Это стиль сцены библиотеки, а не QSS обычных виджетов."""
    theme = deepcopy(Style.default_style)
    theme["NodeStyle"].update({
        "NormalBoundaryColor": "#3a4253",
        "SelectedBoundaryColor": "#a99bff",
        **{f"GradientColor{i}": "#202632" for i in range(4)},
        "ShadowColor": [0, 0, 0, 45],
        "FontColor": "#e8edf5",
        "FontColorFaded": "#98a5b9",
        "ConnectionPointColor": "#8d7cf6",
        "FilledConnectionPointColor": "#c2b6ff",
        "Opacity": 1.0,
        "PenWidth": 1.0,
        "HoveredPenWidth": 1.5,
        "ConnectionPointDiameter": 6.0,
    })
    theme["FlowViewStyle"].update({
        "BackgroundColor": "#141820",
        "FineGridColor": "#242b38",
        "CoarseGridColor": "#242b38",
    })
    theme["ConnectionStyle"].update({
        "NormalColor": "#9b8af6",
        "SelectedColor": "#d3caff",
        "SelectedHaloColor": "#61518b",
        "HoveredColor": "#c2b6ff",
        "LineWidth": 2.0,
        "UseDataDefinedColors": False,
    })
    return StyleCollection.from_json(theme)


class Canvas(FlowView):
    """Спокойная точечная сетка вместо двух слоёв линий."""

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#141820"))
        painter.setPen(QPen(QColor("#303746"), 1.5))
        step = 28
        left = int(rect.left() // step) * step
        top = int(rect.top() // step) * step
        for x in range(left, int(rect.right()) + step, step):
            for y in range(top, int(rect.bottom()) + step, step):
                painter.drawPoint(x, y)


class PreviewData(NodeData):
    # Пустая подпись убирает лишний текст рядом с портом.
    data_type = NodeDataType("preview_table", "")

    def __init__(self, path, steps=()):
        self.path = path
        self.steps = steps


class CardNode(NodeDataModel):
    name = "Card"
    caption_visible = False
    num_ports = {PortType.input: 1, PortType.output: 1}
    data_type = PreviewData.data_type
    title = "Operation"
    subtitle = "Transform your data"
    accent = "#a99bff"

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self.incoming = None
        self.panel = QWidget()
        self.panel.setObjectName("nodePanel")
        self.panel.setFixedWidth(240)
        self.layout = QVBoxLayout(self.panel)
        self.layout.setContentsMargins(10, 14, 10, 14)
        self.layout.setSpacing(12)
        title = QLabel(self.title)
        title.setObjectName("nodeTitle")
        title.setStyleSheet(f"color: {self.accent};")
        self.layout.addWidget(title)
        subtitle = QLabel(self.subtitle)
        subtitle.setObjectName("muted")
        self.layout.addWidget(subtitle)
        self.build_controls()
        self.status = QLabel("Connect an input")
        self.status.setObjectName("nodeStatus")
        self.layout.addWidget(self.status)
        self.panel.adjustSize()

    def build_controls(self):
        pass

    def field(self, label, widget):
        caption = QLabel(label)
        caption.setObjectName("fieldLabel")
        self.layout.addWidget(caption)
        self.layout.addWidget(widget)

    def embedded_widget(self):
        return self.panel

    def set_in_data(self, data, port):
        self.incoming = data
        self.refresh()

    def out_data(self, port):
        if self.incoming is None:
            return None
        return PreviewData(
            self.incoming.path, self.incoming.steps + (self.name,)
        )

    def refresh(self, *args):
        self.status.setText("●  Input connected" if self.incoming else "○  No input")
        self.data_updated.emit(0)


class DatasetNode(CardNode):
    name = "Dataset"
    title = "01   Dataset"
    subtitle = "DATA SOURCE  /  CSV"
    accent = "#78d8ba"
    num_ports = {PortType.input: 0, PortType.output: 1}

    def build_controls(self):
        self.path = QLineEdit("sample.csv")
        self.path.setPlaceholderText("Path to a CSV file")
        self.field("FILE", self.path)
        browse = QPushButton("Choose CSV…")
        browse.clicked.connect(self.browse)
        self.layout.addWidget(browse)
        self.target = QLineEdit("target")
        self.field("TARGET COLUMN", self.target)
        self.dropna = QCheckBox("Drop rows with missing values")
        self.dropna.setChecked(True)
        self.layout.addWidget(self.dropna)
        self.path.textChanged.connect(self.refresh)

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self.panel, "Choose dataset", "", "CSV files (*.csv)"
        )
        if path:
            self.path.setText(path)

    def out_data(self, port):
        path = self.path.text().strip()
        return PreviewData(path) if path else None

    def refresh(self, *args):
        self.status.setText("●  Source configured" if self.path.text().strip() else "○  Choose a file")
        self.data_updated.emit(0)


class MinMaxNode(CardNode):
    name = "MinMax"
    title = "03   MinMax scaler"
    subtitle = "PREPROCESSING  /  NUMERIC"

    def build_controls(self):
        columns = QLineEdit("age, income")
        self.field("COLUMNS", columns)
        scale = QComboBox()
        scale.addItems(["0 → 1", "−1 → 1"])
        self.field("OUTPUT RANGE", scale)
        note = QLabel("Fit on train\nApply to validation and test")
        note.setObjectName("note")
        self.layout.addWidget(note)


class SplitNode(CardNode):
    name = "Split"
    title = "02   Train / validation / test"
    subtitle = "DATA SPLIT  /  REPRODUCIBLE"
    accent = "#e9b97e"

    def build_controls(self):
        fractions = QFrame()
        fractions.setObjectName("fractions")
        row = QHBoxLayout(fractions)
        for text in ("TRAIN\n70%", "VAL\n15%", "TEST\n15%"):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(label)
        self.layout.addWidget(fractions)
        seed = QLineEdit("42")
        self.field("RANDOM STATE", seed)
        stratify = QCheckBox("Stratify by target")
        stratify.setChecked(True)
        self.layout.addWidget(stratify)


def build_window():
    window = QMainWindow()
    window.setWindowTitle("Pipeline — qtpynodeeditor style demo")
    window.resize(1260, 720)
    root = QWidget()
    layout = QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    header = QFrame()
    header.setObjectName("header")
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(28, 20, 28, 20)
    brand = QLabel("PIPELINE  /  Playground")
    brand.setObjectName("brand")
    header_layout.addWidget(brand)
    header_layout.addStretch()
    badge = QLabel("PyQt6  ·  qtpynodeeditor")
    badge.setObjectName("muted")
    header_layout.addWidget(badge)
    layout.addWidget(header)

    graph_style = make_graph_style()
    registry = DataModelRegistry()
    for model in (DatasetNode, SplitNode, MinMaxNode):
        registry.register_model(model, category="Data", style=graph_style)
    scene = FlowScene(registry=registry, style=graph_style)
    view = Canvas(scene)
    layout.addWidget(view, 1)
    footer = QLabel(
        "   STYLE DEMO — CSV is not read; controls preview the UI only."
        "     Drag nodes · Wheel to zoom · Right-click to add"
    )
    footer.setObjectName("footer")
    footer.setMinimumHeight(44)
    layout.addWidget(footer)
    window.setCentralWidget(root)

    # Правильная последовательность для будущей обработки: split до scaler.
    source, split, scaler = [
        scene.create_node(model)
        for model in (DatasetNode, SplitNode, MinMaxNode)
    ]
    for node, x, y in ((source, -540, -150), (split, -165, -100), (scaler, 210, -50)):
        node.position = (x, y)
        node.graphics_object.setGraphicsEffect(None)
    for left, right in ((source, split), (split, scaler)):
        scene.create_connection(left[PortType.output][0], right[PortType.input][0])
    source.model.refresh()

    # Реальная проверка передачи данных и очистки результата при пустом входе.
    assert scaler.model.out_data(0).path == "sample.csv"
    source.model.path.setText("")
    assert scaler.model.out_data(0) is None
    source.model.path.setText("sample.csv")
    assert scaler.model.out_data(0).steps == ("Split", "MinMax")

    # Сохраняем ссылки на сцену и view на всё время жизни окна.
    window.graph_scene = scene
    window.graph_view = view
    return window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setFont(QFont("DejaVu Sans", 10))
    app.setStyleSheet(Path(__file__).with_name("demo_nodes.qss").read_text())
    window = build_window()
    window.show()

    def frame_graph():
        view = window.graph_view
        bounds = window.graph_scene.itemsBoundingRect().adjusted(-65, -85, 65, 85)
        view.setSceneRect(bounds)
        view.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        if args.screenshot:
            if not window.grab().save(str(args.screenshot)):
                print("Could not save screenshot", file=sys.stderr)
                app.exit(1)
                return
            print(f"Screenshot: {args.screenshot}")
            app.quit()

    QTimer.singleShot(200, frame_graph)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
