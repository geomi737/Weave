import sys

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QHeaderView
import pandas as pd
from FlowNodes import (
    BinaryReplaceNode, DatasetNode, DropNode, Model, OneHotNode, SeriesNode,
    SplitColumnNode, TVTSplitNode, TextInputNode, TextOutputNode,
)
from FlowTypes import DataframeData
from FlowStyle import create_flow_style
from FlowScene import WeaveFlowScene
from ui import Ui_MainWindow
from qtpynodeeditor import (
    DataModelRegistry,
    FlowView,
    NodeData,
    NodeDataModel,
    NodeDataType,
    PortType
)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.betterUi()
        self.setupNode()

    def betterUi(self):
        self.datasetViewerAction = self.datasetDock.toggleViewAction()
        self.datasetViewerAction.setObjectName("Dataset Viewer")
        self.menuView.addAction(self.datasetViewerAction)

    # Node Functions
    def setupNode(self):
        style = create_flow_style()
        registry = DataModelRegistry()
        registry.register_model(TextInputNode, style=style)
        registry.register_model(TextOutputNode, style=style)
        registry.register_model(DatasetNode, style=style)
        registry.register_model(DropNode, style=style)
        registry.register_model(SplitColumnNode, style=style)
        registry.register_model(SeriesNode, style=style)
        registry.register_model(BinaryReplaceNode, style=style)
        registry.register_model(OneHotNode, style=style)
        registry.register_model(TVTSplitNode, style=style)
        registry.register_model(Model, style=style)

        self.flowScene = WeaveFlowScene(registry, style=style)

        # This line is needed for autoscroll turnoff
        self.flowView = FlowView(self.flowScene, self.nodeEditTab)
        self.flowView.setSceneRect(-100000, -100000, 200000, 200000)

        self.flowScene.selectionChanged.connect(self.node_focus_changed_or_updated)
        self.flowScene.node_created.connect(self.on_node_created)

        self.nodeEditTab.layout().addWidget(self.flowView)

    def on_node_created(self, node):
        node.model.data_updated.connect(lambda _port: self.node_focus_changed_or_updated())

    def node_focus_changed_or_updated(self):
        nodes = self.flowScene.selected_nodes()
        node = nodes[0] if len(nodes) == 1 else None

        if node:
            model = node.model
            if isinstance(model, (DatasetNode, SeriesNode)):
                dt = model.get_dataset()
                self.dataset_preview(dt)
                self.dataset_analysis(dt)
            else:
                self.clear_dataset_preview()
        else:
            self.clear_dataset_preview()

    # Dataset Preview Functions
    def dataset_preview(self, dataset: pd.DataFrame):
        if dataset is None:
            self.clear_dataset_preview()
            return

        dataset = dataset.head(100)
        row_count = dataset.shape[0]
        column_count = dataset.shape[1]

        #! Without turning autoresize off program will lag on big amounts of columns
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        headers = [str(column) for column in dataset.columns]
        self.tableWidget.setRowCount(row_count)
        self.tableWidget.setColumnCount(column_count)
        self.tableWidget.setHorizontalHeaderLabels(headers)

        dt = dataset.to_numpy()

        for r, row in enumerate(dt):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.tableWidget.setItem(r, c, item)

        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

    def dataset_analysis(self, dataset: pd.DataFrame):
        if dataset is None:
            self.datasetAnalysis.clear()
            return

        columns = dataset.shape[1]
        rows = dataset.shape[0]
        na_values = dataset.isna().sum().to_frame("NA in column")
        describe = (
            dataset.describe([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            if columns else pd.DataFrame()
        )
        uniques = dataset.nunique().to_frame("Number of unique items")
        types = dataset.dtypes.to_frame("Types")

        text = f"""<h1>Size</h1>
        <h3>Columns: {columns}</h3>
        <h3>Rows: {rows}</h3>

        <h1>NA Values</h1>
        <p>{na_values.to_html()}</p>

        <h1>Describe</h1>
        <p>{describe.to_html()}</p>

        <h1>Unique Items</h1>
        <p>{uniques.to_html()}</p>

        <h1>Types</h1>
        <p>{types.to_html()}<p>"""

        self.datasetAnalysis.setText(text)

    def clear_dataset_preview(self):
        self.tableWidget.clear()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(0)

        self.datasetAnalysis.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open("style.qss") as f:
        app.setStyleSheet(f.read())
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec())
