import sys

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow
from FlowNodes import TextInputNode, TextOutputNode, TextTransformNode
from ui import Ui_MainWindow
from qtpynodeeditor import (
    DataModelRegistry,
    FlowScene,
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

    def setupNode(self):
        registry = DataModelRegistry()
        registry.register_model(TextInputNode)
        registry.register_model(TextOutputNode)
        registry.register_model(TextTransformNode)
        
        self.flowScene = FlowScene(registry)
        self.flowView = FlowView(self.flowScene, self.nodeEditTab)
        self.nodeEditTab.layout().addWidget(self.flowView)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec())
