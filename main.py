"""Главное окно, действия с проектом и просмотр данных."""

import sys
from pathlib import Path
from weakref import WeakSet

import pandas as pd
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication,
    QHeaderView,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
)
from qtpynodeeditor import (
    FlowView,
)

from FlowNodes import (
    BinaryReplaceNode,
    ClassificationModel,
    DatasetNode,
    DropNode,
    Model,
    OneHotNode,
    RegressionModel,
    SeriesNode,
    SplitColumnNode,
    TVTSplitNode,
)
from FlowScene import WeaveFlowScene, WeaveRegistry
from FlowStyle import create_flow_style
from ProjectLoader import prepare_project
from ProjectStorage import capture_project, read_project, write_project
from Settings import AppSettings, SettingsDialog
from TrainingUI import TrainingPanel
from ui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    """Связывает граф, просмотр данных, обучение и файлы проекта."""

    def __init__(self, settings=None) -> None:
        super().__init__()
        self.settings = settings if settings is not None else AppSettings()
        self.project_path = None
        self._watched_controls = WeakSet()
        self._clearing = False
        self.setupUi(self)
        self.configure_ui()
        self.setup_nodes()
        self.init_training_panel()
        self.actionNew.triggered.connect(self.new_project)
        self.actionOpen.triggered.connect(self.open_project)
        self.actionOpen.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.actionSave.triggered.connect(self.save_project)
        self.actionSave_as.triggered.connect(self.save_project_as)
        self.actionSave.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.actionSave_as.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)
        self.connect_project_signals()
        self.update_project_title()
        self.menuEdit = QtWidgets.QMenu("Edit", self)
        self.menubar.insertMenu(self.menuView.menuAction(), self.menuEdit)
        self.actionSettings = self.menuEdit.addAction("Settings...")
        self.actionSettings.setShortcut(QtGui.QKeySequence("Ctrl+,"))
        self.actionSettings.triggered.connect(self.open_settings)
        self.statusbar.showMessage("Ready")

    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            backend = self.settings.backend
            for card in self.training_panel.cards.values():
                card.set_compute_backend(backend)
            self.statusbar.showMessage(f"Compute device: {backend}")
        dialog.deleteLater()

    def connect_project_signals(self):
        for signal in (
            self.flowScene.node_created,
            self.flowScene.node_deleted,
            self.flowScene.connection_created,
            self.flowScene.connection_deleted,
            self.flowScene.node_moved,
        ):
            signal.connect(self.mark_modified)

    def new_project(self):
        answer = QtWidgets.QMessageBox.warning(
            self,
            "New project",
            "All nodes and training results will be deleted. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._clearing = True
            try:
                self.flowScene.clear_scene()
                self.clear_dataset_preview()
            finally:
                self._clearing = False
            self.project_path = None
            self.setWindowModified(False)
            self.update_project_title()
            self.statusbar.showMessage("New project")

    def init_training_panel(self):
        self.training_panel = TrainingPanel()
        self.trainingTab.layout().addWidget(self.training_panel)
        self.flowScene.node_deleted.connect(self.training_panel.remove_node)

    def closeEvent(self, event):
        if self.isWindowModified():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "Save changes before closing?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Save,
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if (
                answer == QtWidgets.QMessageBox.StandardButton.Save
                and not self.save_project(closing=True)
            ):
                event.ignore()
                return
        self.training_panel.shutdown()
        super().closeEvent(event)

    def update_project_title(self):
        name = self.project_path.name if self.project_path else "Untitled"
        self.setWindowTitle(f"{name}[*] — Weave")

    def mark_modified(self, *_args):
        if not self._clearing:
            self.setWindowModified(True)
            self.statusbar.showMessage("Unsaved changes")

    def watch_controls(self, widget):
        """Отслеживает изменения в том числе у новых виджетов слоёв."""
        for control in [widget, *widget.findChildren(QtWidgets.QWidget)]:
            if control in self._watched_controls:
                continue
            if isinstance(control, (QtWidgets.QLineEdit, QtWidgets.QComboBox)):
                signal = (
                    control.textChanged
                    if isinstance(control, QtWidgets.QLineEdit)
                    else control.currentIndexChanged
                )
            elif isinstance(
                control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
            ):
                signal = control.valueChanged
            elif (
                isinstance(control, QtWidgets.QAbstractButton)
                and control.isCheckable()
            ):
                signal = control.toggled
            else:
                continue
            signal.connect(self.mark_modified)
            self._watched_controls.add(control)

    def open_project(self, checked=False):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open project",
            str(self.project_path.parent if self.project_path else ""),
            "Weave projects (*.weave)",
        )
        if not path:
            return False
        if self.isWindowModified():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Unsaved changes",
                "Save changes before opening another project?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Save,
            )
            if answer == QtWidgets.QMessageBox.StandardButton.Cancel or (
                answer == QtWidgets.QMessageBox.StandardButton.Save
                and not self.save_project()
            ):
                self.statusbar.showMessage("Open cancelled")
                return False
        self.statusbar.showMessage("Opening project...")
        try:
            path = Path(path).resolve()
            if path.suffix.lower() != ".weave":
                raise ValueError("Use the .weave project extension")
            # Текущий проект остаётся на месте до проверки нового файла.
            prepared = prepare_project(
                read_project(path),
                self.flowScene.registry,
                self.flowScene.style_collection,
            )
        except Exception as error:
            self.statusbar.showMessage("Open failed")
            QtWidgets.QMessageBox.warning(self, "Open error", str(error))
            return False
        # Служебная замена графа не считается пользовательской правкой.
        self._clearing = True
        try:
            old_scene, old_panel, old_view = (
                self.flowScene,
                self.training_panel,
                self.flowView,
            )
            old_panel.shutdown()
            old_scene.clear_scene()
            self.flowScene, self.training_panel = prepared
            self.flowScene.setParent(self)
            self.flowView = FlowView(self.flowScene, self.nodeEditTab)
            self.flowView.setSceneRect(-100000, -100000, 200000, 200000)
            self.nodeEditTab.layout().replaceWidget(old_view, self.flowView)
            self.trainingTab.layout().replaceWidget(
                old_panel, self.training_panel
            )
            old_view.hide()
            old_panel.hide()
            old_view.deleteLater()
            old_panel.deleteLater()
            old_scene.deleteLater()
            self.flowScene.selectionChanged.connect(
                self.node_focus_changed_or_updated
            )
            self.flowScene.node_created.connect(self.on_node_created)
            self.connect_project_signals()
            for node in self.flowScene.nodes.values():
                self.on_node_created(node)
            self.clear_dataset_preview()
            self.project_path = path
        finally:
            self._clearing = False
        self.setWindowModified(False)
        self.update_project_title()
        self.statusbar.showMessage(f"Opened: {path.name}")
        return True

    def save_project(self, checked=False, *, closing=False):
        if self.project_path is None:
            return self.save_project_as(closing=closing)
        return self.write_project_file(self.project_path, closing=closing)

    def save_project_as(self, checked=False, *, closing=False):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save project as",
            str(self.project_path or "Untitled.weave"),
            "Weave projects (*.weave)",
        )
        if not path:
            return False
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".weave")
        return self.write_project_file(path, closing=closing)

    def write_project_file(self, path, *, closing=False):
        self.statusbar.showMessage("Saving project...")
        self.statusbar.repaint()
        try:
            path = Path(path).resolve()
            if path.suffix.lower() != ".weave":
                raise ValueError("Use the .weave project extension")
            # Веса считываются на границе батча и не меняются во время записи.
            with self.training_panel.paused_for_snapshot():
                snapshot = capture_project(self.flowScene, self.training_panel)
                write_project(path, *snapshot)
                if closing:
                    # После сохранения перед выходом обучение не возобновляем.
                    self.training_panel.shutdown()
        except Exception as error:
            self.statusbar.showMessage("Save failed")
            QtWidgets.QMessageBox.warning(self, "Save error", str(error))
            return False
        self.project_path = path
        self.setWindowModified(False)
        self.update_project_title()
        running = any(
            card.training_thread is not None
            and not card.pause_training_btn.isChecked()
            for card in self.training_panel.cards.values()
        )
        if running and not closing:
            self.setWindowModified(True)
        suffix = " — training continues" if running and not closing else ""
        self.statusbar.showMessage(f"Saved: {path.name}{suffix}")
        return True

    def configure_ui(self):
        self.datasetViewerAction = self.datasetDock.toggleViewAction()
        self.datasetViewerAction.setObjectName("Dataset Viewer")
        self.menuView.addAction(self.datasetViewerAction)

    # Создание графа и обработчики нод.
    def setup_nodes(self):
        style = create_flow_style()
        registry = WeaveRegistry()
        for node_type in (
            DatasetNode,
            DropNode,
            SplitColumnNode,
            SeriesNode,
            BinaryReplaceNode,
            OneHotNode,
            TVTSplitNode,
        ):
            registry.register_model(
                node_type, category="preprocessing", style=style
            )
        for node_type in (RegressionModel, ClassificationModel):
            registry.register_model(node_type, category="models", style=style)

        self.flowScene = WeaveFlowScene(registry, style=style)

        # Большая область сцены предотвращает автосдвиг границ редактора.
        self.flowView = FlowView(self.flowScene, self.nodeEditTab)
        self.flowView.setSceneRect(-100000, -100000, 200000, 200000)

        self.flowScene.selectionChanged.connect(
            self.node_focus_changed_or_updated
        )
        self.flowScene.node_created.connect(self.on_node_created)

        self.nodeEditTab.layout().addWidget(self.flowView)

    def on_node_created(self, node):
        node.model.data_updated.connect(
            lambda _port: self.node_focus_changed_or_updated()
        )
        node.model.data_updated.connect(self.mark_modified)
        self.watch_controls(node.model.embedded_widget())
        node.model.embedded_widget_size_updated.connect(
            lambda: self.watch_controls(node.model.embedded_widget())
        )
        node.model.embedded_widget_size_updated.connect(self.mark_modified)
        if isinstance(node.model, Model):
            card = self.training_panel.cards.get(
                node.id
            ) or self.training_panel.add_node(node)
            card.set_compute_backend(self.settings.backend)
            card.project_changed.connect(self.mark_modified)
            self.watch_controls(card)

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

    # Просмотр данных выбранной ноды.
    def dataset_preview(self, dataset: pd.DataFrame):
        if dataset is None:
            self.clear_dataset_preview()
            return

        dataset = dataset.head(100)
        row_count = dataset.shape[0]
        column_count = dataset.shape[1]

        header = self.tableWidget.horizontalHeader()
        # Отключаем автоподбор ширины во время заполнения большой таблицы.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        headers = [str(column) for column in dataset.columns]
        self.tableWidget.setRowCount(row_count)
        self.tableWidget.setColumnCount(column_count)
        self.tableWidget.setHorizontalHeaderLabels(headers)
        self.tableWidget.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

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
            if columns
            else pd.DataFrame()
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
