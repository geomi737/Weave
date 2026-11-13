import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication, QMessageBox

from FlowNodes import ClassificationModel, RegressionModel
from main import MainWindow


class ModelLabelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        with patch(
            "main.QtWidgets.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Discard,
        ):
            self.window.close()
        self.window.flowScene.clear_scene()
        self.app.processEvents()

    def test_node_and_training_titles_use_the_same_short_id(self):
        for node_type in (
            ClassificationModel,
            ClassificationModel,
            RegressionModel,
        ):
            node = self.window.flowScene.create_node(node_type)
            expected = f"{node_type.caption} [{node.id[:8]}]"
            self.assertEqual(node.model.caption, expected)
            self.assertEqual(
                self.window.training_panel.cards[node.id].title(), expected
            )
            self.assertEqual(node.model.name, node_type.name)
        self.assertEqual(ClassificationModel.caption, "Classification Model")
        self.assertEqual(RegressionModel.caption, "Regression Model")

    def test_groups_and_node_names_are_ordered(self):
        registry = self.window.flowScene.registry
        self.assertEqual(registry.categories(), ["preprocessing", "models"])
        categories = registry.registered_models_category_association()
        self.assertEqual(
            list(categories), sorted(categories, key=str.casefold)
        )
        for name, category in categories.items():
            expected = (
                "models"
                if name in (ClassificationModel.name, RegressionModel.name)
                else "preprocessing"
            )
            self.assertEqual(category, expected)

    def test_short_ids_survive_save_and_open(self):
        node = self.window.flowScene.create_node(ClassificationModel)
        title = node.model.caption
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.weave"
            self.assertTrue(self.window.write_project_file(path))
            with patch(
                "main.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(str(path), ""),
            ):
                self.assertTrue(self.window.open_project())
        restored = self.window.flowScene.nodes[node.id]
        self.assertEqual(restored.model.caption, title)
        self.assertEqual(
            self.window.training_panel.cards[node.id].title(), title
        )
        self.assertEqual(
            self.window.flowScene.registry.categories(),
            ["preprocessing", "models"],
        )
