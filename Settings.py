"""Общие настройки устройства вычислений и диалог их выбора."""

import platform

import torch
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QDialog, QMessageBox

from settings_ui import Ui_SettingsDialog

BACKENDS = ("CPU", "CUDA", "ROCm")


def detect_backends():
    """Проверяет доступность GPU с учётом сборки PyTorch."""
    available = {name: False for name in BACKENDS}
    available["CPU"] = True
    if torch.cuda.is_available():
        available["ROCm" if torch.version.hip else "CUDA"] = True
    return available


def device_for_backend(backend):
    if not detect_backends().get(backend, False):
        raise ValueError(
            f"{backend} is unavailable in this PyTorch environment"
        )
    # ROCm в PyTorch использует имя устройства 'cuda', как и CUDA.
    return torch.device("cpu" if backend == "CPU" else "cuda:0")


class AppSettings:
    """Хранит выбранное устройство отдельно от файла проекта."""

    def __init__(self, storage=None):
        self.storage = (
            storage if storage is not None else QSettings("Weave", "Weave")
        )
        self.available = detect_backends()

    @property
    def backend(self):
        # На другой машине сохранённый GPU может быть недоступен.
        saved = self.storage.value("compute/backend", "CPU", type=str)
        return saved if self.available.get(saved, False) else "CPU"

    def set_backend(self, backend):
        if not self.available.get(backend, False):
            raise ValueError(f"{backend} is unavailable")
        self.storage.setValue("compute/backend", backend)
        self.storage.sync()
        if self.storage.status() != QSettings.Status.NoError:
            raise OSError("Could not save application settings")


class SettingsDialog(QDialog, Ui_SettingsDialog):
    """Предлагает доступные в текущей среде устройства."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.settings = settings
        for backend in BACKENDS:
            self.backend_combo.addItem(backend, backend)
            item = self.backend_combo.model().item(
                self.backend_combo.count() - 1
            )
            enabled = settings.available[backend]
            item.setEnabled(enabled)
            if not enabled:
                item.setToolTip("Unavailable in this PyTorch environment")
        self.backend_combo.setCurrentText(settings.backend)
        self.backend_combo.currentIndexChanged.connect(self.update_info)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.update_info()

    def update_info(self, *_args):
        backend = self.backend_combo.currentData()
        if backend == "CPU":
            description = platform.processor() or platform.machine()
        elif self.settings.available.get(backend, False):
            try:
                description = torch.cuda.get_device_name(0)
            except RuntimeError, AssertionError:
                description = "Device information unavailable"
        else:
            description = "Unavailable in this PyTorch environment"
        self.device_info.setText(f"{backend}: {description}")

    def accept(self):
        try:
            self.settings.set_backend(self.backend_combo.currentData())
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Settings error", str(error))
            return
        super().accept()
