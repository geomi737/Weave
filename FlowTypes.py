"""Типы значений портов и виды задач машинного обучения."""

from enum import Enum, auto

import pandas as pd
from qtpynodeeditor import (
    NodeData,
    NodeDataType,
)


class NodeTypes(Enum):
    """Определяет совместимость портов при создании соединения."""

    text = auto()
    dataframe = auto()
    series = auto()


class ModelTypes(Enum):
    """Различает функции потерь и метрики для двух видов задач."""

    regression = auto()
    classification = auto()


class TextData(NodeData):
    data_type = NodeDataType(NodeTypes.text, "Text")

    def __init__(self, value: str) -> None:
        self.value = value


class DataframeData(NodeData):
    data_type = NodeDataType(NodeTypes.dataframe, "Dataframe")

    def __init__(self, value: pd.DataFrame) -> None:
        self.value = value


class SeriesData(NodeData):
    data_type = NodeDataType(NodeTypes.series, "Series")

    def __init__(self, value: pd.Series) -> None:
        self.value = value
