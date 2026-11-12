from enum import Enum, auto
import pandas as pd
from qtpynodeeditor import (
    NodeData,
    NodeDataType,
)


class Types(Enum):
    text = auto()
    dataframe = auto()
    series = auto()

class TextData(NodeData):
    data_type = NodeDataType(Types.text, "Text")

    def __init__(self, value: str) -> None:
        self.value = value


class DataframeData(NodeData):
    data_type = NodeDataType(Types.dataframe, "Dataframe")

    def __init__(self, value: pd.DataFrame) -> None:
        self.value = value

class SeriesData(NodeData):
    data_type = NodeDataType(Types.series, "Series")

    def __init__(self, value: pd.Series) -> None:
        self.value = value
