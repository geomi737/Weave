from enum import Enum, auto
from qtpynodeeditor import (
    DataModelRegistry,
    FlowScene,
    FlowView,
    NodeData,
    NodeDataModel,
    NodeDataType,
    PortType
)


class Types(Enum):
    text = auto()


class TextData(NodeData):
    data_type = NodeDataType(Types.text, "Text Output")

    def __init__(self, value) -> None:
        self.value = value
