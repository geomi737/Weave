"""Оформление нод, соединений и сетки редактора Weave."""

from PyQt6.QtGui import QColor
from qtpynodeeditor.style import ConnectionStyle, StyleCollection

from FlowTypes import DataframeData, SeriesData


class WeaveConnectionStyle(ConnectionStyle):
    """Цвета портов и проводов по идентификатору типа данных."""

    type_colors = {
        DataframeData.data_type.id: "#ad96f5",
        SeriesData.data_type.id: "#72d4a1",
    }

    def get_normal_color(self, type_id=None) -> QColor:
        if type_id is None:
            return super().get_normal_color()
        return QColor(self.type_colors.get(type_id, "#9ba4bb"))


def create_flow_style() -> StyleCollection:
    """Создать независимый стиль редактора в цветах style.qss."""
    style = StyleCollection(connection=WeaveConnectionStyle())

    # Фон и сетка.
    style.flow_view.background_color = QColor("#171a22")
    style.flow_view.fine_grid_color = QColor("#202430")
    style.flow_view.coarse_grid_color = QColor("#303544")

    # Однотонный корпус ноды, рамка и тень.
    style.node.gradient_colors = (
        (0.0, QColor("#232733")),
        (1.0, QColor("#232733")),
    )
    style.node.normal_boundary_color = QColor("#474e65")
    style.node.selected_boundary_color = QColor("#ad96f5")
    style.node.shadow_color = QColor("#101218")
    style.node.opacity = 1.0
    style.node.pen_width = 1.0
    style.node.hovered_pen_width = 1.5

    # Текст и порты.
    style.node.font_color = QColor("#e5e7ef")
    style.node.font_color_faded = QColor("#9ba4bb")
    style.node.connection_point_diameter = 8.0
    style.node.connection_point_color = QColor("#747b90")
    style.node.filled_connection_point_color = QColor("#ad96f5")

    # Цвет портов и проводов определяется типом данных.
    style.connection.use_data_defined_colors = True
    style.connection.normal_color = QColor("#ad96f5")
    style.connection.construction_color = QColor("#747b90")
    style.connection.selected_color = QColor("#d3c9ff")
    style.connection.selected_halo_color = QColor("#655187")
    style.connection.hovered_color = QColor("#d3c9ff")
    style.connection.line_width = 2.0

    return style
