"""Обработка соединений и удаления нод поверх qtpynodeeditor."""

from contextlib import contextmanager
from weakref import WeakSet

from qtpynodeeditor import DataModelRegistry, FlowScene

from FlowNodes import Model


class WeaveRegistry(DataModelRegistry):
    """Упорядочивает ветки и названия в меню создания нод."""

    def categories(self):
        return sorted(
            super().categories(),
            key=lambda name: (name != "preprocessing", name.casefold()),
        )

    def registered_models_category_association(self):
        return dict(
            sorted(
                super().registered_models_category_association().items(),
                key=lambda item: item[0].casefold(),
            )
        )


class WeaveFlowScene(FlowScene):
    """Сцена с исправлением повторной подписки в qtpynodeeditor 0.3.3."""

    def __init__(self, *args, **kwargs):
        self._configured_connections = WeakSet()
        super().__init__(*args, **kwargs)

    @contextmanager
    def _new_node_context(self, data_model_name, **kwargs):
        with super()._new_node_context(data_model_name, **kwargs) as node:
            yield node
            if isinstance(node.model, Model):
                # При загрузке UUID восстанавливается внутри yield.
                node.model.caption = (
                    f"{type(node.model).caption} [{node.id[:8]}]"
                )
                node.on_node_size_updated()

    def remove_node(self, node):
        # После удаления геометрии отложенные сигналы ноды уже не нужны.
        node.model.blockSignals(True)
        super().remove_node(node)

    def _setup_connection_signals(self, connection):
        # При переподключении connection_created приходит снова.
        # Подписка сохраняется: повторный UniqueConnection недопустим.
        if connection in self._configured_connections:
            return
        super()._setup_connection_signals(connection)
        self._configured_connections.add(connection)
