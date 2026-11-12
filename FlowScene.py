from weakref import WeakSet

from qtpynodeeditor import FlowScene


class WeaveFlowScene(FlowScene):
    """Сцена с исправлением повторной подписки в qtpynodeeditor 0.3.3."""

    def __init__(self, *args, **kwargs):
        self._configured_connections = WeakSet()
        super().__init__(*args, **kwargs)

    def _setup_connection_signals(self, connection):
        # При переподключении того же провода connection_created приходит снова.
        # Подписка сохраняется, поэтому повторять connect с UniqueConnection нельзя.
        if connection in self._configured_connections:
            return
        super()._setup_connection_signals(connection)
        self._configured_connections.add(connection)
