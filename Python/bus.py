"""
Socket.io client shared by every producer (watcher, depth worker, seg worker).

Producers are fire-and-forget: if the relay is down the mission must not stop,
so emit() never raises and connect() retries in the background. The pipeline
being visualised is the important process here; the GUI is an observer.
"""

import logging
import os
import threading

import socketio

log = logging.getLogger("bus")

DEFAULT_URL = os.environ.get("GUI_URL", "http://127.0.0.1:7100")


class Bus:
    def __init__(self, role, url=DEFAULT_URL):
        self.role = role
        self.url = url
        self._sio = socketio.Client(
            reconnection=True,
            reconnection_delay=1,
            reconnection_delay_max=10,
            ssl_verify=False,          # the relay's cert is self-signed in the field
            logger=False,
            engineio_logger=False,
        )
        self._lock = threading.Lock()
        self._handlers = {}

        # Re-sent on every (re)connect. connect() is asynchronous, so a producer
        # that emits its opening state immediately would emit into a socket that
        # is not up yet and that state would simply be lost — the relay would
        # then serve a snapshot with logs and GIDs but no mission. The same
        # replay covers a relay restart mid-mission, which otherwise leaves the
        # console with no run identity for the rest of the run.
        self._hello = None
        self._on_connect = []
        self._connected = threading.Event()

        @self._sio.event(namespace="/py")
        def connect():
            log.info("connected to relay at %s as %s", self.url, self.role)
            self._connected.set()

            if self._hello:
                event, data = self._hello
                if self.emit(event, data):
                    log.info("replayed %s to relay", event)

            for fn in self._on_connect:
                try:
                    fn()
                except Exception:
                    log.exception("on_connect handler failed")

        @self._sio.event(namespace="/py")
        def disconnect():
            log.warning("disconnected from relay")
            self._connected.clear()

    def on_connect(self, fn):
        """
        Run `fn` after every successful (re)connect, once the hello is sent.

        For re-publishing state that was emitted while the socket was down.
        emit() is a no-op when disconnected, so anything sent before the relay
        was up is simply gone unless a producer replays it here.
        """
        self._on_connect.append(fn)

    @property
    def connected(self):
        return self._connected.is_set()

    def wait_connected(self, timeout=None):
        """Block until connected. Returns False on timeout."""
        return self._connected.wait(timeout)

    def set_hello(self, event, data):
        """
        Declare the producer's opening state.

        Emitted now if connected, and again on every reconnect. Use it for
        anything the relay must know to interpret later events — the mission
        identity, a worker's configuration — as opposed to a stream item.
        """
        self._hello = (event, data)
        self.emit(event, data)

    def on(self, event, fn):
        """Register a command handler (replay_control, run_worker, ...)."""
        self._handlers[event] = fn
        self._sio.on(event, fn, namespace="/py")

    def connect(self, block=False):
        def _go():
            try:
                self._sio.connect(
                    self.url,
                    namespaces=["/py"],
                    auth={"role": self.role},
                    wait_timeout=10,
                )
            except Exception as exc:
                # socketio's own reconnect loop only runs after a successful
                # first connect, so a relay that is not up yet needs this.
                log.warning("relay not reachable (%s) — retrying in background", exc)
                threading.Timer(3.0, _go).start()

        if block:
            _go()
        else:
            threading.Thread(target=_go, daemon=True).start()

    def emit(self, event, data):
        # Guard on our own flag, not sio.connected: python-socketio only sets
        # that AFTER every connect handler has returned, so anything emitted
        # from inside on_connect — the hello, a producer's state resync — would
        # be dropped here without a word. self._connected is set at the top of
        # the handler, which is what "may I send?" actually means.
        with self._lock:
            if not self._connected.is_set():
                return False
            try:
                self._sio.emit(event, data, namespace="/py")
                return True
            except Exception as exc:
                log.debug("emit %s failed: %s", event, exc)
                return False

    def wait(self):
        self._sio.wait()

    def close(self):
        try:
            self._sio.disconnect()
        except Exception:
            pass
