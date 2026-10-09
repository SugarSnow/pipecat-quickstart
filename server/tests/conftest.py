"""A stand-in flow manager, shared by the tests that drive the callback flow.

The flow's decisions are made in plain functions that read and write
``flow_manager.state``, look at ``current_node``, and speak fixed lines through
``flow_manager.worker``. None of that needs a pipeline, so the tests hand them
this instead and read back what was said.
"""

import asyncio
from types import SimpleNamespace

import pytest


class FakeWorker:
    """Records the text of every frame queued on it."""

    def __init__(self):
        self.spoken: list[str] = []

    async def queue_frames(self, frames):
        self.spoken.extend(getattr(frame, "text", repr(frame)) for frame in frames)


@pytest.fixture
def flow():
    """A flow in the node the details are first collected in."""
    return make_flow()


def make_flow(node="callback_collect", **state):
    """Return a stand-in flow manager.

    Args:
        node: What ``current_node`` should report.
        state: The flow state to start from.
    """
    return SimpleNamespace(state=dict(state), current_node=node, worker=FakeWorker())


def run(coro):
    """Run one coroutine to completion, for tests that are otherwise sync."""
    return asyncio.run(coro)
