from __future__ import annotations

import bedmock
import bedrock_bridge


def test_bedmock_alias_exposes_facade() -> None:
    assert bedmock.__version__ == bedrock_bridge.__version__
    assert bedmock.client is bedrock_bridge.client
    assert bedmock.Session is bedrock_bridge.Session
