from __future__ import annotations

import bedmock


def test_public_facade_exports_runtime_api() -> None:
    assert isinstance(bedmock.__version__, str)
    assert callable(bedmock.client)
    assert bedmock.Session.__name__ == "Session"
