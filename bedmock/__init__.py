"""Public Bedmock facade.

Bedmock keeps `bedrock_bridge` as a compatibility namespace and exposes the same
drop-in Bedrock Runtime facade from the shorter project name.
"""

from bedrock_bridge import Session, __version__, client, session

__all__ = ["Session", "__version__", "client", "session"]
