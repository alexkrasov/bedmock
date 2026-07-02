"""Drop-in facade for supported boto3 Bedrock Runtime calls."""

from . import session
from .facade import client
from .session import Session
from .version import __version__

__all__ = ["Session", "__version__", "client", "session"]
