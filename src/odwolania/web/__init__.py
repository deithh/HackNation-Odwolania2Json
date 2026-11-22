"""Lightweight local Web GUI (stdlib ThreadingHTTPServer + PDF field highlighting + Zero-Code Type Studio)."""
from .evidence import EvidenceLocator
from .server import WebServer
from .service import WebDocument, WebWorkspaceService

__all__ = [
    "EvidenceLocator",
    "WebDocument",
    "WebServer",
    "WebWorkspaceService",
]
