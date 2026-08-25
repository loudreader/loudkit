"""Transports: three adapters over one synthesis surface.

HTTP (with SSE streaming and the OpenAI-compatible route), MCP over stdio,
and gRPC. Each imports the same ``loudkit.synthesis`` symbols and each is a
peer — none of them imports another, and none of them synthesises audio
itself. Layering rule enforced by ``tests/test_import_graph.py``.
"""

from __future__ import annotations
