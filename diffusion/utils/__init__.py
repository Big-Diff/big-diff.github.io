"""Utility package for diffusion modules.

Decoder modules are imported explicitly by their callers. Keeping package
initialisation lightweight avoids loading deprecated decoder files during
unrelated imports.
"""

__all__ = []