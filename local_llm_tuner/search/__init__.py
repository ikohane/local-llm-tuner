"""Pluggable search backends for the lit-search phase.

Implement SearchBackend to add a new source. Two built-ins:
  - NullSearch: no-op; used when you don't want external lookups.
  - PubMedSearch: NCBI E-utilities client (biomedical).
"""

from .base import SearchBackend, NullSearch
from .pubmed import PubMedSearch

__all__ = ["SearchBackend", "NullSearch", "PubMedSearch"]
