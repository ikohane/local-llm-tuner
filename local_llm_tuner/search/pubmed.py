"""PubMed E-utilities search backend.

No API key required for the default 3 req/s rate. Sanitizes queries by
stripping common small-model mistakes (hyphenated compounds, date-range
operators, stray quotation marks).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

logger = logging.getLogger("local_llm_tuner.search.pubmed")

_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_HEADERS = {"User-Agent": "local-llm-tuner/0.1 (research)"}


def sanitize_query(q: str) -> str:
    """Clean a query emitted by a small LLM into PubMed-friendly form."""
    s = q.strip()
    # Replace hyphens between letters with spaces (RAG-based -> RAG based)
    s = re.sub(r"(?<=\w)-(?=\w)", " ", s)
    # Drop date-range patterns: "2023..2025" or "2023-2025"
    s = re.sub(r"\b(19|20)\d{2}\s*(?:\.{2,}|-|to)\s*(19|20)\d{2}\b", "", s)
    # Strip quotation marks
    s = re.sub(r"[\"'`]+", "", s)
    # Collapse whitespace, strip trailing punctuation
    s = re.sub(r"\s+", " ", s).strip(" .,;:")
    return s


class PubMedSearch:
    """NCBI PubMed E-utilities client.

    Args:
        api_key: optional NCBI API key for higher rate limits (10 req/s).
        max_retries: how many times to retry on HTTP 429.
        abstract_max_chars: truncate long abstracts to fit small-LLM prompts.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        abstract_max_chars: int = 1500,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.abstract_max_chars = abstract_max_chars

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        cleaned = sanitize_query(query)
        if cleaned != query:
            logger.info(f"Sanitized query: {query!r} -> {cleaned!r}")
        pmids = self._retry(lambda: self._esearch(cleaned, max_results))
        if not pmids:
            return []
        time.sleep(0.4 if not self.api_key else 0.1)
        return self._retry(lambda: self._efetch(pmids))

    def format_for_prompt(self, hits: list[dict]) -> str:
        if not hits:
            return "(no results returned)"
        lines = []
        for i, r in enumerate(hits, 1):
            header = f"[{i}] {r.get('first_author','')} ({r.get('year','')}) — {r.get('journal','')}"
            lines.append(header.strip())
            if r.get("title"):
                lines.append(f"    Title: {r['title']}")
            if r.get("abstract"):
                lines.append(f"    Abstract: {r['abstract']}")
            lines.append("")
        return "\n".join(lines)

    # ---------- internal ---------- #

    def _retry(self, fn):
        for attempt in range(self.max_retries):
            try:
                return fn()
            except HTTPError as e:
                if e.code == 429 and attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"PubMed 429, backing off {wait}s")
                    time.sleep(wait)
                    continue
                raise
        return []

    def _esearch(self, query: str, max_results: int) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{_BASE}/esearch.fcgi?" + urlencode(params)
        req = Request(url, headers=_HEADERS)
        with urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("esearchresult", {}).get("idlist", []) or []

    def _efetch(self, pmids: list[str]) -> list[dict]:
        params: dict = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{_BASE}/efetch.fcgi?" + urlencode(params)
        req = Request(url, headers=_HEADERS)
        with urlopen(req, timeout=30) as r:
            xml_text = r.read().decode("utf-8", errors="replace")

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"PubMed XML parse error: {e}")
            return []

        results = []
        for art in root.findall(".//PubmedArticle"):
            title_el = art.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            year = ""
            for xp in (".//ArticleDate/Year", ".//PubDate/Year", ".//PubDate/MedlineDate"):
                el = art.find(xp)
                if el is not None and el.text:
                    year = el.text.strip()[:4]
                    break

            journal = ""
            for xp in (".//Journal/Title", ".//Journal/ISOAbbreviation"):
                el = art.find(xp)
                if el is not None and el.text:
                    journal = el.text.strip()
                    break

            first_author = ""
            author_el = art.find(".//AuthorList/Author")
            if author_el is not None:
                last = author_el.findtext("LastName") or ""
                initials = author_el.findtext("Initials") or ""
                first_author = f"{last} {initials}".strip()

            parts = []
            for at in art.findall(".//Abstract/AbstractText"):
                label = at.get("Label")
                txt = "".join(at.itertext()).strip()
                if txt:
                    parts.append(f"{label}: {txt}" if label else txt)
            abstract = " ".join(parts)
            if len(abstract) > self.abstract_max_chars:
                abstract = abstract[: self.abstract_max_chars].rsplit(" ", 1)[0] + "..."

            pmid_el = art.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            results.append({
                "pmid": pmid,
                "first_author": first_author,
                "year": year,
                "journal": journal,
                "title": title,
                "abstract": abstract,
            })
        return results
