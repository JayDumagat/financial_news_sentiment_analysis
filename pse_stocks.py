"""
Philippine Stock Exchange (PSE) listed companies database and news impact matcher.

Provides a lookup of major PSE-listed companies keyed by their ticker symbol,
along with alternate name keywords and the sector they belong to.

The :func:`find_affected_stocks` function scans news text and returns a list
of stocks that may be impacted, either through a direct mention of the company /
ticker or through sector-level macro triggers (e.g., an interest-rate cut news
item affects all listed banks).
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type definition
# ---------------------------------------------------------------------------


class StockEntry(TypedDict):
    name: str
    keywords: list[str]
    sector: str


# ---------------------------------------------------------------------------
# PSE company database (major index / blue-chip companies)
# ---------------------------------------------------------------------------

PSE_COMPANIES: dict[str, StockEntry] = {
    # --- Holding Firms ---
    "SM": {
        "name": "SM Investments Corporation",
        "keywords": ["SM Investments", "SM Group", "SM Corp", "Henry Sy"],
        "sector": "Holding Firms",
    },
    "AC": {
        "name": "Ayala Corporation",
        "keywords": ["Ayala Corporation", "Ayala Corp", "Ayala Group"],
        "sector": "Holding Firms",
    },
    "AEV": {
        "name": "Aboitiz Equity Ventures",
        "keywords": ["Aboitiz Equity", "Aboitiz Group"],
        "sector": "Holding Firms",
    },
    "AGI": {
        "name": "Alliance Global Group",
        "keywords": ["Alliance Global", "Andrew Tan"],
        "sector": "Holding Firms",
    },
    "DMC": {
        "name": "DMCI Holdings",
        "keywords": ["DMCI Holdings", "DMCI", "Consunji"],
        "sector": "Holding Firms",
    },
    "MPI": {
        "name": "Metro Pacific Investments Corporation",
        "keywords": ["Metro Pacific Investments", "MPIC", "Manuel Pangilinan"],
        "sector": "Holding Firms",
    },
    "LTG": {
        "name": "LT Group",
        "keywords": ["LT Group", "Lucio Tan"],
        "sector": "Holding Firms",
    },
    "FPH": {
        "name": "First Philippine Holdings Corporation",
        "keywords": ["First Philippine Holdings", "First Gen", "Lopez Group"],
        "sector": "Holding Firms",
    },
    # --- Financials ---
    "BDO": {
        "name": "BDO Unibank",
        "keywords": ["BDO Unibank", "Banco de Oro"],
        "sector": "Financials",
    },
    "MBT": {
        "name": "Metropolitan Bank & Trust",
        "keywords": ["Metrobank", "Metropolitan Bank"],
        "sector": "Financials",
    },
    "BPI": {
        "name": "Bank of the Philippine Islands",
        "keywords": ["Bank of the Philippine Islands"],
        "sector": "Financials",
    },
    "SECB": {
        "name": "Security Bank Corporation",
        "keywords": ["Security Bank"],
        "sector": "Financials",
    },
    "EW": {
        "name": "East West Banking Corporation",
        "keywords": ["EastWest Bank", "East West Bank"],
        "sector": "Financials",
    },
    "PNB": {
        "name": "Philippine National Bank",
        "keywords": ["Philippine National Bank"],
        "sector": "Financials",
    },
    "RCB": {
        "name": "Rizal Commercial Banking Corporation",
        "keywords": ["RCBC", "Rizal Commercial Banking"],
        "sector": "Financials",
    },
    # --- Property ---
    "ALI": {
        "name": "Ayala Land",
        "keywords": ["Ayala Land"],
        "sector": "Property",
    },
    "SMPH": {
        "name": "SM Prime Holdings",
        "keywords": ["SM Prime", "SM Mall"],
        "sector": "Property",
    },
    "RLC": {
        "name": "Robinsons Land Corporation",
        "keywords": ["Robinsons Land", "Robinsons Malls"],
        "sector": "Property",
    },
    "MEG": {
        "name": "Megaworld Corporation",
        "keywords": ["Megaworld"],
        "sector": "Property",
    },
    "DD": {
        "name": "DoubleDragon Properties",
        "keywords": ["DoubleDragon", "Double Dragon Properties"],
        "sector": "Property",
    },
    "VLL": {
        "name": "Vista Land & Lifescapes",
        "keywords": ["Vista Land", "Villar"],
        "sector": "Property",
    },
    # --- Services ---
    "JFC": {
        "name": "Jollibee Foods Corporation",
        "keywords": ["Jollibee", "Chowking", "Greenwich", "Red Ribbon", "Mang Inasal"],
        "sector": "Services",
    },
    "MER": {
        "name": "Manila Electric Company",
        "keywords": ["Meralco", "Manila Electric"],
        "sector": "Services",
    },
    "TEL": {
        "name": "PLDT Inc.",
        "keywords": ["PLDT", "Philippine Long Distance", "Smart Communications"],
        "sector": "Services",
    },
    "GLO": {
        "name": "Globe Telecom",
        "keywords": ["Globe Telecom"],
        "sector": "Services",
    },
    "CNVRG": {
        "name": "Converge ICT Solutions",
        "keywords": ["Converge ICT", "Converge"],
        "sector": "Services",
    },
    "PGOLD": {
        "name": "Puregold Price Club",
        "keywords": ["Puregold"],
        "sector": "Services",
    },
    "WLCON": {
        "name": "Wilcon Depot",
        "keywords": ["Wilcon"],
        "sector": "Services",
    },
    "BLOOM": {
        "name": "Bloomberry Resorts Corporation",
        "keywords": ["Bloomberry", "Solaire"],
        "sector": "Services",
    },
    "RWM": {
        "name": "Travellers International Hotel Group",
        "keywords": ["Resorts World Manila", "Travellers International"],
        "sector": "Services",
    },
    "CEB": {
        "name": "Cebu Air",
        "keywords": ["Cebu Pacific", "Cebu Air"],
        "sector": "Services",
    },
    # --- Industrial ---
    "ICT": {
        "name": "International Container Terminal Services",
        "keywords": ["ICTSI", "International Container Terminal"],
        "sector": "Industrial",
    },
    "AP": {
        "name": "Aboitiz Power Corporation",
        "keywords": ["Aboitiz Power"],
        "sector": "Industrial",
    },
    "EMP": {
        "name": "Emperador Inc.",
        "keywords": ["Emperador", "Tanduay"],
        "sector": "Industrial",
    },
    "URC": {
        "name": "Universal Robina Corporation",
        "keywords": ["Universal Robina", "Jack n Jill"],
        "sector": "Industrial",
    },
    "MONDE": {
        "name": "Monde Nissin Corporation",
        "keywords": ["Monde Nissin", "Lucky Me"],
        "sector": "Industrial",
    },
    # --- Mining & Oil ---
    "SCC": {
        "name": "Semirara Mining and Power Corporation",
        "keywords": ["Semirara"],
        "sector": "Mining & Oil",
    },
    "NIKL": {
        "name": "Nickel Asia Corporation",
        "keywords": ["Nickel Asia"],
        "sector": "Mining & Oil",
    },
    "PX": {
        "name": "Philex Mining Corporation",
        "keywords": ["Philex Mining"],
        "sector": "Mining & Oil",
    },
}

# ---------------------------------------------------------------------------
# Sector-level macro triggers — news about these topics may affect every
# company in the corresponding sector.
# ---------------------------------------------------------------------------

SECTOR_TRIGGERS: dict[str, list[str]] = {
    "Financials": [
        "interest rate",
        "BSP",
        "Bangko Sentral",
        "central bank",
        "banking sector",
        "inflation",
        "monetary policy",
        "reserve requirement",
        "credit rating",
        "peso depreciation",
        "foreign exchange",
        "FX rate",
        "bank regulation",
        "RRR cut",
        "RRR hike",
        "policy rate",
    ],
    "Property": [
        "real estate",
        "property sector",
        "housing market",
        "condominium",
        "REIT",
        "Real Estate Investment Trust",
        "land prices",
        "housing loans",
        "mortgage rate",
    ],
    "Services": [
        "consumer spending",
        "retail sales",
        "tourism",
        "OFW remittances",
        "minimum wage",
        "labor market",
    ],
    "Industrial": [
        "manufacturing output",
        "supply chain",
        "import tariff",
        "energy cost",
        "oil price",
        "power rate",
        "industrial output",
    ],
    "Mining & Oil": [
        "mining",
        "nickel price",
        "coal price",
        "gold price",
        "mineral exports",
        "DENR",
        "mining moratorium",
    ],
}

# Maximum characters of text to scan (title + summary + content prefix)
_MAX_SCAN_CHARS = 2000


def find_affected_stocks(text: str) -> list[dict]:
    """
    Scan *text* for mentions of PSE-listed companies or sector-level macro
    triggers and return a list of potentially affected stocks.

    Each entry in the returned list is a dict with keys:

    - ``ticker``          — PSE ticker symbol (e.g. ``"BDO"``)
    - ``name``            — Full company name
    - ``sector``          — PSE sector classification
    - ``match_type``      — ``"direct"`` (company/ticker name found) or
                            ``"sector"`` (macro trigger for that sector found)
    - ``matched_keyword`` — The specific keyword or phrase that triggered the match

    Args:
        text: Combined news text (title + summary + article body).

    Returns:
        List of affected stock dicts, deduplicated by ticker.
    """
    if not text:
        return []

    scan_text = text[:_MAX_SCAN_CHARS]
    results: list[dict] = []
    seen_tickers: set[str] = set()

    # 1. Direct company / ticker mentions
    for ticker, info in PSE_COMPANIES.items():
        if ticker in seen_tickers:
            continue

        # Whole-word ticker match (case-sensitive — tickers are all-caps)
        if re.search(r"\b" + re.escape(ticker) + r"\b", scan_text):
            results.append(
                {
                    "ticker": ticker,
                    "name": info["name"],
                    "sector": info["sector"],
                    "match_type": "direct",
                    "matched_keyword": ticker,
                }
            )
            seen_tickers.add(ticker)
            continue

        # Keyword / alternate name match (case-insensitive)
        scan_lower = scan_text.lower()
        for kw in info["keywords"]:
            if kw.lower() in scan_lower:
                results.append(
                    {
                        "ticker": ticker,
                        "name": info["name"],
                        "sector": info["sector"],
                        "match_type": "direct",
                        "matched_keyword": kw,
                    }
                )
                seen_tickers.add(ticker)
                break

    # 2. Sector-level macro triggers
    sector_hits: set[str] = set()
    scan_lower = scan_text.lower()
    for sector, triggers in SECTOR_TRIGGERS.items():
        for trigger in triggers:
            if trigger.lower() in scan_lower:
                sector_hits.add(sector)
                break

    for ticker, info in PSE_COMPANIES.items():
        if ticker in seen_tickers:
            continue
        if info["sector"] in sector_hits:
            results.append(
                {
                    "ticker": ticker,
                    "name": info["name"],
                    "sector": info["sector"],
                    "match_type": "sector",
                    "matched_keyword": info["sector"],
                }
            )
            seen_tickers.add(ticker)

    return results
