from dataclasses import dataclass
from typing import List, Optional
import os
import re
import requests
from bs4 import BeautifulSoup


@dataclass
class Listing:
    source: str
    title: str
    price: Optional[float]
    url: Optional[str]


def _parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    # Extract number-like strings: 1,23,456 or 123,456 or 123456
    m = re.search(r"(\d[\d,]{4,})", text.replace("\xa0", " "))
    if not m:
        return None
    digits = m.group(1).replace(",", "")
    try:
        return float(digits)
    except ValueError:
        return None


def _heuristic_price(make: str, model: str, year: int, mileage: int) -> float:
    # Simple fallback: base by year and mileage depreciation
    years_old = max(0, 2025 - int(year))
    base = 1500000.0  # ~ INR for typical sedan baseline; adjust down by age
    depreciation_year = 0.12
    price = base * ((1.0 - depreciation_year) ** years_old)
    mileage_factor = max(0.5, 1.0 - (mileage / 150000.0))
    return max(50000.0, price * mileage_factor)


def _fetch_cars24(make: str, model: str, year: int, city: Optional[str]) -> List[Listing]:
    # Basic HTML fetch; endpoints can change. We parse conservatively.
    # If blocked, return empty list and caller will fallback.
    try:
        q_city = (city or "").lower().strip()
        url = f"https://www.cars24.com/buy-used-{make}-{model}/"
        headers = {"User-Agent": os.getenv("SCRAPER_USER_AGENT", "Mozilla/5.0")}
        resp = requests.get(url, headers=headers, timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12")))
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        cards = soup.select("a[href*='/used-car']")
        listings: List[Listing] = []
        for a in cards[:12]:
            title = a.get_text(strip=True) or f"{make} {model}"
            price_el = a.find_next(string=re.compile(r"\₹|\d"))
            price = _parse_price(price_el) if price_el else None
            href = a.get("href")
            if href and not href.startswith("http"):
                href = "https://www.cars24.com" + href
            listings.append(Listing(source="Cars24", title=title, price=price, url=href))
        return listings
    except Exception:
        return []


def _fetch_cardekho(make: str, model: str, year: int, city: Optional[str]) -> List[Listing]:
    try:
        url = f"https://www.cardekho.com/used-{make}-{model}+cars+{year}"  # heuristic path
        headers = {"User-Agent": os.getenv("SCRAPER_USER_AGENT", "Mozilla/5.0")}
        resp = requests.get(url, headers=headers, timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "12")))
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        items = soup.select("a[title]")
        listings: List[Listing] = []
        for a in items[:12]:
            title = a.get("title") or a.get_text(strip=True)
            if not title:
                continue
            price_el = a.find_next(string=re.compile(r"\₹|\d"))
            price = _parse_price(price_el) if price_el else None
            href = a.get("href")
            if href and not href.startswith("http"):
                href = "https://www.cardekho.com" + href
            listings.append(Listing(source="CarDekho", title=title, price=price, url=href))
        return listings
    except Exception:
        return []


def fetch_market_prices(make: str, model: str, year: int, mileage: int, city: Optional[str]) -> List[Listing]:
    results: List[Listing] = []
    # Try a couple sources; tolerate failures discreetly
    results.extend(_fetch_cars24(make, model, year, city))
    results.extend(_fetch_cardekho(make, model, year, city))

    # If scraping failed, provide heuristic baseline and synthetic comps
    if not any(l.price for l in results):
        baseline = _heuristic_price(make, model, year, mileage)
        results = [
            Listing(source="Heuristic", title=f"{make} {model} {year} - est A", price=baseline * 0.95, url=None),
            Listing(source="Heuristic", title=f"{make} {model} {year} - est B", price=baseline * 1.00, url=None),
            Listing(source="Heuristic", title=f"{make} {model} {year} - est C", price=baseline * 1.05, url=None),
        ]
    else:
        # Deduplicate and keep top 6 entries with valid prices
        seen = set()
        filtered: List[Listing] = []
        for l in results:
            key = (l.source, l.url or l.title)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(l)
        results = [l for l in filtered if l.price is not None][:6]
        if not results:
            baseline = _heuristic_price(make, model, year, mileage)
            results = [Listing(source="Heuristic", title=f"{make} {model} {year}", price=baseline, url=None)]

    return results


