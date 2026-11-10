"""
Telepítés:  pip install requests beautifulsoup4
Futtatás:   python moravoseed_webscraper.py
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://moravoseed.cz/"
CATEGORIES = {
    "Zeleniny": "https://moravoseed.cz/index.php?stranka=eshop&akce=kategorie&kategorie=4&subkat=1",
    "Květiny": "https://moravoseed.cz/index.php?stranka=eshop&akce=kategorie&kategorie=4&subkat=7",
    "Léčivé a aromatické rostliny": "https://moravoseed.cz/index.php?stranka=eshop&akce=kategorie&kategorie=4&subkat=5",
}

OUT_JSON = Path("moravoseed.json")
JSON_DIR = Path("json")
IMG_DIR = Path("images")

WORKERS = 4    # egyszerre hány szál tölt le (ha a szerver gyakran szakít, vedd lejjebb 2-3-ra)
DELAY = 0.2    # szálanként ennyi szünet két kérés között (másodperc)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; personal-scraper)"}
IMG_RE = re.compile(r"""[^\s'"()<>=,]+\.(?:jpe?g|png|gif|webp)""", re.I)
IGNORE = ("info.png", "logo", "kosik", "uk.gif", "eu-info")  # az oldal saját ikonjai

_local = threading.local()
_print_lock = threading.Lock()
_debug_lock = threading.Lock()


def get_session():
    """Szálanként külön session, automatikus újrapróbálkozással."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        retry = Retry(
            total=5, connect=5, read=5, backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=WORKERS))
        s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=WORKERS))
        _local.session = s
    return s


def to_number(text):
    """'21,28' -> 21.28 ; '100' -> 100 ; hibás érték -> None"""
    try:
        n = float(text.replace(",", ".").replace(" ", ""))
    except (ValueError, AttributeError):
        return None
    return int(n) if n.is_integer() else n


def to_record(row):
    """A táblázatsorból a példafájlhoz hasonló szerkezetű JSON rekord."""
    amount_raw = row["Gramáž"]
    m = re.fullmatch(r"([\d.,]+)\s*([A-Za-zµ]*)", amount_raw.strip())
    value = to_number(m.group(1)) if m else None
    unit = m.group(2).lower() if m else ""

    return {
        "id": row["Obj. číslo"],
        "category": row["Kategória"],
        "name": row["Druh česky"],
        "latin_name": row["Druh latinsky"],
        "variety": row["Odrůda"],
        "amount_raw": amount_raw,
        "seed_count": value if unit == "s" else None,   # "s" = počet semen
        "weight_g": value if unit == "g" else None,     # "g" = gramm
        "price_category": row["Cenová skupina"],
        "price_without_vat": to_number(row["Cena (Kč/Ks) bez DPH"]),
        "price_with_vat": to_number(row["Cena (Kč/Ks) s DPH"]),
        "currency": "CZK",
        "image_file": row["Kép fájl"] or None,
        "image_url": row["Kép URL"] or None,
        "detail_url": row["_detail"] or None,
    }


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def fetch(url):
    r = get_session().get(url, timeout=30)
    r.raise_for_status()
    return r


def parse_page(category, url):
    r = fetch(url)
    r.encoding = r.apparent_encoding or "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 8:
            continue
        texts = [c.get_text(" ", strip=True) for c in cells]
        if not re.fullmatch(r"\d{4,6}", texts[0]):
            continue

        # a részletek (és kép) oldal: onclick="window.open('eshop/detail.php?id=7&kat=4', ...)"
        detail_url = ""
        for tag in tr.find_all(True):
            m = re.search(r"window\.open\('([^']*detail[^']*)'", tag.get("onclick", ""))
            if m:
                detail_url = urljoin(BASE, m.group(1).replace("&amp;", "&"))
                break

        rows.append({
            "Kategória": category,
            "Obj. číslo": texts[0],
            "Druh česky": texts[1],
            "Druh latinsky": texts[2],
            "Odrůda": texts[3],
            "Gramáž": texts[4],
            "Cenová skupina": texts[5],
            "Cena (Kč/Ks) bez DPH": texts[6],
            "Cena (Kč/Ks) s DPH": texts[7],
            "Kép URL": "",
            "Kép fájl": "",
            "_detail": detail_url,
        })
    return rows


def find_image_url(detail_url):
    r = fetch(detail_url)
    r.encoding = r.apparent_encoding or "utf-8"

    with _debug_lock:  # az első részletek oldalt elmentjük hibakereséshez
        dbg = Path("detail_debug.html")
        if not dbg.exists():
            dbg.write_text(r.text, encoding="utf-8")

    sub = BeautifulSoup(r.text, "html.parser")
    found = []
    for img in sub.find_all("img"):
        src = img.get("src", "")
        if src and not any(x in src.lower() for x in IGNORE):
            found.append(urljoin(detail_url, src))
    for a in sub.find_all("a", href=True):
        if IMG_RE.fullmatch(a["href"]) and not any(x in a["href"].lower() for x in IGNORE):
            found.append(urljoin(detail_url, a["href"]))
    return found[0] if found else None


def process(row):
    """Egy termék képének letöltése. Visszatér: (row, állapot)."""
    obj_id = row["Obj. číslo"]

    # már letöltött kép -> nem kérünk le semmit
    existing = next(IMG_DIR.glob(f"{obj_id}.*"), None)
    if existing:
        row["Kép fájl"] = existing.name
        return row, "kihagyva"

    if not row["_detail"]:
        return row, "nincs részletek link"

    try:
        img_url = find_image_url(row["_detail"])
        time.sleep(DELAY)
        if not img_url:
            return row, "nincs kép"
        ext = Path(img_url.split("?")[0]).suffix.lower() or ".jpg"
        target = IMG_DIR / f"{obj_id}{ext}"
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(fetch(img_url).content)
        tmp.rename(target)  # csak teljes letöltés után kerül a végleges névre
        row["Kép URL"] = img_url
        row["Kép fájl"] = target.name
        time.sleep(DELAY)
        return row, "letöltve"
    except requests.RequestException as e:
        return row, f"HIBA: {e}"


def main():
    IMG_DIR.mkdir(exist_ok=True)

    all_rows = []
    for category, url in CATEGORIES.items():
        rows = parse_page(category, url)
        log(f"{category}: {len(rows)} termék")
        all_rows.extend(rows)

    total = len(all_rows)
    stats = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process, row) for row in all_rows]
        for fut in as_completed(futures):
            row, status = fut.result()
            done += 1
            key = status.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            if status not in ("letöltve", "kihagyva"):
                log(f"  [{row['Obj. číslo']}] {status}")
            if done % 25 == 0 or done == total:
                log(f"  {done}/{total} kész")

    records = [to_record(r) for r in all_rows]

    # 1) egyetlen összesített fájl
    OUT_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) termékenként külön fájl, a kép nevével egyezően (pl. json/60303.json)
    JSON_DIR.mkdir(exist_ok=True)
    for rec in records:
        (JSON_DIR / f"{rec['id']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    log(f"\nKész: {total} termék -> {OUT_JSON} és {JSON_DIR}/")
    log("Összesítés: " + ", ".join(f"{k}: {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()