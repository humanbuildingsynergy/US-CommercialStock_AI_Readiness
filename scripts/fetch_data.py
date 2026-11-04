"""Download the third-party inputs from their official sources and check each file's SHA-256.

Nothing here is redistributed by this repository: each file is fetched from its publisher into data/ (gitignored) and
must match the hash recorded below, the hash of the file the results were computed from. A mismatch stops the script;
the publisher has then changed the file, and the results may not reproduce.

    python scripts/fetch_data.py            # download what is missing, verify everything
    python scripts/fetch_data.py --verify   # verify only
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
UA = {"User-Agent": "Mozilla/5.0 (research data download; US-CommercialStock_AI_Readiness)"}

# (local path under data/, URL, sha256, what it is)
FILES = [
    ("cbecs/cbecs2018_final_public.csv",
     "https://www.eia.gov/consumption/commercial/data/2018/xls/cbecs2018_final_public.csv",
     "3e37c81aab2938e63fae47bec837270ba696699aceef904d1326dd25970e7247",
     "EIA, 2018 Commercial Buildings Energy Consumption Survey, public-use microdata"),
    ("cbecs/2018microdata_codebook.xlsx",
     "https://www.eia.gov/consumption/commercial/data/2018/xls/2018microdata_codebook.xlsx",
     "5ee2ccb1ae8a2c4196cdb5e3ffd2be11e877d13cae3474f550306f93e92bd26e",
     "EIA, CBECS 2018 microdata codebook"),
    ("sources/eia861/f8612018.zip",
     "https://www.eia.gov/electricity/data/eia861/archive/zip/f8612018.zip",
     "90f1d9e34665d91347d476ade7ff88261896a158eda640b7ec7002b8027eb8f8",
     "EIA, Form EIA-861 annual electric power industry report, 2018"),
    ("sources/eia861/f8612025.zip",
     "https://www.eia.gov/electricity/data/eia861/zip/f8612025.zip",
     "f59f6c732bf685fe0a1591b32227f5ce572dc0dfd26bb5caaeed128ac4e3a396",
     "EIA, Form EIA-861, 2025 final data"),
    ("sources/seds/use_all_btu.csv",
     "https://www.eia.gov/state/seds/sep_use/total/csv/use_all_btu.csv",
     "cbf0e26022e90303d355856ce750eaeedcb1cc6642c01c8b21f46d0284a9cedc",
     "EIA, State Energy Data System, consumption by state, sector and year (MSN TECCB used)"),
    ("sources/seds/Codes_and_Descriptions.xlsx",
     "https://www.eia.gov/state/seds/CDF/Codes_and_Descriptions.xlsx",
     "c34324b0ae9c793b45261a1377a2d4fb1ad002d27b54195329bed6514897169f",
     "EIA, SEDS series codes (TECCB = total energy consumption in the commercial sector)"),
    ("sources/census/cb_2018_us_state_20m.zip",
     "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip",
     "95902e4cda23f5f403e576d1bcd0e54a0f453d9b81893b080f4c15029302e348",
     "U.S. Census Bureau, 2018 cartographic boundary file, states, 1:20,000,000 (map only)"),
    ("sources/readiness/census_v2.json",
     "https://raw.githubusercontent.com/humanbuildingsynergy/building-ai-readiness/v1.0.0/results/census/census_v2.json",
     "33efc815578c0f04be7fd44c6d138bb15c8dd1957d363ae8a995738554bc867f",
     "the readiness framework's census of 45 building knowledge graphs (release v1.0.0; CC BY-NC 4.0)"),
    ("sources/readiness/canon.py",
     "https://raw.githubusercontent.com/humanbuildingsynergy/building-ai-readiness/v1.0.0/bara/canon.py",
     "c90e731acee84686d2ba7f68204d3100c521a7df5e2fcd33bdf0be61adcce3ff",
     "the readiness framework's canon module (release v1.0.0; Apache-2.0), to check the copy in atlas/canon.py"),
]
# members extracted from the EIA-861 archives: (archive, member, sha256)
MEMBERS = [
    ("sources/eia861/f8612018.zip", "Advanced_Meters_2018.xlsx",
     "1b1755281329cbd4fedef68481d79de17d647408f8491a84fe1f5f8aaaeaf334"),
    ("sources/eia861/f8612018.zip", "Sales_Ult_Cust_2018.xlsx",
     "bf049721bdaa78eaff7612093c74522450de7e59fa2c95c86991a0c956ec6b7c"),
    ("sources/eia861/f8612025.zip", "Advanced_Meters_2025.xlsx",
     "fcf86a2054a43e4bf34c0646f47dfd76ec0202d4a8753d3b41f75c6cb37e1e8f"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def main(verify_only: bool = False) -> int:
    bad = 0
    for rel, url, digest, what in FILES:
        p = DATA / rel
        if not p.exists() and not verify_only:
            print(f"downloading {rel} ({what})", flush=True)
            download(url, p)
        ok = p.exists() and sha256(p) == digest
        bad += not ok
        print(f"{'ok      ' if ok else 'MISMATCH' if p.exists() else 'MISSING '} {rel}")
    for archive, member, digest in MEMBERS:
        p = DATA / Path(archive).parent / member
        if not p.exists() and (DATA / archive).exists() and not verify_only:
            with zipfile.ZipFile(DATA / archive) as z:
                p.write_bytes(z.read(member))
        ok = p.exists() and sha256(p) == digest
        bad += not ok
        print(f"{'ok      ' if ok else 'MISMATCH' if p.exists() else 'MISSING '} {p.relative_to(DATA).as_posix()} (from {Path(archive).name})")
    if bad:
        print(f"{bad} file(s) missing or not matching their recorded SHA-256", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(verify_only="--verify" in sys.argv))
