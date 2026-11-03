"""The EIA CBECS 2018 codebook, read with the standard library (an .xlsx is zipped XML).

``variables()`` returns name -> Variable(order, type, label, codes, question). ``codes`` parses the
"Values/Format codes" column into {code: label}; "Missing" is kept as the key "Missing". The
codebook has a few rows with two codes on one line ("29=High school 30=Preschool/daycare"), so the
split is on every ``<code>=`` token, not on line breaks.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

from atlas import config

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_CODE = re.compile(r"(?:^|(?<=\s)|(?<=;))(\d+|Missing)=")


def read_xlsx(path: Path) -> list[dict[str, str]]:
    """Rows of the first worksheet as {column letter: text}."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{_NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{_NS}t")))
        sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        rows = []
        for r in ET.fromstring(z.read(sheet)).iter(f"{_NS}row"):
            row = {}
            for c in r.findall(f"{_NS}c"):
                col = re.match(r"[A-Z]+", c.get("r")).group(0)
                v = c.find(f"{_NS}v")
                if c.get("t") == "s" and v is not None:
                    row[col] = shared[int(v.text)]
                elif c.get("t") == "inlineStr":
                    row[col] = "".join(t.text or "" for t in c.iter(f"{_NS}t"))
                else:
                    row[col] = v.text if v is not None else ""
            rows.append(row)
    return rows


def parse_codes(text: str) -> dict[str, str]:
    """'1=Yes\\n2=No\\nMissing=Not applicable' -> {'1': 'Yes', '2': 'No', 'Missing': ...}."""
    marks = list(_CODE.finditer(text or ""))
    out = {}
    for m, nxt in zip(marks, marks[1:] + [None]):
        label = text[m.end(): nxt.start() if nxt else len(text)]
        out[m.group(1)] = " ".join(label.replace(";", " ").split())
    return out


@dataclass(frozen=True)
class Variable:
    order: int
    name: str
    vtype: str
    label: str
    codes: dict[str, str]
    raw_codes: str
    question: str


@lru_cache(maxsize=1)
def variables(path: Path = config.CODEBOOK_XLSX) -> dict[str, Variable]:
    out = {}
    for r in read_xlsx(path):
        order, name = r.get("A", ""), r.get("B", "")
        if not order.isdigit() or not name:
            continue
        raw = r.get("E", "")
        out[name] = Variable(int(order), name, r.get("C", ""), " ".join(r.get("D", "").split()),
                             parse_codes(raw), raw, " ".join(r.get("F", "").split()))
    return out
