"""Building-level heterogeneity. The edition and linkage (variant B) drawn per building are the primary model;
the fixed-value results are the comparison.

The fixed-value map gives every building the same premise values. Here, each building draws its
own, from the committed tables of ``atlas.linkage``:

  linkage            A: each building's r_i is a census graph's C_linked / C_declared, the graph drawn
                        with probability proportional to C_declared (which makes E[r_i] = R1, the
                        ratio of sums, exactly; a uniform draw is a labelled sensitivity).
                     B: each building draws a linkage type (all / none / partial, census shares); a
                        partial building links each declared class with its census linked-given-
                        declared probability. The floor stays undiscounted.
                     contrast: independent per-class linkage, no building type.
  edition            a state within the census division (prob. ∝ 2018 commercial energy use) and a
                     year within the construction bin; the edition in force from tables/V_states.csv;
                     renovated systems re-dated the same way over the renovation window.
  compliance         each 90.1-2013-onward clause row installs its streams with probability c.
  presence           each reported-control row produces its streams with its census presence share.

Two kinds of uncertainty are kept apart. Heterogeneity is the spread over draws (5th–95th
percentiles of each stock share). Sampling uncertainty is the CBECS jackknife applied to the weighted
sum of each building's expectation over draws; the jackknife never runs inside a draw.

C(I) and tier coverage are looked up per stream-class set (bitmask), so draws are cheap. One master
seed; each run's generator is seeded with (MASTER_SEED, crc32(run name)).

    python -m atlas.draws      -> the per-building draw runs of an earlier design (not part of the release results)
"""
from __future__ import annotations

import zlib
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from atlas import canon, cbecs, config, declared, floors, linkage, tiers, vintage
from atlas.replicates import REPLICATES

MASTER_SEED = 20261017
N_DRAWS = 1000
Z = 1.96
TH = tiers.THRESHOLD - 1e-12
F_MIN = 20 / 76                          # the minimal code profile T+Sp+Whole

CLS = tuple(canon.STREAMS)
BIT = {c: 1 << i for i, c in enumerate(CLS)}
MEASURES = ("C", *tiers.TIERS)           # C(I), then coverage of status / summary / diagnosis
READINGS = {                             # reading -> (Table E reading, drop rare classes)
    "primary rows": ("primary", False),  # formerly "primary rows": the frozen Table E rows
    "enriched": ("enriched", False),
    "census-supported": ("census_primary", False),
    "strict": ("primary", True),
    "S4 controls=wo": ("wo", False),
    "S5 controls=primary_e02": ("primary_e02", False),
}
FOUR = ("strict", "census-supported", "primary rows", "enriched")


def to_mask(s) -> int:
    m = 0
    for c in s:
        m |= BIT[c]
    return m


def from_mask(m: int) -> frozenset:
    return frozenset(c for c, b in BIT.items() if m & b)


def _fs(s: str) -> frozenset:
    return frozenset() if s == "-" else frozenset(s.split("+"))


@lru_cache(maxsize=None)
def _score_one(m: int, scoring: str = "primary") -> tuple[float, ...]:
    s = from_mask(m)
    return (canon.readiness(s, scoring), *(tiers.coverage(s, t, scoring) for t in tiers.TIERS))


def score(masks: np.ndarray, scoring: str = "primary") -> np.ndarray:
    """(N,) int masks -> (N, 4): C(I) and the three tier coverages (scoring "hvac": the HVAC-credit scoring)."""
    u, inv = np.unique(masks, return_inverse=True)
    tab = np.array([_score_one(int(m), scoring) for m in u])
    return tab[inv]


CELLS = tuple(canon.CANONS["v2"].rungs)        # the 31 domain x intent cells


@lru_cache(maxsize=None)
def _cells_one(m: int, scoring: str = "primary") -> tuple[int, ...]:
    s = set(from_mask(m))
    return tuple(canon.level(d, i, s, scoring) for d, i in CELLS)


def cell_levels(masks: np.ndarray, scoring: str = "primary") -> np.ndarray:
    """(N,) masks -> (N, 31): achieved level of every domain x intent cell."""
    u, inv = np.unique(masks, return_inverse=True)
    tab = np.array([_cells_one(int(m), scoring) for m in u])
    return tab[inv]


@lru_cache(maxsize=1)
def rare_mask() -> int:
    from atlas import bounds
    return to_mask(bounds.rare_classes())


def bits_matrix(masks: np.ndarray) -> np.ndarray:
    """(N,) masks -> (N, K) bool."""
    return ((masks[:, None] >> np.arange(len(CLS))) & 1).astype(bool)


def pack(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.int64) << np.arange(len(CLS))).sum(1)


# ── the stock and the deterministic inputs ───────────────────────────────────────────────────────
@dataclass
class Stock:
    df: pd.DataFrame
    W: np.ndarray            # (N, 152): FINALWT and the 151 replicates
    sq: np.ndarray
    size: np.ndarray
    rungs: dict[str, np.ndarray] = field(default_factory=dict)   # deterministic central rungs

    @property
    def a(self) -> np.ndarray:
        return self.W[:, 0] * self.sq


@lru_cache(maxsize=1)
def stock() -> Stock:
    df = cbecs.load(replicates=True)
    rg = vintage.assign(df)
    rungs = {k: rg[f"{k}_central"].to_numpy() for k in ("new", *vintage.SYSTEMS)}
    return Stock(df, df[["FINALWT", *REPLICATES]].to_numpy(dtype=float), df.SQFT.to_numpy(dtype=float),
                 floors.size_class(df.SQFT).to_numpy(), rungs)


@lru_cache(maxsize=None)
def base(reading: str, floor_reading: str = "main") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(floor mask, Table E mask, declared mask) at the deterministic central level."""
    st = stock()
    controls, strict = READINGS[reading]
    R = declared.records(st.df, floor_reading=floor_reading, controls=controls)
    F = np.array([to_mask(_fs(s)) for s in R.floor_streams_central], dtype=np.int64)
    D = np.array([to_mask(_fs(s)) for s in R.declared_streams_central], dtype=np.int64)
    E = np.array([to_mask(s) for s in declared.declared_streams(st.df, controls)], dtype=np.int64)
    if strict:
        keep = ~rare_mask()
        F, D, E = F & keep, D & keep, E & keep
    return F, E, D


# ── the floor as a vectorised function of edition rungs and compliance ───────────────────────────
@lru_cache(maxsize=None)
def floor_rows(floor_reading: str = "main", held: bool = False) -> tuple[dict, ...]:
    """The clause rows of a reading; with *held*, also the rows floors.HELD keeps in that reading's imputation."""
    st = stock()
    L = vintage.ladder()
    need13 = L.index(linkage.COMPLIANCE_FROM)
    tags = floors.READINGS[floor_reading] | (floors.HELD.get(floor_reading, frozenset()) if held else frozenset())
    rows = []
    for r in floors.table_c():
        if r["reading"] not in tags:
            continue
        rows.append({"id": r["id"], "bits": to_mask(r["streams"].split("+")), "path": r["path"],
                     "need": L.index(r["rung_from"]), "system": r["system"],
                     "cond": floors._condition(st.df, r["condition"]).to_numpy(),
                     "renov": floors._renovated(st.df, r["system"]).to_numpy() if r["path"] == "alteration" else None,
                     "comply": L.index(r["rung_from"]) >= need13})
    return tuple(rows)


def floor_masks(rungs: dict[str, np.ndarray], comply: dict[str, np.ndarray] | None = None,
                floor_reading: str = "main", detail: dict | None = None, held: bool = False) -> np.ndarray:
    """Code-required (floor) masks; with *detail* a dict, also records each Table C row's applies-mask."""
    n = len(stock().sq)
    out = np.zeros(n, dtype=np.int64)
    for r in floor_rows(floor_reading, held):
        if r["path"] == "new":
            ok = rungs["new"] >= r["need"]
        elif r["path"] == "new+alteration":
            ok = rungs[r["system"]] >= r["need"]
        else:
            ok = r["renov"] & (rungs[r["system"]] >= r["need"])
        ok = ok & r["cond"]
        if comply is not None and r["comply"]:
            ok = ok & comply[r["id"]]
        if detail is not None:
            detail[r["id"]] = (ok, r["bits"])
        out |= np.where(ok, r["bits"], 0)
    return out


# ── edition: per-building state and year ─────────────────────────────────────────────────────────
YEARS = np.arange(1900, vintage.SURVEY_YEAR + 1)


@lru_cache(maxsize=1)
def state_rung_table() -> tuple[list[str], np.ndarray]:
    L = vintage.ladder()
    states = sorted(vintage.state_rows())
    tab = np.array([[L.index(vintage.state_edition_at(s, int(y))) for y in YEARS] for s in states])
    return states, tab


@lru_cache(maxsize=None)
def division_draw(weighted: bool) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    states, _ = state_rung_table()
    w = vintage.state_weights()
    out = {}
    for d in vintage.divisions():
        ss = vintage.division_states(d)
        p = np.array([w[s] if weighted else 1.0 for s in ss], dtype=float)
        out[d] = (np.array([states.index(s) for s in ss]), p / p.sum())
    return out


# Timing sensitivities: an adoption lag (the edition in force `lag` years before the drawn construction and renovation
# years), and whether the 90.1-2013 monitoring clause is credited only where the code in force was 90.1 itself.
# The primary is lag 0 and no family rule. LAST keeps the last draw's years and states for the family rule.
TIMING = {"lag": 0, "monitoring_901_only": False}
LAST: dict = {}


@lru_cache(maxsize=1)
def family_901_table() -> np.ndarray:
    """(states, years) bool: the code in force in that state and year is ASHRAE 90.1 itself (tables/V_states.csv
    code_family of the latest row in force)."""
    states, _ = state_rung_table()
    V = pd.read_csv(config.TABLES / "V_states.csv", dtype={"edition": str})
    out = np.zeros((len(states), len(YEARS)), bool)
    for i, s in enumerate(states):
        r = V[(V.state == s) & (V.kind != "implicit")].sort_values("year_in_force")
        for y_i, y in enumerate(YEARS):
            live = r[r.year_in_force <= y]
            out[i, y_i] = bool(len(live)) and live.iloc[-1].code_family == "90.1"
    return out


def draw_rungs(rng: np.random.Generator, weighted: bool = True) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One draw of every building's edition rung per system; also returns the drawn state index."""
    st = stock()
    df = st.df
    n = len(df)
    _, tab = state_rung_table()
    cd = df.CENDIV.to_numpy().astype(int)
    sidx = np.empty(n, dtype=int)
    for d, (idx, p) in division_draw(weighted).items():
        m = cd == d
        sidx[m] = rng.choice(idx, size=int(m.sum()), p=p)
    yc = df.YRCONC.to_numpy().astype(int)
    lo = np.array([vintage.YRCONC_YEARS[y][0] for y in yc])
    hi = np.array([vintage.YRCONC_YEARS[y][1] for y in yc])
    year = rng.integers(lo, hi + 1)
    lag = int(TIMING["lag"])
    year_e = np.maximum(year - lag, YEARS[0])              # the edition in force `lag` years earlier (0 = primary)
    base_r = tab[sidx, year_e - YEARS[0]]
    rungs = {"new": base_r.copy(), **{s: base_r.copy() for s in vintage.SYSTEMS}}
    renov = (df.RENOV == 1).to_numpy()
    rlo = np.maximum(vintage.RENOV_FROM, lo)
    ryear = rng.integers(rlo, vintage.SURVEY_YEAR + 1)
    ren_r = tab[sidx, np.maximum(ryear - lag, YEARS[0]) - YEARS[0]]
    LAST.update(year=year_e, sidx=sidx)
    for var, systems in vintage.renovation_rules().items():
        hit = renov & (df[var] == 1).to_numpy()
        for s in systems:
            rungs[s] = np.where(hit, np.maximum(rungs[s], ren_r), rungs[s])
    return rungs, sidx


# ── samplers: each returns one draw of every building's central (N, 4) measures ─────────────────
def _beta_or_fixed(rng, v):
    return v if isinstance(v, float) else rng.beta(*v)


@dataclass
class Spec:
    name: str
    reading: str = "primary rows"
    linkage: str = "det"               # det | A | A_uniform | B | contrast
    prior: str = "beta11"
    population: str = "R1"             # class rates for partial buildings: R1 (37) | R3 (30) | partial (6)
    type_population: str = "R1"        # linkage-type shares: R1 | R3
    floor_reading: str = "main"        # Table C reading (S2 early_occ, S3 the readiness framework's mapping)
    th: float = tiers.THRESHOLD        # tier threshold (S7)
    discount_floor: bool = False       # S8: the floor's classes are linked like declared ones
    drop_implicit: bool = False        # S10: Table V state rows without the implicit-practice rows
    seed_key: str = ""                 # common random numbers: runs sharing a key share their draws
    track: bool = False                # keep the class-group decomposition and the (floor, declared) patterns
    # recording layer and description; with recording on and the three rules off it reduces to the model without it
    recording: bool = False
    rec_thermo: bool = True            # conventional thermostats record nothing
    rec_meter: bool = True             # only interval data counts for the whole-building meter
    rec_route: bool = True             # routes for every other class (occupancy/lighting, flagged rows)
    smart_credit: bool = True          # smart thermostats record zone temperature and setpoint
    flagged: str = "provisional"       # provisional | credited (rows whose route CBECS leaves unclear)
    d: float = 1.0                     # description probability per recorded class
    edition: str = "det"               # det | weighted | uniform
    c: float = 1.0                     # compliance of 2013-onward clause rows
    presence: str = "det"              # det | beta11 | beta0505
    add: frozenset = frozenset()       # classes added to floor and declared (H5 counterfactuals)
    add_scope: str = "all"             # all | 25k (buildings >= 25,000 ft2, where §8.4.3 applies)
    note: str = ""


def _graph_ratios(weighted: bool) -> tuple[np.ndarray, np.ndarray]:
    g = linkage.graphs()
    pos = g[g.in_R1]
    p = pos.c_ans_declared.to_numpy(dtype=float) if weighted else np.ones(len(pos))
    return pos.ratio.to_numpy(dtype=float), p / p.sum()


@lru_cache(maxsize=None)
def _e_rows(controls: str) -> list[tuple[str, int, np.ndarray]]:
    st = stock()
    tags = declared.E_READINGS[controls]
    out = []
    for row in declared.table_e():
        if row["reading"] in tags and row["streams"] != "-":
            out.append((row["id"], to_mask(row["streams"].split("+")),
                        st.df.eval(row["condition"]).fillna(False).astype(bool).to_numpy()))
    return out


def sample(spec: Spec, rng: np.random.Generator, rng2: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
    st = stock()
    n = len(st.sq)
    F0, E0, D0 = base(spec.reading, spec.floor_reading)
    controls, strict = READINGS[spec.reading]
    keep = ~rare_mask() if strict else -1
    keep_bas = keep                       # with the recording layer, the strict census test
    if spec.recording and strict:         # applies to BAS-route streams only; the meter stays via the AMI draw
        keep = -1
    info = {}
    # floor: edition draw and compliance
    detail = {} if spec.recording else None
    sidx = None
    if spec.edition == "det" and spec.c >= 1.0 and not spec.recording:
        F = F0
    else:
        rungs = st.rungs
        if spec.edition != "det":
            rungs, sidx = draw_rungs(rng, weighted=(spec.edition == "weighted"))
            info["new_rung"] = rungs["new"]
            info["max_rung"] = np.maximum.reduce([rungs[s] for s in ("new", *vintage.SYSTEMS)])
        comply = None
        if spec.c < 1.0:
            rows = [r for r in floor_rows(spec.floor_reading) if r["comply"]]
            comply = {r["id"]: rng.random(n) < spec.c for r in rows if r["id"] not in floors.OWN_STREAM}
            own = [r for r in rows if r["id"] in floors.OWN_STREAM]
            if own:                       # a child stream (spawning does not advance rng), so no other draw moves
                rng_own = rng.spawn(1)[0]
                comply |= {r["id"]: rng_own.random(n) < spec.c for r in own}
        F = floor_masks(rungs, comply, spec.floor_reading, detail) & keep
    # reported controls: presence
    if spec.presence == "det":
        E = E0
    else:
        pb = linkage.presence_beta(spec.presence)
        E = np.zeros(n, dtype=np.int64)
        for rid, bits, cond in _e_rows(controls):
            p = _beta_or_fixed(rng, pb[rid])
            E |= np.where(cond & (rng.random(n) < p), bits, 0)
        E = E & keep
    add = np.zeros(n, dtype=np.int64)
    if spec.add:
        add = np.where((st.sq >= 25000) if spec.add_scope == "25k" else np.ones(n, bool), to_mask(spec.add), 0)
        F, E = F | add, E | add
    D = F | E
    sF, sD = score(F), score(D)
    info.update(F=F, D=D, sF=sF, sD=sD)
    if spec.recording:
        RD, BASR = _recording_routes(spec, F, E, add, detail, sidx, controls, keep_bas, rng2)
        info.update(F_eff=RD, D_eff=RD | BASR)
    # linkage. With discount_floor (S8) the floor's classes are linked like the declared ones.
    lo_s = np.zeros_like(sF) if spec.discount_floor else sF
    if spec.linkage == "det":
        r = declared.discount()["R1"]
        info["r"] = np.full(n, r)
        return lo_s + (sD - lo_s) * r, info
    if spec.linkage in ("A", "A_uniform"):
        vals, p = _graph_ratios(weighted=(spec.linkage == "A"))
        ri = rng.choice(vals, size=n, p=p)
        info["r"] = ri
        return lo_s + (sD - lo_s) * ri[:, None], info
    keep_base = 0 if spec.discount_floor else F
    extra = bits_matrix(D & ~keep_base)
    if spec.recording:
        keep_base = RD
        extra = bits_matrix(BASR & ~RD)
    cb = linkage.class_beta(spec.prior, spec.population)
    pc = np.array([_beta_or_fixed(rng, cb[c]) for c in CLS])
    u = rng.random(extra.shape) < pc[None, :]
    if spec.linkage == "contrast":
        linked = extra & u
    else:
        tp = linkage.type_probs_pop(spec.type_population)
        ty = rng.choice(3, size=n, p=[tp["all"], tp["none"], tp["partial"]])
        linked = extra & ((ty == 0)[:, None] | ((ty == 2)[:, None] & u))
    cm = keep_base | pack(linked)
    if spec.recording and spec.d < 1.0:
        desc = rng2.random((n, len(CLS))) < spec.d
        cm = cm & pack(desc)
    info["Cmask"] = cm
    return score(cm), info


def _recording_routes(spec, F, E, add, detail, sidx, controls, keep, rng2):
    """(recorded without the BAS draw, BAS-routed installed classes) per building."""
    from atlas import recording as rec
    st = stock()
    n = len(st.sq)
    R = rec.routes(controls)
    tsp, whole, occ = to_mask(("T", "Sp")), to_mask(("Whole",)), to_mask(("Occ",))
    RD = np.zeros(n, dtype=np.int64)
    BASR = np.zeros(n, dtype=np.int64)
    code_log = np.zeros(n, dtype=bool)
    r_system = {r["id"]: r["system"] for r in floors.table_c()}
    for rid, (ok, bits) in detail.items():
        if bits == tsp:                                        # C01: the thermostat
            if spec.rec_thermo:
                BASR |= np.where(ok & R["bas_hvac"], tsp, 0)
                if spec.smart_credit:
                    RD |= np.where(ok & R["smart"], tsp, 0)
            else:
                RD |= np.where(ok, tsp, 0)
        elif bits == whole:                                    # C02: the utility meter (handled below)
            continue
        elif bits == occ:                                      # occupancy clauses
            if spec.rec_route:
                BASR |= np.where(ok & R["bas_light"], occ, 0)
            else:
                RD |= np.where(ok, occ, 0)
        else:                                                  # monitoring clauses (§8.4.3): code-required logging
            RD |= np.where(ok, bits, 0)
            if r_system[rid] == "electrical":                  # §10.4.5 (C15) records district energy, not electricity
                code_log |= ok
    c02 = next((ok for ok, bits in detail.values() if bits == whole), np.zeros(n, bool))
    if spec.rec_meter:
        if sidx is None:                                       # no edition draw: draw the state for the meter
            cd = st.df.CENDIV.to_numpy().astype(int)
            sidx = np.empty(n, dtype=int)
            for d_, (idx, p) in division_draw(True).items():
                m = cd == d_
                sidx[m] = rng2.choice(idx, size=int(m.sum()), p=p)
        ami = rng2.random(n) < rec.ami_probs(sidx)
        RD |= np.where(c02 & (ami | code_log), whole, 0)        # §8.4.3 logs total electricity; else interval meter
    else:
        RD |= np.where(c02, whole, 0)
    if spec.rec_route:
        BASR |= R["E_bas"] & E
        if spec.smart_credit:
            RD |= R["E_smart"] & E
        if spec.flagged == "credited":
            RD |= R["E_flag_all"] & E
        else:
            BASR |= R["E_flag_bas"] & E
    else:
        BASR |= E
        if not spec.smart_credit:
            BASR &= ~(R["E_smart"] & ~R["E_bas"])
    RD |= add                                                  # counterfactual additions count as recorded data
    return RD, BASR & keep


# ── class-group decomposition and vintage reconfiguration ───────────────────────────────────────
GROUP_NAMES = ('thermostat', 'metering', 'BAS HVAC points', 'occupancy, lighting, CO2')


def _groups():
    from atlas import bounds
    return bounds.GROUPS


@lru_cache(maxsize=None)
def _phi_one(m: int, scoring: str = "primary") -> tuple[float, ...]:
    from atlas import bounds
    return bounds._phi(from_mask(m), scoring)


def phi(masks: np.ndarray, scoring: str = "primary") -> np.ndarray:
    """(N,) masks -> (N, 4): exact Shapley share of C(I) per stream-class group (atlas.bounds), under a scoring."""
    u, inv = np.unique(masks, return_inverse=True)
    return np.array([_phi_one(int(m), scoring) for m in u])[inv]


def _configure_vintage(drop_implicit: bool = False, doe_dates: bool = False) -> None:
    vintage.configure(drop_implicit=drop_implicit, doe_dates=doe_dates)
    state_rung_table.cache_clear()
    division_draw.cache_clear()


# ── running a spec ───────────────────────────────────────────────────────────────────────────────
def seed_for(name: str) -> np.random.Generator:
    return np.random.default_rng([MASTER_SEED, zlib.crc32(name.encode())])


HIST_BINS = np.linspace(0, 1, 101)


def run(spec: Spec, n_draws: int = N_DRAWS) -> dict:
    st = stock()
    n = len(st.sq)
    rng = seed_for(spec.seed_key or spec.name)
    rng2 = seed_for((spec.seed_key or spec.name) + " | recording")
    TH = spec.th - 1e-12
    if spec.drop_implicit:
        _configure_vintage(drop_implicit=True)
    a = st.a
    ephi = {k: np.zeros((n, len(GROUP_NAMES))) for k in ("floor", "central", "declared")} if spec.track else None
    keys = np.zeros((n_draws, n), dtype=np.int64) if spec.track else None
    efd = np.zeros((n, 5))                         # E[C floor], E[C declared], E[1 tier at declared] x 3
    sizes = list(floors.SIZE_CLASSES)
    acc = np.zeros((n, 4))                         # E[C], E[1 status], E[1 summary], E[1 diagnosis]
    aux = np.zeros((n, 5))                         # E[1 diagnosis, declared], E[1 floor = minimal], E[1 nothing above floor],
    #                                                  E[1 linked < code-required], E[1 linked = 0]
    per_draw = np.zeros((n_draws, 5))              # mean C, status, summary, diagnosis, no tier
    per_draw_size = np.zeros((n_draws, len(sizes), 4))
    hist = np.zeros(len(HIST_BINS) - 1)
    ed = {"new_hist": None, "reach13_any": np.zeros(n, bool), "reach13_n": np.zeros(n)}
    L13 = vintage.ladder().index("2013")
    for k in range(n_draws):
        X, info = sample(spec, rng, rng2)
        hit = (X[:, 1:] >= TH).astype(float)
        efd[:, 0] += info["sF"][:, 0]
        efd[:, 1] += info["sD"][:, 0]
        efd[:, 2:] += info["sD"][:, 1:] >= TH
        if spec.track:
            pf, pdd = phi(info["F"]), phi(info["D"])
            ephi["floor"] += pf
            ephi["declared"] += pdd
            if "Cmask" in info:
                ephi["central"] += phi(info["Cmask"])
            else:
                lo_p = np.zeros_like(pf) if spec.discount_floor else pf
                ephi["central"] += lo_p + (pdd - lo_p) * info["r"][:, None]
            keys[k] = (info.get("F_eff", info["F"]) << len(CLS)) | info.get("D_eff", info["D"])
        acc[:, 0] += X[:, 0]
        acc[:, 1:] += hit
        aux[:, 0] += info["sD"][:, 3] >= TH
        aux[:, 1] += np.abs(info["sF"][:, 0] - F_MIN) < 1e-9
        aux[:, 2] += info["sD"][:, 0] <= info["sF"][:, 0] + 1e-12   # nothing installed above the floor
        aux[:, 3] += X[:, 0] < info["sF"][:, 0] - 1e-12
        aux[:, 4] += X[:, 0] <= 1e-12
        per_draw[k, 0] = (a * X[:, 0]).sum() / a.sum()
        per_draw[k, 1:4] = (a[:, None] * hit).sum(0) / a.sum()
        per_draw[k, 4] = 1 - per_draw[k, 1]
        for j, s in enumerate(sizes):
            m = st.size == s
            per_draw_size[k, j, 0] = (a[m] * X[m, 0]).sum() / a[m].sum()
            per_draw_size[k, j, 1:] = (a[m, None] * hit[m]).sum(0) / a[m].sum()
        hist += np.histogram(np.clip(X[:, 0], 0, 1 - 1e-12), bins=HIST_BINS, weights=a)[0]
        if "new_rung" in info:
            r13 = info["new_rung"] >= L13
            ed["reach13_any"] |= r13
            ed["reach13_n"] += r13
            h = np.bincount(info["new_rung"], weights=a, minlength=len(vintage.ladder()))
            ed["new_hist"] = h if ed["new_hist"] is None else ed["new_hist"] + h
    if spec.drop_implicit:
        _configure_vintage(drop_implicit=False)
    E = acc / n_draws
    AUX = aux / n_draws
    EFD = efd / n_draws
    W, sq = st.W, st.sq

    def jack(x, m=None):
        m = np.ones(n, bool) if m is None else m
        aa = W[m] * sq[m, None]
        v = (aa * x[m, None]).sum(0) / aa.sum(0)
        return float(v[0]), float(np.sqrt(((v[1:] - v[0]) ** 2).sum()))

    out = {"spec": spec, "n_draws": n_draws, "seed": f"({MASTER_SEED}, crc32('{spec.name}'))",
           "mean C": jack(E[:, 0]), "status": jack(E[:, 1]), "summary": jack(E[:, 2]),
           "diagnosis": jack(E[:, 3]), "no tier": jack(1 - E[:, 1]),
           "pct": {q: np.percentile(per_draw, q, axis=0) for q in (5, 50, 95)},
           "draw_mean": per_draw.mean(0), "hist": hist / n_draws / a.sum(), "E": E,
           "diagnosis declared": jack(AUX[:, 0]), "at minimal floor": jack(AUX[:, 1]),
           "nothing above floor": jack(AUX[:, 2]),
           "EFD": EFD, "AUX": AUX, "phi": ({k: v / n_draws for k, v in ephi.items()} if spec.track else None),
           "keys": keys,
           "H2b": float((a * E[:, 1]).sum() / a.sum() / max((st.W[:, 0] * E[:, 1]).sum() / st.W[:, 0].sum(), 1e-12))}
    out["size"] = {s: {"mean C": jack(E[:, 0], st.size == s), "status": jack(E[:, 1], st.size == s),
                       "summary": jack(E[:, 2], st.size == s), "diagnosis": jack(E[:, 3], st.size == s),
                       "p5": np.percentile(per_draw_size[:, j], 5, axis=0),
                       "p95": np.percentile(per_draw_size[:, j], 95, axis=0)}
                   for j, s in enumerate(sizes)}
    if ed["new_hist"] is not None:
        out["edition"] = {"share_by_rung": ed["new_hist"] / n_draws / a.sum(),
                          "reach13_any": float((a * ed["reach13_any"]).sum() / a.sum()),
                          "reach13_expected": jack(ed["reach13_n"] / n_draws)}
    return out


def deterministic(reading: str = "primary rows") -> dict:
    """The deterministic central for a reading, in the same output shape (no draws)."""
    X, _ = sample(Spec("det-" + reading, reading=reading), seed_for("det"))
    st = stock()
    a = st.a
    hit = (X[:, 1:] >= TH).astype(float)

    def jack(x):
        aa = st.W * st.sq[:, None]
        v = (aa * x[:, None]).sum(0) / aa.sum(0)
        return float(v[0]), float(np.sqrt(((v[1:] - v[0]) ** 2).sum()))
    return {"mean C": jack(X[:, 0]), "status": jack(hit[:, 0]), "summary": jack(hit[:, 1]),
            "diagnosis": jack(hit[:, 2]), "no tier": jack(1 - hit[:, 0]), "X": X,
            "hist": np.histogram(np.clip(X[:, 0], 0, 1 - 1e-12), bins=HIST_BINS, weights=a)[0] / a.sum()}


# ── the per-building draw runs (earlier design) ──────────────────────────────────────────────────
BUNDLE = frozenset({"Lgt", "Pwr", "Sub", "Sub2"})          # the §8.4.3 bundle (atlas.stock.BUNDLE)
