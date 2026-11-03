"""Multiple imputation of the unobserved layers, combined with Rubin's rules.

CBECS observes each building's size, type, period, controls and region. It does not observe the code edition in
force, whether each stream is recorded, or whether recorded streams are linked and described. Those layers are
treated as missing data and multiply imputed from external evidence (``tables/G_imputation_models.csv``):

  edition               state ∝ 2018 commercial energy use within the division, year uniform in the period
                        (state adoption tables: treated as fixed administrative counts)
  BAS linkage           type all / none / partial and per-class rates from the 37 census graphs with data:
                        proper imputation, i.e. each imputation refits them on a census bootstrap resample and
                        draws rare-class rates from their Beta posteriors
  interval meter        the state's 2018 commercial AMI share (EIA-861; fixed administrative counts)
  imputed answers       every EIA-imputed control answer redrawn from reported donors in its type × size ×
                        period cell (coarser cells below 10 donors)
  thermostat route, §8.4.3 recording: read from the survey and the code text (no imputation)
Compliance c and description d are scenario dimensions, reported as functions, not imputed.

Each imputation m draws every building's missing layers once and computes each estimate Q_m with its CBECS
jackknife variance U_m (151 replicate weights). Rubin's rules: Q̄, W = mean U_m, B = between-imputation
variance, T = W + (1 + 1/M) B; Barnard–Rubin degrees of freedom with complete-data ν_com = 150 (replicates − 1);
fraction of missing information λ̂ = (r + 2/(ν + 3)) / (r + 1), r = (1 + 1/M) B / W.

Each imputation keeps its own seed, so every grid point (c, d, δ) re-uses the same uniforms (common random numbers).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from atlas import config, declared, draws, floors, linkage, recording, stock as stk, tiers, vintage
from atlas.replicates import REPLICATES

OUT = config.RESULTS / "estimates"
MASTER_SEED = 20261017
M_DEFAULT = 200
NU_COM = 150
CLS, K = draws.CLS, len(draws.CLS)
SCEN = {"strict (low)": "strict", "primary": "primary rows", "enriched (high)": "enriched"}
CONTROLS = {"strict": "primary", "primary rows": "primary", "enriched": "enriched"}
IMPUTED_VARS = ("EMCS", "EMCSHT", "EMCSCL", "EMCSLT", "SMRTTHRM", "OCSN", "DCV", "VAV", "DOAS", "DAYHARV")
BAS_BLOCK = ("EMCS", "EMCSHT", "EMCSCL", "EMCSLT", "SMRTTHRM")
MIN_DONORS = 10
TSP, WHOLE, OCC = draws.to_mask(("T", "Sp")), draws.to_mask(("Whole",)), draws.to_mask(("Occ",))


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng([MASTER_SEED, *(zlib.crc32(str(k).encode()) for k in key)])


# ── redraw EIA-imputed control answers from reported donors ──────────────────────────────────────
@lru_cache(maxsize=1)
def _cells() -> list[np.ndarray]:
    df = draws.stock().df
    size = floors.size_class(df.SQFT).to_numpy()
    pba, yr = df.PBA.to_numpy(), df.YRCONC.to_numpy()
    return [pd.factorize(pd.Series(list(zip(pba, size, yr))))[0], pd.factorize(pd.Series(list(zip(pba, size))))[0],
            pd.factorize(pd.Series(pba))[0], np.zeros(len(df), int)]


def redraw_imputed(rng: np.random.Generator, log: dict | None = None) -> pd.DataFrame:
    """A copy of the survey with every EIA-imputed control answer redrawn from a reported donor."""
    df = draws.stock().df
    out = df.copy()
    cells = _cells()

    def donor_pool(target_mask, donor_mask, i):
        for level, cell in enumerate(cells):
            pool = np.flatnonzero(donor_mask & (cell == cell[i]))
            if len(pool) >= MIN_DONORS or level == len(cells) - 1:
                if log is not None:
                    log[level] = log.get(level, 0) + 1
                return pool
    # the BAS block moves together from one donor when EMCS itself was imputed (keeps the skip pattern intact)
    z_emcs = (df.ZEMCS == 1).to_numpy()
    donors = (~z_emcs) & df.EMCS.notna().to_numpy()
    for i in np.flatnonzero(z_emcs):
        pool = donor_pool(None, donors, i)
        j = pool[rng.integers(len(pool))]
        for v in BAS_BLOCK:
            out.iat[i, out.columns.get_loc(v)] = df[v].iat[j]
    for v in IMPUTED_VARS:
        z = (df[f"Z{v}"] == 1).to_numpy().copy()
        if v in BAS_BLOCK:
            z &= ~z_emcs
        if not z.any():
            continue
        reported = (df[f"Z{v}"] != 1).to_numpy() & df[v].notna().to_numpy()
        emcs = out.EMCS.to_numpy()
        for i in np.flatnonzero(z):
            dm = reported & ((emcs == emcs[i]) if v in BAS_BLOCK else True)
            pool = donor_pool(None, dm, i)
            j = pool[rng.integers(len(pool))]
            out.iat[i, out.columns.get_loc(v)] = df[v].iat[j]
    return out


# ── imputation-model parameters (proper imputation) ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def _census_arrays():
    g = linkage.graphs()
    dec = np.array([[c in s for c in CLS] for s in g.declared], dtype=float)
    lin = np.array([[c in s and c in d for c in CLS] for s, d in zip(g.linked, g.declared)], dtype=float)
    ty = g.type.map({"all": 0, "none": 1, "partial": 2, "-": -1}).to_numpy()
    return dec, lin, ty


RATES = {"basis": "partial"}   # a partial building's class rates come from the partial graphs (the Method)


def draw_params(rng: np.random.Generator, prior: str = "beta11", degenerate: bool = False,
                drop: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Linkage-model parameters for one imputation: the all / none / partial type shares over the reference set's graphs
    with C_declared > 0, and each class's linked share among the declarations of the *partial* graphs (the Method's
    definition; an earlier design used linked / declared over all 37 graphs, which RATES["basis"] = "all" restores).
    Classes with fewer than MIN_DECL declarations take a Beta prior updated by their counts.

    ``drop`` (sensitivities S21/S22): graphs treated as missing. They leave the fit like the graphs with C_declared = 0
    (type -1), and the bootstrap still resamples all 45 graphs, so the random stream stays the primary imputation's."""
    dec, lin, ty = _census_arrays()
    if drop:
        ty = np.where(linkage.graphs().building.isin(drop).to_numpy(), -1, ty)
    n = len(ty)
    mult = np.ones(n) if degenerate else np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    pos = ty >= 0
    w = mult * pos
    types = np.array([(w * (ty == t)).sum() for t in (0, 1, 2)]) / w.sum()
    wr = w * (ty == 2) if RATES["basis"] == "partial" else w
    d = (wr[:, None] * dec).sum(0)
    l = (wr[:, None] * lin).sum(0)
    a, b = linkage.PRIORS[prior]
    q = np.where(d >= linkage.MIN_DECL, l / np.maximum(d, 1), np.nan)
    rare = d < linkage.MIN_DECL
    q[rare] = (a + l[rare]) / (a + b + d[rare]) if degenerate else rng.beta(a + l[rare], b + d[rare] - l[rare])
    return types, q


# ── one imputation ───────────────────────────────────────────────────────────────────────────────
@dataclass
class Imp:
    key: tuple
    types: np.ndarray
    rates: np.ndarray
    E: dict          # controls -> declared-control mask per building
    R: dict          # controls -> route masks per building
    rows: dict       # Table C row id -> (applies before compliance, bits, compliance row?)
    sidx: np.ndarray
    df: pd.DataFrame  # the columns H4 needs (redrawn EMCS scope)
    new_rung: np.ndarray = None   # imputed construction edition (rung on the Table V ladder)
    emcs: np.ndarray = None       # BAS reported (redrawn answers)
    latest_rung: np.ndarray = None  # latest edition among the building's systems (construction or renovation)


def make_imp(scenario: str, m: int, prior: str = "beta11", degenerate: bool = False, redraw: bool = True,
             log: dict | None = None, state_weighted: bool = True, floor_reading: str = "main",
             extra_controls: tuple = (), drop_graphs: tuple = ()) -> Imp:
    """One imputation. The keyword arguments are the sensitivities (S1 states equal, S3/S4 floor reading,
    S5 extra controls reading, S6 prior, S21/S22 reference-set graphs dropped); S2 (no implicit rows) is set by the
    caller with ``draws._configure_vintage``.
    The random stream is the same as the primary imputation m, so a sensitivity differs only in what it varies."""
    key = ("imp", scenario, m) if not degenerate else ("imp-degenerate", scenario)
    rng = _rng(*key)
    types, rates = draw_params(rng, prior, degenerate, drop_graphs)
    df_m = redraw_imputed(rng, log) if (redraw and not degenerate) else draws.stock().df
    if degenerate:
        rungs, sidx = draws.stock().rungs, _rng("state", scenario).integers(0, 1, 1)
        sidx = _det_states()
    else:
        rungs, sidx = draws.draw_rungs(rng, weighted=state_weighted)
    detail: dict = {}
    draws.floor_masks(rungs, None, floor_reading, detail, held=True)   # held rows keep their place in the random stream
    comply_ids = {r["id"] for r in draws.floor_rows(floor_reading, held=True) if r["comply"]}
    rows = {rid: (ok, bits, rid in comply_ids) for rid, (ok, bits) in detail.items()}
    if draws.TIMING["monitoring_901_only"] and not degenerate:   # monitoring clause only where 90.1 itself was in force
        fam = draws.family_901_table()[draws.LAST["sidx"], draws.LAST["year"] - draws.YEARS[0]]
        meter_ids = {r["id"] for r in floors.table_c() if r["reading"] == "meter_clause"}
        rows = {rid: ((ok & fam) if rid in meter_ids else ok, bits, c) for rid, (ok, bits, c) in rows.items()}
    E, R = {}, {}
    for ctl in ("primary", "enriched", *extra_controls):
        E[ctl] = np.array([draws.to_mask(s) for s in declared.declared_streams(df_m, ctl)], dtype=np.int64)
        R[ctl] = recording.routes_df(df_m, ctl)
    from atlas import distributions
    hdf = df_m[["YRCONC", "PBA", "SQFT"]].assign(SIZE=floors.size_class(df_m.SQFT), SCOPE=distributions.emcs_scope(df_m))
    latest = np.max(np.vstack([np.asarray(v) for v in rungs.values()]), axis=0)
    return Imp(key, types, rates, E, R, rows, sidx, hdf, np.asarray(rungs["new"]), (df_m.EMCS == 1).to_numpy(), latest)


@lru_cache(maxsize=1)
def _det_states() -> np.ndarray:
    """Degenerate imputation: the most energy-weighted state of each division, for every building."""
    st = draws.stock()
    cd = st.df.CENDIV.to_numpy().astype(int)
    out = np.empty(len(cd), int)
    for d, (idx, p) in draws.division_draw(True).items():
        out[cd == d] = idx[int(np.argmax(p))]
    return out


def _shift(p, delta):
    lo = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    return 1 / (1 + np.exp(-(lo + delta)))


def shifted_types(types: np.ndarray, delta: float) -> np.ndarray:
    """Tipping (a): the log-odds of 'all' against 'none' move by δ, of 'partial' against 'none' by δ/2."""
    w = types * np.exp(np.array([delta, 0.0, delta / 2]))
    return w / w.sum()


@dataclass(frozen=True)
class Cfg:
    scenario: str = "primary"
    c: float = 1.0
    d: float = 1.0
    da: float = 0.0
    db: float = 0.0
    add: frozenset = frozenset()
    add_scope: str = "all"
    controls: str = ""           # S5: override the Table E reading (e.g. "wo")
    smart_credit: bool = True    # S9
    flagged: str = "provisional" # S10: "credited"
    rec_thermo: bool = True      # gap split: with all four off the linked level equals the declared level
    rec_meter: bool = True
    rec_route: bool = True
    link: bool = True
    ami_year: int = 2018         # S12: commercial advanced-meter shares of a later EIA-861 year
    scoring: str = "primary"     # "hvac" = the HVAC-credit scoring
    occ_any: bool = False        # optimistic: occupancy streams recorded without BAS lighting control
    big_meter: bool = False      # S14: an interval meter credited for every building of at least 25,000 ft2
    delta_scope: str = "all"     # the reference-set shift applied to "all", "small" (< 25,000) or "large"
    desc_scope: str = "nonbas"   # design: d applies to non-BAS routes only; "all" = the earlier design
    desc_unit: str = "stream"    # variant: "building" = one description draw per building
    meter_measure: str = "ami"   # "daily" = commercial customers with daily digital access
    alt_lighting: bool = False   # S24: credit C06 (2016 lighting alterations), held out of the primary reading


@lru_cache(maxsize=1)
def held_ids() -> frozenset[str]:
    """Clause rows held in the imputation but applied only under Cfg.alt_lighting (C06)."""
    tags = frozenset().union(*floors.HELD.values())
    return frozenset(r["id"] for r in floors.table_c() if r["reading"] in tags)


@lru_cache(maxsize=1)
def elec_log_ids() -> frozenset[str]:
    """Monitoring rows that log total electricity, so the utility meter counts as recorded (§8.4.3, C08). C15 (§10.4.5,
    district thermal energy) records Sub only."""
    return frozenset(r["id"] for r in floors.table_c() if r["system"] == "electrical")


@lru_cache(maxsize=1)
def clause_scope() -> np.ndarray:
    """Where §8.4.3 applies by size and type: at least 25,000 ft² and not a hotel, motel or restaurant (exception f).
    The lever's "25k" scope adds the monitoring bundle here."""
    return floors._condition(draws.stock().df, "sqft>=25000&not-hotel-motel-restaurant").to_numpy()


def evaluate(imp: Imp, cfg: Cfg) -> dict:
    """One completed dataset: masks and scores of the code-required, declared and linked levels per building."""
    st = draws.stock()
    n = len(st.sq)
    reading = SCEN[cfg.scenario]
    ctl = cfg.controls or CONTROLS[reading]
    strict = reading == "strict"
    rng = _rng(*imp.key, "uniforms")
    comply_u = {rid: rng.random(n) for rid in sorted(imp.rows) if imp.rows[rid][2] and rid not in floors.OWN_STREAM}
    comply_u |= {rid: _rng(*imp.key, "uniforms", rid).random(n)   # own stream, so no other draw moves
                 for rid in sorted(imp.rows) if imp.rows[rid][2] and rid in floors.OWN_STREAM}
    ami_u, type_u = rng.random(n), rng.random(n)
    class_u, desc_u = rng.random((n, K)), rng.random((n, K))
    F = np.zeros(n, np.int64)
    ok_c = {}
    for rid, (ok, bits, comply) in imp.rows.items():
        if rid in held_ids() and not cfg.alt_lighting:   # drawn above (same stream), not applied
            continue
        okc = ok & (comply_u[rid] < cfg.c) if comply else ok
        ok_c[rid] = (okc, bits)
        F |= np.where(okc, bits, 0)
    E = imp.E[ctl]
    add = np.zeros(n, np.int64)
    if cfg.add:
        add = np.where(clause_scope() if cfg.add_scope == "25k" else np.ones(n, bool), draws.to_mask(cfg.add), 0)
        F, E = F | add, E | add
    D = F | E
    R = imp.R[ctl]
    RD, BASR = np.zeros(n, np.int64), np.zeros(n, np.int64)
    code_log = np.zeros(n, bool)
    for rid, (ok, bits) in ok_c.items():
        if bits == TSP:
            if cfg.rec_thermo:
                BASR |= np.where(ok & R["bas_hvac"], TSP, 0)
                if cfg.smart_credit:
                    RD |= np.where(ok & R["smart"], TSP, 0)
            else:
                RD |= np.where(ok, TSP, 0)
        elif bits == WHOLE:
            continue
        elif bits == OCC:
            if cfg.rec_route:
                BASR |= np.where(ok & (R["bas_light"] | cfg.occ_any), OCC, 0)
            else:
                RD |= np.where(ok, OCC, 0)
        else:
            RD |= np.where(ok, bits, 0)
            if rid in elec_log_ids():           # §8.4.3 logs total electricity; §10.4.5 (C15) does not
                code_log |= ok
    c02 = next((ok for ok, bits in ok_c.values() if bits == WHOLE), np.zeros(n, bool))
    ami_p = recording.ami_probs(imp.sidx, cfg.ami_year, cfg.meter_measure)
    ami_hit = c02 & (ami_u < ami_p) & ~code_log
    if cfg.big_meter:                    # S14: large (demand-billed) customers have interval data whatever the AMI share
        big = st.sq >= 25000
        ami_hit = ami_hit | (c02 & big & ~code_log)
        ami_p = np.where(big, 1.0, ami_p)
    if cfg.rec_meter:
        RD |= np.where(c02 & ((ami_u < ami_p) | code_log), WHOLE, 0)
    else:
        RD |= np.where(c02, WHOLE, 0)
    if cfg.rec_route:
        BASR |= R["E_bas"] & E
        if cfg.flagged == "credited":
            RD |= R["E_flag_all"] & E
        else:
            BASR |= R["E_flag_bas"] & E
        if cfg.smart_credit:
            RD |= R["E_smart"] & E
        if cfg.occ_any:
            BASR |= R["E_occ_any"] & E
    else:
        BASR |= E
    RD |= add
    if strict:
        BASR &= ~draws.rare_mask()
    extra = draws.bits_matrix(BASR & ~RD)
    tp = shifted_types(imp.types, cfg.da) if cfg.da else imp.types
    q = _shift(imp.rates, cfg.db) if cfg.db else imp.rates
    if cfg.delta_scope != "all":         # the shift for one size group only; the other keeps the reference-set values
        grp = (st.sq >= 25000) if cfg.delta_scope == "large" else (st.sq < 25000)
        c0, c1 = np.cumsum(imp.types), np.cumsum(tp)
        ty = np.where(grp, np.searchsorted(c1, type_u * c1[-1], side="right"), np.searchsorted(c0, type_u * c0[-1], side="right"))
        qm = np.where(grp[:, None], q[None, :], imp.rates[None, :])
    else:
        cum = np.cumsum(tp)
        ty = np.searchsorted(cum, type_u * cum[-1], side="right")
        qm = q[None, :]
    sel = (ty == 0)[:, None] | ((ty == 2)[:, None] & (class_u < qm))
    if not cfg.link:
        sel = np.ones_like(sel)
    if cfg.d < 1.0:
        # a stream's description draw, or one draw per building for all its streams
        described = draws.pack(desc_u < cfg.d) if cfg.desc_unit == "stream" else \
            np.where(desc_u[:, 0] < cfg.d, (1 << K) - 1, 0).astype(np.int64)
    if cfg.d < 1.0 and cfg.desc_scope == "nonbas":
        # a reference-set link is a point both recorded and described, so BAS-route streams already
        # carry the description loss; d applies to the other routes only (smart thermostat, advanced meter, monitoring
        # clause, and any stream recorded outside the BAS). A non-BAS stream left undescribed may still arrive by the BAS.
        RDd = RD & described
        extra = draws.bits_matrix(BASR & ~RDd)
        C = RDd | draws.pack(extra & sel)
    else:
        C = RD | draws.pack(extra & sel)
        if cfg.d < 1.0:                    # earlier design: every linked stream drawn against d
            C &= described
    sc = cfg.scoring
    return {"F": F, "D": D, "C": C, "sF": draws.score(F, sc), "sD": draws.score(D, sc), "sC": draws.score(C, sc), "scoring": sc,
            "ami_p": ami_p, "ami_hit": ami_hit, "smart": R["smart"], "bas": imp.emcs, "new_rung": imp.new_rung}


# ── many statistics, jackknifed at once ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def weights() -> dict:
    st = draws.stock()
    mf = pd.read_csv(config.CBECS_CSV, usecols=["PUBID", "MFBTU"]).set_index("PUBID").MFBTU.reindex(st.df.index)
    energy = mf.fillna(0.0).to_numpy(dtype=float)
    return {"area": st.W * st.sq[:, None], "energy": st.W * energy[:, None], "count": st.W.copy(),
            "_mfbtu_missing": int(mf.isna().sum())}


XS = np.round(np.arange(0, 1.0001, 0.01), 2)
CUTOFFS = (0.40, 0.45, 0.50, 0.55, 0.60)


def columns(ev: dict, full: bool = True) -> dict[str, np.ndarray]:
    """Per-building columns whose weighted means are the reported statistics."""
    sF, sD, sC = ev["sF"], ev["sD"], ev["sC"]
    th = tiers.THRESHOLD - 1e-12
    col = {"mean code-required level": sF[:, 0], "mean declared level": sD[:, 0], "mean linked level": sC[:, 0],
           "status tier (linked)": (sC[:, 1] >= th) * 1.0, "summary tier (linked)": (sC[:, 2] >= th) * 1.0,
           "diagnosis tier (linked)": (sC[:, 3] >= th) * 1.0, "below the status tier (linked)": (sC[:, 1] < th) * 1.0,
           "diagnosis tier (declared)": (sD[:, 3] >= th) * 1.0,
           "linked below code-required": (sC[:, 0] < sF[:, 0] - 1e-12) * 1.0, "linked level zero": (sC[:, 0] <= 1e-12) * 1.0,
           "at the minimal code-required profile": (np.abs(sF[:, 0] - 20 / 76) < 1e-9) * 1.0}
    if full:
        for x in XS:
            col[f"linked at or above {x:.2f}"] = (sC[:, 0] >= x - 1e-12) * 1.0
        for cut in CUTOFFS:
            for j, t in enumerate(tiers.TIERS):
                col[f"{t} tier at cutoff {cut:.2f}"] = (sC[:, 1 + j] >= cut - 1e-12) * 1.0
    return col


def jackknife(cols: dict[str, np.ndarray], w: np.ndarray, seg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(Q, U, names): weighted means of every column over the rows in *seg*, at the full weight and each replicate."""
    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    A = w if seg is None else w * seg[:, None]
    num = A.T @ X
    den = A.sum(0)[:, None]
    est = num / den
    Q = est[0]
    U = ((est[1:] - Q) ** 2).sum(0)
    return Q, U, names


def h2b(ev: dict) -> tuple[float, float]:
    th = tiers.THRESHOLD - 1e-12
    s = (ev["sC"][:, 1] >= th) * 1.0
    W = weights()
    a = (W["area"].T @ s) / W["area"].sum(0)
    b = (W["count"].T @ s) / W["count"].sum(0)
    v = a / np.maximum(b, 1e-12)
    return float(v[0]), float(((v[1:] - v[0]) ** 2).sum())


def h4(imp: Imp, ev: dict, jack: bool = True) -> tuple[float, float]:
    R = pd.DataFrame({"central_C_central": ev["sC"][:, 0]}, index=imp.df.index)
    Wdf = _wdf() if jack else _wdf()[["FINALWT"]]
    df = pd.concat([imp.df, Wdf], axis=1)
    t = stk_h4(df, R, jack)
    return float(t.loc["vintage - type", "estimate"]), float(t.loc["vintage - type", "se"] ** 2) if jack else np.nan


@lru_cache(maxsize=1)
def _wdf() -> pd.DataFrame:
    st = draws.stock()
    return pd.DataFrame(st.W, index=st.df.index, columns=["FINALWT", *REPLICATES])


def stk_h4(df: pd.DataFrame, R: pd.DataFrame, jack: bool) -> pd.DataFrame:
    """atlas.stock.h4's 'with controls' decomposition on a frame that already carries SIZE and SCOPE."""
    y = R["central_C_central"].to_numpy(dtype=float)
    W = df[["FINALWT", *REPLICATES]].to_numpy(dtype=float) if jack else df[["FINALWT"]].to_numpy(dtype=float)
    S = stk.shapley(df, y, {"vintage": "YRCONC", "type": "PBA", "size": "SIZE", "controls": "SCOPE"}, W, df.SQFT.to_numpy(dtype=float))
    S["vintage - type"] = S["vintage"] - S["type"]
    est = S.iloc[0]
    se = np.sqrt(((S.iloc[1:] - est) ** 2).sum()) if jack else est * np.nan
    return pd.DataFrame({"estimate": est, "se": se})


HEADLINE_ROWS = (("area", "mean code-required level"), ("area", "mean declared level"), ("area", "mean linked level"),
                 ("area", "status tier (linked)"), ("area", "summary tier (linked)"), ("area", "diagnosis tier (linked)"),
                 ("area", "below the status tier (linked)"), ("area", "linked level zero"),
                 ("energy", "mean linked level"), ("count", "mean linked level"))


def largest_fmi(N: pd.DataFrame) -> tuple[float, int]:
    """The one 'largest fraction of missing information' quoted everywhere: the largest FMI among the primary
    estimates of the headline table (national.csv rows), and von Hippel's M = 1 + 200 FMI² (rounded up) for it."""
    P = N[N.scenario == "primary"].set_index(["weight", "statistic"])
    f = max(float(P.loc[k, "fmi"]) for k in HEADLINE_ROWS if P.loc[k, "T"] > 0)
    return f, int(np.ceil(1 + 200 * f ** 2))


# ── Rubin's rules ────────────────────────────────────────────────────────────────────────────────
def rubin(Q: np.ndarray, U: np.ndarray, nu_com: float = NU_COM) -> pd.DataFrame:
    """Q, U: (M, S). Returns per statistic Q̄, W, B, T, ν, λ̂ (fraction of missing information), 95 % interval."""
    from scipy import stats
    M = Q.shape[0]
    qbar = Q.mean(0)
    Wv = U.mean(0)
    Bv = Q.var(0, ddof=1) if M > 1 else np.zeros_like(qbar)
    T = Wv + (1 + 1 / M) * Bv
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(T > 0, (1 + 1 / M) * Bv / T, 0.0)
        nu_m = np.where(lam > 0, (M - 1) / lam ** 2, np.inf)
        nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
        nu = 1 / (1 / nu_m + 1 / nu_obs)
        r = np.where(Wv > 0, (1 + 1 / M) * Bv / Wv, np.inf)
        fmi = np.where(np.isfinite(r), (r + 2 / (nu + 3)) / (r + 1), 1.0)
    t = stats.t.ppf(0.975, np.maximum(nu, 1))
    half = t * np.sqrt(T)
    return pd.DataFrame({"estimate": qbar, "W": Wv, "B": Bv, "T": T, "se": np.sqrt(T), "df": nu, "fmi": fmi,
                         "lo95": qbar - half, "hi95": qbar + half, "M": M})
