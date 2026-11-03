"""The results ledger: every number the paper cites, by its claim ID, regenerated from the result tables.

Each row is (ID, claim, number, source). The IDs are the paper's; the numbers are recomputed from results/ and tables/
by `python -m atlas all` (or this script alone, once the results exist). Rows whose content is a fixed statement (a
method constant or a source fact) are given as text; every other number is read from a result or input table here.

    python scripts/build_ledger.py   -> results/ledger.md, results/ledger.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas import canon, cbecs, config, declared, draws, link_none, linkage, mi  # noqa: E402
from atlas.replicates import REPLICATES  # noqa: E402

RES, TAB = config.RESULTS, config.TABLES
EST, SCO, EDI, SCN, WGT = (RES / d for d in ("estimates", "scoring", "editions", "scenarios", "weighting"))
Z = 1.96
pct0 = lambda x: f"{100 * x:.1f}%"
f3 = lambda r: f"{r.estimate:.3f} [{r.lo95:.3f}–{r.hi95:.3f}]"
pct = lambda r: f"{100 * r.estimate:.1f}% [{100 * r.lo95:.1f}%–{100 * r.hi95:.1f}%]"
sg = lambda r: f"{r.estimate:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]"
sp = lambda r: f"{100 * r.estimate:+.1f} pp [{100 * r.lo95:+.1f}, {100 * r.hi95:+.1f}]"
ci = lambda r, f: f"{f(r.estimate)} [{f(r.lo95)}–{f(r.hi95)}]"
f3n = lambda x: f"{x:.3f}"
GROUPS = ("thermostat", "metering", "BAS HVAC points", "occupancy, lighting, CO2")
TIERS = ("status tier (linked)", "summary tier (linked)", "diagnosis tier (linked)")
SIZES = ("<= 5,000", "5,001-24,999", ">= 25,000")


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ── method constants and source facts ────────────────────────────────────────────────────────────
def method_rows() -> list[tuple]:
    import csv as _csv
    with open(TAB / "V_states.csv", encoding="utf-8", newline="") as f:
        n_states = len({r["state"] for r in _csv.DictReader(f)})
    c04 = canon.ceiling({"T", "Sp", "Whole"})
    g = linkage.graphs()
    over = int((g.c_ans_declared > g.c_ans_linked + 1e-9).sum())
    import math
    st = [k for k in canon.CANONS["v2"].rungs if k[1] == "status"]
    tot = sum(len(canon.CANONS["v2"].rungs[k]) for k in st)
    lv = lambda have: sum(canon.level(d, i, have) for d, i in st)
    base = {"T", "Sp", "Whole"}
    return [
        ("C-01", "Records in the 2018 CBECS public file", f"{config.N_RECORDS:,}", "atlas/config.py; atlas/tests/test_cbecs.py"),
        ("C-02", "Weighted stock (reproduces EIA Table B1)",
         "5,918 thousand buildings; 96,527 million ft² (B1: 96,528; SQFT top-coded at 2.1 M ft² for 20 records)",
         "atlas/tests/test_cbecs.py"),
        ("C-03", "Sampling-variance method",
         "CBECS jackknife, 151 replicate weights, α = 1; reproduces EIA's worked example (2,832,884 buildings, SE 139,667)",
         "atlas/replicates.py; atlas/tests/test_replicates.py (EIA, User's Guide to the 2018 CBECS microdata)"),
        ("C-06", "Smart-thermostat question universe", "asked only of buildings without a BAS (EMCS = 2)",
         "atlas/cbecs.py UNIVERSE; atlas/tests/test_cbecs.py"),
        ("C-07", "Canon",
         "the readiness framework's canon of 76 rungs: 31 question cells (domain × intent), each answerable at up to three "
         "levels of completeness; copied with citation", "atlas/canon.py"),
        ("C-08", "C(I) of the 2004 code profile (zone temperature, setpoint, whole-building meter)",
         f"{c04.score:.3f} ({c04.achieved} of 76)", "atlas/canon.py; atlas/tests/test_floors.py"),
        ("C-09", "First 90.1 edition with electrical energy monitoring (§8.4.3)",
         "90.1-2013; new buildings ≥ 25,000 ft²; total, HVAC, interior/exterior lighting, receptacles; 15-min recording, "
         "36 months kept", "tables/C_clauses.csv C08; tables/C_sources.csv S4, S13, S14"),
        ("C-13", "Reference-set graphs declaring more than they link", f"{over} of {len(g)}", "tables/R_reference_set.csv"),
        ("C-14", "Tier definition (this study's, not the readiness framework's)",
         "tier X reached when graded coverage of the canon's X-intent cells ≥ 0.5; discount applied per intent", "atlas/tiers.py"),
        ("C-46", "CBECS construction-year information", "8 categorical bins (before 1946 … 2013–2018); no year",
         "CBECS codebook YRCONC; atlas/vintage.py"),
        ("C-47", "Geography in the public file", "4 census regions, 9 census divisions; no state", "CBECS codebook REGION/CENDIV"),
        ("C-48", "State code-adoption evidence behind the edition table",
         f"{n_states} jurisdictions (50 states + DC) from PNNL-19887, PNNL-31437, PNNL-25611 and DOE status snapshots "
         "(2002, 2004, 2010, 2017)", "tables/V_states.csv"),
        ("C-50", "Canon copy check against the readiness framework's published ceilings",
         "matches the readiness framework's published ceilings of 7, 12, 20, 49 and 37 of 76 (its minimal, comfort and "
         "standard profiles, its metered offices and its small offices) on the corresponding stream-class sets, and "
         "recomputes all 90 ceilings of its 45-graph census", "atlas/tests/test_floors.py, test_declared.py"),
        ("C-52", "Scope of the measure",
         "this paper estimates the answerable-readiness ceiling C(I) only; the readiness framework's realized readiness "
         "multiplies C(I) by an AI agent's accuracy A, which this paper does not estimate (no agents, no model calls)",
         "atlas/canon.py"),
        ("C-61", "Newest CBECS microdata",
         "the 2018 cycle remains the latest CBECS with released microdata; EIA's status page (latest items December 2024) "
         "describes exploratory surveys that inform the next CBECS",
         "EIA, CBECS status page, eia.gov/consumption/commercial (read 2026-10)"),
        ("C-170", "CBECS 2018 metering items",
         "none: no question on meter type, interval data, demand metering or submetering (nearest: energy-management plan, "
         "demand-responsive lighting, BAS)", "CBECS 2018 microdata codebook (searched)"),
        ("C-172", "State drawn by climate",
         "not done: the county ASHRAE 169-2021 zones behind CBECS's climate categories are not public; the free 169-2013 "
         "lists miss CBECS's division climate shares beyond 2 SE in 8 of 33 cells",
         "feasibility check, not part of the analysis"),
        ("C-259", "Status tier: answer levels of the status question types; levels needed for the tier (coverage of one half); "
                  "levels reached by zone temperature, setpoint and a whole-building meter; with occupancy added; with occupancy "
                  "and lighting added",
         f"{tot}; {math.ceil(tot / 2)}; {lv(base)}; {lv(base | {'Occ'})}; {lv(base | {'Occ', 'Lgt'})}",
         "atlas/canon.py (primary scoring)"),
        ("C-221", "Occupancy sensing in the code-required level (primary reading)",
         "from 90.1-2013 (Table 9.6.1 (h)). 90.1-2004 (addendum ae to 90.1-2001) requires a control device that turns "
         "lighting off within 30 minutes of occupants leaving, in three space types (college classrooms, conference/meeting "
         "rooms, employee lunch and break rooms), names no timer switch and states no building size; 90.1-2010 (addendum x "
         "to 90.1-2007) allows an occupant sensor or a timer switch in eight space types. Before 2013 the requirement "
         "covered only space types the survey cannot identify, so the primary reading starts at 2013; 2004 and 2010 are a "
         "sensitivity", "tables/C_clauses.csv C03–C05; tables/C_sources.csv S8, S10"),
    ]


# ── survey and reference-set facts ───────────────────────────────────────────────────────────────
def survey_rows() -> list[tuple]:
    df = cbecs.load(replicates=True)
    W = df[["FINALWT", *REPLICATES]].to_numpy(dtype=float)
    emcs = (df.EMCS == 1).to_numpy(dtype=float)
    from atlas import floors
    size = floors.size_class(df.SQFT).to_numpy()
    share = lambda m: (W.T @ (m * emcs)) / (W.T @ m)
    jk = lambda v: (float(v[0]), float(np.sqrt(((v[1:] - v[0]) ** 2).sum())))
    cip = lambda es, f=pct0: f"{f(es[0])} [{f(es[0] - Z * es[1])}–{f(es[0] + Z * es[1])}]"
    by_size = [jk(share((size == s).astype(float))) for s in SIZES]
    yr = df.YRCONC.to_numpy()
    diff = jk(share(np.isin(yr, (8, 9)).astype(float)) - share(np.isin(yr, (2, 3)).astype(float)))
    a = df.FINALWT * df.SQFT
    exempt = float(a[df.SQFT < 25000].sum() / a.sum())
    A = csv(RES / "reference_set" / "assumption_check.csv").drop_duplicates("declares").set_index("declares")
    order = ["SAT", "OAT", "Run", "Load", "T", "ActC", "Sp", "Sub", "Pwr", "ActP", "Occ", "CO2", "SpC", "Lgt"]
    g = linkage.graphs()
    pos, tc = g[g.in_R1], linkage.type_counts()
    part = g[g.type == "partial"].ratio
    am = csv(TAB / "L_ami_states.csv")
    return [
        ("C-24", "H2a: building share with a BAS by size class (≤ 5,000 / 5,001–24,999 / ≥ 25,000 ft²)",
         " / ".join(cip(x) for x in by_size), "CBECS 2018 (EMCS, FINALWT and its 151 replicate weights)"),
        ("C-25", "H2a: BAS share, built 2000–2018 minus pre-1960",
         cip(diff, lambda x: f"{100 * x:+.1f} pp") + " → H2a supported", "CBECS 2018 (EMCS, YRCONC)"),
        ("C-58", "Reference-set evidence: real automated-building graphs (of 45) declaring each stream class",
         "; ".join(f"{k} {A.loc[k, 'declared_in_census']}" for k in order if k in A.index),
         "results/reference_set/assumption_check.csv; tables/R_reference_set.csv"),
        ("C-67", "Floor-area share below the §8.4.3 size threshold (< 25,000 ft²)", pct0(exempt), "CBECS 2018 (SQFT, FINALWT)"),
        ("C-76", "Reference-set linkage types",
         f"of the {len(pos)} census graphs with C_declared > 0: {tc['all']} link everything, {tc['none']} nothing, "
         f"{tc['partial']} partly (by C_linked / C_declared); mean of per-graph ratios {pos.ratio.mean():.3f} against R1 "
         f"(ratio of sums) {declared.discount()['R1']:.3f}", "tables/R_linkage_graphs.csv (atlas.linkage)"),
        ("C-92", "Commercial advanced-meter (AMI) share, 2018 (EIA-861)",
         f"{int(am.ami_commercial.sum()):,} of {int(am.meters_commercial.sum()):,} commercial meters "
         f"({pct0(am.ami_commercial.sum() / am.meters_commercial.sum())}) nationally, equal to EIA Table 10.10; states "
         f"{pct0(am.ami_share.min())}–{pct0(am.ami_share.max())}", "tables/L_ami_states.csv (Form EIA-861 2018)"),
        ("C-105", "Linked shares (C_linked / C_declared) of the 6 partial reference-set graphs",
         f"{part.min():.2f}–{part.max():.2f} (" + ", ".join(f"{x:.3f}" for x in sorted(part)) + ")", "tables/R_linkage_graphs.csv"),
    ]


# ── the primary result and its checks (results/estimates) ────────────────────────────────────────
def estimate_rows() -> list[tuple]:
    N, RD, H, CU = csv(EST / "national.csv"), csv(EST / "reads_decomposition.csv"), csv(EST / "h5_lever.csv"), csv(EST / "cuts.csv")
    TT, MC, DC = csv(EST / "transfer_tipping.csv"), csv(EST / "model_checks.csv"), csv(EST / "donor_cells.csv")
    G, GE, ZP = csv(EST / "gap.csv"), csv(EST / "geography.csv"), csv(EST / "zero_profile.csv")
    E, DS, S, CW = csv(EST / "editions_imputed.csv"), csv(EST / "decomposition_scenarios.csv"), csv(EST / "sensitivities.csv"), csv(EST / "cuts_weights.csv")
    pr = "primary"
    e = lambda scen, stat, w="area": N[(N.scenario == scen) & (N.weight == w) & (N.statistic == stat)].iloc[0]
    rdv = lambda grp: RD[(RD.scenario == pr) & (RD.group == grp)].iloc[0]
    lv = lambda item, stat: H[(H.group == "lever") & (H.item == item) & (H.statistic == stat)].iloc[0]
    h5 = lambda item, stat: H[(H.group == "h5") & (H.item == item) & (H.statistic == stat)].iloc[0]
    cut = lambda var, seg, stat: CU[(CU.scenario == pr) & (CU.weight == "area") & (CU.cut == var) & (CU.segment == seg) & (CU.statistic == stat)].iloc[0]
    gap = lambda stat, w="area": G[(G.weight == w) & (G.statistic == stat)].iloc[0]
    sv = lambda sid, stat: S[(S.id == sid) & (S.statistic == stat)].iloc[0]
    h1 = TT[(TT.variant == "ab") & TT.conclusion.str.contains("H1")].iloc[0]
    up = h1["interval includes opposite (up)"]
    h1s = "holds over δ ∈ [−3, 3]" if str(up) == "none in range" else f"reverses at δ = {float(up):+.2f} (odds ratio {np.exp(float(up)):.1f})"
    en = E[E.cut == "national"].set_index("statistic")
    r13 = en.loc["2013", "estimate"] + en.loc["2016", "estimate"]
    import csv as _csv
    vs = list(_csv.DictReader(open(TAB / "V_states.csv", encoding="utf-8")))
    n13 = lambda y: len({r["state"] for r in vs if r["kind"] == "adopted" and r["edition"] in ("2013", "2016") and int(r["year_in_force"]) <= y})
    dec = DS[DS.scenario == pr].set_index("statistic")
    types = ("Office", "Nonrefrigerated warehouse", "Education", "Inpatient health care", "Religious worship")
    periods = list(dict.fromkeys(CU[(CU.cut == "YRCONC")].segment))
    ge = GE[GE.cut == "CENDIV"]
    gev = lambda seg, stat: ge[(ge.segment == seg) & (ge.statistic == stat)].iloc[0].estimate
    divs = ge[ge.statistic == "mean linked level"].sort_values("estimate", ascending=False).segment
    za = ZP[ZP.cut == "attribute"]
    zv = lambda pop, stat: za[(za.population == pop) & (za.statistic == stat)].iloc[0].estimate
    cwz = lambda var, seg: CW[(CW.weight == "area") & (CW.cut == var) & (CW.segment == seg) & (CW.statistic == "linked level zero")].iloc[0]
    st_ = draws.stock()
    wh_share = float(st_.a[(st_.df.PBA == 5).to_numpy()].sum() / st_.a.sum())   # PBA 5 = nonrefrigerated warehouse
    tip = lambda df, conc, col: "; ".join(f"{r.get('axis', r.get('variant', ''))}: {r[col]}" for _, r in df[df.conclusion.str.contains(conc, regex=False)].iterrows())
    drops = ("drop: thermostat rule", "drop: meter rule", "drop: BAS and network route rule", "drop: BAS linkage draw")
    fmi, m_rule = mi.largest_fmi(N)
    piv = ge.pivot_table(index="segment", columns="statistic", values="estimate")
    lab = {1: "New England", 2: "Middle Atlantic", 3: "East North Central", 4: "West North Central", 5: "South Atlantic",
           6: "East South Central", 7: "West South Central", 8: "Mountain", 9: "Pacific"}
    area = pd.Series({lab[d]: float(st_.a[(st_.df.CENDIV == d).to_numpy()].sum()) for d in lab})
    w = area.reindex(piv.index).to_numpy()

    def wcorr(x, y):
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        return float(np.average((x - mx) * (y - my), weights=w) /
                     np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))
    yl = piv["mean linked level"].to_numpy()
    r_ami, r_bas = wcorr(yl, piv["advanced-meter share (expected)"].to_numpy()), wcorr(yl, piv["BAS share"].to_numpy())
    T3 = TT[(TT.axis == "delta") & (TT.variant == "ab") & TT.conclusion.str.contains("construction period")].iloc[0]
    lo_ = MC.loc[MC["two-sided p"].idxmin()]
    COUNTS = {"all": "number of fully linked graphs", "none": "number of graphs linking none", "partial": "number of partly linked graphs"}
    pm = csv(EST / "partial_means.csv").iloc[0]
    PP = csv(EST / "model_check_partial.csv")
    pa = PP[PP["declared streams"].astype(str) == "all"].iloc[0]
    src = "results/estimates/"
    return [
        ("C-116", "Transfer tipping point for 'most floor area reaches no tier' (type and class rates shifted together)", h1s,
         src + "transfer_tipping.csv"),
        ("C-118", "Model checks: observed reference-set statistics inside the simulated 95 % ranges", f"{int(MC.inside.sum())} of {len(MC)}",
         src + "model_checks.csv"),
        ("C-267", "Model checks: posterior predictive two-sided value (2 × the smaller tail over the 2,000 simulated reference sets) "
                  "— the smallest, its statistic and its two tails; statistics with a value below 0.05",
         f"{lo_['two-sided p']:.3f} ({COUNTS.get(lo_.statistic, lo_.statistic)}: observed {lo_.observed:g}, "
         f"Pr(sim ≥ obs) = {lo_['Pr(sim >= obs)']:.3f}, Pr(sim ≤ obs) = {lo_['Pr(sim <= obs)']:.3f}); "
         f"{int((MC['two-sided p'] < 0.05).sum())} of {len(MC)}", src + "model_checks.csv"),
        ("C-268", "Model checks, partly linked graphs: mean number of reference-set graphs simulated as the partial type; mean "
                  "number of them ending strictly between a linked share of 0 and 1; shares of the partial-type graphs ending at "
                  "0 and at 1",
         f"{pm['mean partial-class graphs']:.2f}; {pm['mean strictly between']:.2f}; "
         f"{100 * pa['share ends at 0']:.1f}% and {100 * pa['share ends at 1']:.1f}%", src + "model_check_partial.csv, partial_means.csv"),
        ("C-121", "EIA-imputed control answers: share of redraws from the full type × size × period cell",
         " / ".join(f"{r.scenario} {pct0(r.share)}" for _, r in DC[DC["donor cell"] == "type x size x period"].iterrows()),
         src + "donor_cells.csv"),
        ("C-124", "Floor area at the minimal code-required profile", ci(e(pr, "at the minimal code-required profile"), pct0), src + "national.csv"),
        ("C-125", "Floor area by imputed construction edition: 90.1-2013 or later",
         f"{pct0(r13)} (2013 {pct0(en.loc['2013', 'estimate'])}, 2016 {pct0(en.loc['2016', 'estimate'])})", src + "editions_imputed.csv"),
        ("C-126", "States with 90.1-2013 or a later 90.1-equivalent edition adopted (kind = adopted) by 2015 / 2016 / 2017 / 2018",
         f"{n13(2015)} / {n13(2016)} / {n13(2017)} / {n13(2018)}", "tables/V_states.csv"),
        ("C-284", "S14, interval metering for every building of at least 25,000 ft²: mean linked level; status; below status; no linked data",
         f"{f3(sv('S14', 'mean linked level'))}; {pct(sv('S14', 'status tier (linked)'))}; {pct(sv('S14', 'below the status tier (linked)'))}; "
         f"{pct(sv('S14', 'linked level zero'))}", src + "sensitivities.csv"),
        ("C-285", "Division ranking: floor-area-weighted correlation across the nine divisions of the mean linked level with the "
                  "advanced-meter share / the BAS share", f"{r_ami:+.2f} / {r_bas:+.2f} (point estimates, nine divisions)", src + "geography.csv"),
        ("C-287", "Primary result: mean code-required / declared / linked level [95 %]; FMI of the linked mean",
         f"{ci(e(pr, 'mean code-required level'), f3n)} / {ci(e(pr, 'mean declared level'), f3n)} / {ci(e(pr, 'mean linked level'), f3n)}; "
         f"FMI {e(pr, 'mean linked level').fmi:.2f}", src + "national.csv"),
        ("C-288", "Primary result: floor area at the status / summary / diagnosis tier and below the status tier [95 %]",
         " / ".join(ci(e(pr, q), pct0) for q in TIERS) + f"; below status {ci(e(pr, 'below the status tier (linked)'), pct0)}; FMI "
         + " / ".join(f"{e(pr, q).fmi:.2f}" for q in TIERS), src + "national.csv"),
        ("C-290", "Primary result: floor area whose linked level is below its code-required level; zero",
         f"{ci(e(pr, 'linked below code-required'), pct0)}; {ci(e(pr, 'linked level zero'), pct0)}", src + "national.csv"),
        ("C-291", "Primary result: H2b concentration; H3 diagnosis share at the declared level; H4 vintage − type",
         f"H2b {ci(rdv('h2b'), f3n)}; H3 {ci(e(pr, 'diagnosis tier (declared)'), pct0)}; H4 {rdv('h4').estimate:+.3f} "
         f"[{rdv('h4').lo95:+.3f}, {rdv('h4').hi95:+.3f}]", src + "national.csv, reads_decomposition.csv"),
        ("C-292", "Primary result: share of commercial major-fuel energy use at the status / summary / diagnosis tier",
         " / ".join(ci(e(pr, q, "energy"), pct0) for q in TIERS), src + "national.csv"),
        ("C-181", "Primary result: share of buildings (count) at the status / summary / diagnosis tier",
         " / ".join(ci(e(pr, q, "count"), pct0) for q in TIERS), src + "national.csv"),
        ("C-293", "Primary result: mean linked level by size class (<= 5,000 / 5,001-24,999 / >= 25,000 ft²)",
         " / ".join(ci(cut("SIZE", s, "mean linked level"), f3n) for s in SIZES), src + "cuts.csv"),
        ("C-295", "Primary result: monitoring lever at >= 25,000 ft² (mean / summary / diagnosis gain)",
         f"{f3n(lv('bundle, >= 25,000 ft2 only', 'mean linked gain').estimate)} / {100 * lv('bundle, >= 25,000 ft2 only', 'summary gain').estimate:+.1f} pp / "
         f"{100 * lv('bundle, >= 25,000 ft2 only', 'diagnosis gain').estimate:+.1f} pp", src + "h5_lever.csv"),
        ("C-186", "Data gap, floor area: declared minus code-required; drops by thermostat / meter / BAS and network route / BAS linkage",
         f"{gap('declared minus code-required').estimate:+.3f}; " + " / ".join(f"{gap(k).estimate:+.3f} [{gap(k).lo95:+.3f}, {gap(k).hi95:+.3f}]" for k in drops),
         src + "gap.csv"),
        ("C-297", "Decomposition of the linked level, primary (thermostat / metering / BAS HVAC points / occupancy, lighting, CO2)",
         "linked " + " / ".join(f3n(dec.loc[f'linked {g}', 'estimate']) for g in GROUPS) + "; code-required "
         + " / ".join(f3n(dec.loc[f'code-required {g}', 'estimate']) for g in GROUPS) + "; declared "
         + " / ".join(f3n(dec.loc[f'declared {g}', 'estimate']) for g in GROUPS), src + "decomposition_scenarios.csv"),
        ("C-298", "Linked level by construction period [95 %]",
         "; ".join(f"{p_} {ci(cut('YRCONC', p_, 'mean linked level'), f3n)}" for p_ in periods), src + "cuts.csv"),
        ("C-190", "Linked level by building type [95 %]",
         "; ".join(f"{t_} {ci(cut('PBA', t_, 'mean linked level'), f3n)}" for t_ in types), src + "cuts.csv"),
        ("C-299", "Geography: linked level / status share / advanced-meter share / BAS share by census division",
         "; ".join(f"{d_} {gev(d_, 'mean linked level'):.3f} / {pct0(gev(d_, 'status tier (linked)'))} / "
                   f"{pct0(gev(d_, 'advanced-meter share (expected)'))} / {pct0(gev(d_, 'BAS share'))}" for d_ in divs), src + "geography.csv"),
        ("C-300", "Floor area with no linked data: BAS / smart thermostat / AMI-recorded meter share, against the whole stock",
         " / ".join(f"{pct0(zv('zero linked level', k))} (stock {pct0(zv('whole stock', k))})" for k in ("BAS", "smart thermostat", "AMI-recorded meter")),
         src + "zero_profile.csv"),
        ("C-193", "Linked level and floor share with no linked data under the three weightings (floor area / energy / buildings)",
         " / ".join(f"{e(pr, 'mean linked level', w_).estimate:.3f}" for w_ in ("area", "energy", "count")) + "; zero "
         + " / ".join(pct0(e(pr, "linked level zero", w_).estimate) for w_ in ("area", "energy", "count")), src + "national.csv"),
        ("C-301", "Enriched scenario: status / summary / diagnosis tier [95 %]",
         " / ".join(ci(e("enriched (high)", f"{t_} tier (linked)"), pct0) for t_ in ("status", "summary", "diagnosis")), src + "national.csv"),
        ("C-302", "Lever at >= 25,000 ft² without the equipment-power reading (mean / summary / diagnosis gain)",
         f"{f3n(lv('bundle without equipment power, >= 25,000 ft2 only', 'mean linked gain').estimate)} / "
         f"{100 * lv('bundle without equipment power, >= 25,000 ft2 only', 'summary gain').estimate:+.1f} pp / "
         f"{100 * lv('bundle without equipment power, >= 25,000 ft2 only', 'diagnosis gain').estimate:+.1f} pp", src + "h5_lever.csv"),
        ("C-197", "H2b and size ranking under the census-to-stock shift δ (types and classes together): first δ where the interval "
                  "includes the opposite (down / up)",
         "; ".join(f"{c_}: {r['interval includes opposite (down)']} / {r['interval includes opposite (up)']}" for c_, r in
                   ((r_.conclusion.split(' (')[0], r_) for _, r_ in TT[(TT.variant == 'ab') & TT.conclusion.str.contains('H2b|size', regex=True)].iterrows())),
         src + "transfer_tipping.csv"),
        ("C-304", "Imputations per scenario: M (von Hippel 2020 quadratic rule, M = 1 + 200 FMI^2 at CV 0.05; largest headline FMI)",
         f"M = {int(e(pr, 'mean linked level').M)}; largest FMI {fmi:.2f} (rule: M >= {m_rule}), the largest among the primary "
         "estimates of the headline table", src + "national.csv"),
        ("C-305", "Code-required level by construction period [95 %]",
         "; ".join(f"{p_} {ci(cut('YRCONC', p_, 'mean code-required level'), f3n)}" for p_ in periods), src + "cuts.csv"),
        ("C-306", "Floor area with no linked data, by size class (≤ 5,000 / 5,001–24,999 / ≥ 25,000 ft²) [95 %]",
         " / ".join(ci(cwz("SIZE", s_), pct0) for s_ in SIZES), src + "cuts_weights.csv"),
        ("C-202", "Floor area with no linked data, by building type [95 %]; nonrefrigerated warehouses' share of the stock's floor area",
         "; ".join(f"{t_} {ci(cwz('PBA', t_), pct0)}" for t_ in ("Nonrefrigerated warehouse", "Office", "Education", "Service", "Vacant"))
         + f"; warehouses hold {pct0(wh_share)} of the floor area", src + "cuts_weights.csv; CBECS 2018 weights"),
        ("C-307", "S12, advanced meters at 2025 shares: linked level, tier shares, below the status tier, zero share",
         f"linked {ci(sv('S12', 'mean linked level'), f3n)}; status / summary / diagnosis "
         + " / ".join(ci(sv('S12', f'{t_} tier (linked)'), pct0) for t_ in ("status", "summary", "diagnosis"))
         + f"; below status {ci(sv('S12', 'below the status tier (linked)'), pct0)}; zero {ci(sv('S12', 'linked level zero'), pct0)}; "
         + ("most floor area still reaches no readiness tier" if sv('S12', 'below the status tier (linked)').lo95 > 0.5
            else "the claim that most floor area reaches no tier no longer holds with its interval above 50 %"), src + "sensitivities.csv"),
        ("C-308", "Sensitivities S8 (monthly bills credited) and S9 (smart thermostats not credited): mean linked level and status share",
         f"S8 {ci(sv('S8', 'mean linked level'), f3n)}, status {ci(sv('S8', 'status tier (linked)'), pct0)}; "
         f"S9 {ci(sv('S9', 'mean linked level'), f3n)}, status {ci(sv('S9', 'status tier (linked)'), pct0)}", src + "sensitivities.csv"),
        ("C-315", "Status-tier share of floor area at tier cutoffs 0.40 and 0.60 (primary; the cutoff is 0.50)",
         f"0.40: {pct(e(pr, 'status tier at cutoff 0.40'))}; 0.60: {pct(e(pr, 'status tier at cutoff 0.60'))}", src + "national.csv"),
        ("C-213", "Contrast dropping the records with an EIA-imputed control answer (S11): mean linked level [95 %]",
         f"{sv('S11', 'mean linked level').estimate:.3f} [{sv('S11', 'mean linked level').lo95:.3f}–{sv('S11', 'mean linked level').hi95:.3f}] "
         "(weights not recalibrated)", src + "sensitivities.csv"),
        ("C-317", "Single stream added to every building: summary-tier gain for occupancy (Occ) and equipment power (Pwr) [95 %]",
         f"Occ {sp(h5('Occ', 'summary gain'))}; Pwr {sp(h5('Pwr', 'summary gain'))}", src + "h5_lever.csv"),
        ("C-338", "Single stream added to every building: summary / diagnosis-tier gain for zone CO2 (CO2) and outdoor-air "
                  "temperature (OAT) [95 %]",
         "; ".join(f"{k} {sp(h5(k, 'summary gain'))} / {sp(h5(k, 'diagnosis gain'))}" for k in ("CO2", "OAT")), src + "h5_lever.csv"),
        ("C-345", "H3 (construction period explains no more than building type) under the reference-set shift δ (types and classes "
                  "together): δ at which the point estimate reverses, downward / upward; δ at which the interval first includes "
                  "the opposite, downward / upward",
         f"point {T3['point flips (down)']} / {T3['point flips (up)']}; interval {T3['interval includes opposite (down)']} / "
         f"{T3['interval includes opposite (up)']} (0.0: already at the primary result)", src + "transfer_tipping.csv (variant ab)"),
    ]


# ── the question set and the HVAC-credit scoring (results/scoring) ───────────────────────────────
def scoring_rows() -> list[tuple]:
    SC, CE, H5, LS = csv(SCO / "scoring.csv"), csv(SCO / "cells.csv"), csv(SCO / "h5_scoring.csv"), csv(SCO / "linkage_size.csv").set_index("quantity")
    RC = csv(SCO / "reference_set_counts.csv").set_index("stream")
    sco = lambda sc, s, st, w="area": SC[(SC.scoring == sc) & (SC.scenario == s) & (SC.weight == w) & (SC.statistic == st)].iloc[0]
    heads = ("mean linked level", *TIERS, "below the status tier (linked)")
    line = lambda sc, s: "; ".join((f3 if h.startswith("mean") else pct)(sco(sc, s, h)) for h in heads)
    src = "results/scoring/"
    rows = [
        ("C-158", "Reference-set counts: SAT declared by 40 of all 45 graphs and by 35 of the 37 with C_declared > 0; Run + Load jointly "
                  "(45-graph basis) against Load alone",
         f"Run+Load {int(RC.loc['Run+Load', 'declared, all 45'])} of 45, Load {int(RC.loc['Load', 'declared, all 45'])} of 45; "
         f"on 37: {int(RC.loc['Run+Load', 'declared, 37 with C_declared > 0'])} and {int(RC.loc['Load', 'declared, 37 with C_declared > 0'])}",
         src + "reference_set_counts.csv"),
        ("C-277", "Scenario range, question set: strict / primary / enriched / optimistic (mean linked level; status; summary; diagnosis; below status)",
         " | ".join(f"{s}: {line('primary', s)}" for s in ("strict (low)", "primary", "enriched (high)", "optimistic")), src + "scoring.csv"),
        ("C-278", "HVAC-credit scoring: strict / primary / enriched / optimistic (mean linked level; status; summary; diagnosis; below status)",
         " | ".join(f"{s}: {line('hvac', s)}" for s in ("strict (low)", "primary", "enriched (high)", "optimistic")), src + "scoring.csv"),
        ("C-279", "Reads under the HVAC-credit scoring (primary): H1 below status; H2b; H3 diagnosis, declared; H4; H5 bundle summary / diagnosis gain",
         f"H1 {pct(sco('hvac', 'primary', 'below the status tier (linked)'))}; H2b {f3(sco('hvac', 'primary', 'H2b concentration ratio'))}; "
         f"H3 {pct(sco('hvac', 'primary', 'diagnosis tier (declared)'))}; H4 {sg(sco('hvac', 'primary', 'H4 vintage - type'))}; H5 "
         + " / ".join(sp(H5[(H5.scoring == 'hvac') & (H5.item == '§8.4.3 bundle') & (H5.statistic == f'{t} gain')].iloc[0]) for t in ("summary", "diagnosis")),
         src + "scoring.csv, h5_scoring.csv"),
        ("C-280", "Tier shares (status / summary / diagnosis) at the code-required and declared levels, primary result: question set; HVAC credit",
         "; ".join(f"{lvl}: " + " / ".join(pct(sco(sc, 'primary', f'{t} tier ({lvl})')) for t in ("status", "summary", "diagnosis"))
                   + f" ({'question set' if sc == 'primary' else 'HVAC credit'})" for lvl in ("code-required", "declared") for sc in ("primary", "hvac")),
         src + "scoring.csv"),
    ]
    domain_id = {"thermal": "C-163", "energy": "C-281", "airquality": "C-165", "equipment": "C-282", "occupancy": "C-167",
                 "lighting": "C-283"}
    for d in canon.DOMAINS:
        cells = []
        for i in ("status", "summary", "diagnosis"):
            a = CE[(CE.scoring == "primary") & (CE.statistic == f"{d} x {i} (linked)")]
            b = CE[(CE.scoring == "hvac") & (CE.statistic == f"{d} x {i} (linked)")]
            if not a.empty:
                cells.append(f"{i} {100 * a.iloc[0].estimate:.1f}% / {100 * b.iloc[0].estimate:.1f}%")
        rows.append((domain_id[d], f"Domain-level answerability, {d}: floor-area share reaching at least level 1, linked level, primary "
                                     "result (question set / HVAC credit)", "; ".join(cells), src + "cells.csv"))
    q = lambda k, c="value": LS.loc[k, c]
    rows.append(("C-171", "Linkage type by floor area (reference set): slopes per tenfold floor area, none vs all / partial vs all; likelihood-ratio p",
                 f"{q('slope, none vs all (per tenfold floor area)'):+.2f} [{q('slope, none vs all (per tenfold floor area)', 'lo95'):+.2f}, "
                 f"{q('slope, none vs all (per tenfold floor area)', 'hi95'):+.2f}] / {q('slope, partial vs all (per tenfold floor area)'):+.2f} "
                 f"[{q('slope, partial vs all (per tenfold floor area)', 'lo95'):+.2f}, {q('slope, partial vs all (per tenfold floor area)', 'hi95'):+.2f}]; "
                 f"p = {q('p-value'):.2f} ({int(q('graphs in the reference set with floor area'))} graphs); size not used",
                 src + "linkage_size.csv; tables/R_linkage_size.csv"))
    return rows


# ── AI readiness by imputed code edition (results/editions) ──────────────────────────────────────
def edition_rows() -> list[tuple]:
    from atlas import editions
    E = csv(EDI / "edition_levels.csv")
    get = lambda b, sc, g, st: E[(E.basis == b) & (E.scoring == sc) & (E.group == g) & (E.statistic == st)].iloc[0]
    short = {"before any 90.1-equivalent code": "none", "90.1-1989": "1989", "90.1-1999/2001": "1999/2001",
             "90.1-2004 and 2007": "2004/2007", "90.1-2010": "2010", "90.1-2013 or later": "2013+"}
    seq = lambda b, sc, st, fmt: "; ".join(f"{short[g]} {fmt(get(b, sc, g, st))}" for g in editions.GROUPS)
    order = "groups: before any 90.1-equivalent code (none) / 90.1-1989 / 1999/2001 / 2004 and 2007 / 2010 / 2013 or later"
    rec = "; ".join(f"{short[g]} {get('construction', 'primary', g, 'mean linked level')['records (mean over imputations)']:.0f}"
                    for g in editions.GROUPS)
    M = int(E.M.max())
    src = "results/editions/edition_levels.csv"
    return [
        ("C-216", f"AI readiness by imputed construction edition, share of floor area per group [95 %] (primary result, M = {M}, "
                  f"Rubin's rules; {order}); survey records per group (mean over imputations)",
         seq("construction", "primary", "share of floor area", pct) + f". Records: {rec} (none below 30)", src),
        ("C-319", "AI readiness by imputed construction edition, mean code-required / declared / linked level per group [95 %]",
         "code-required: " + seq("construction", "primary", "mean code-required level", f3)
         + ". Declared: " + seq("construction", "primary", "mean declared level", f3)
         + ". Linked: " + seq("construction", "primary", "mean linked level", f3), src),
        ("C-320", "AI readiness by imputed construction edition, status-tier share (linked) and share with no linked data per group [95 %]",
         "status: " + seq("construction", "primary", "status tier (linked)", pct)
         + ". No linked data: " + seq("construction", "primary", "linked level zero", pct), src),
        ("C-321", "Variant, groups by the latest edition among a building's systems (construction, or the renovation edition of a "
                  "system renovated since 2000): share of floor area, mean linked level and status-tier share per group [95 %]",
         "share: " + seq("latest", "primary", "share of floor area", pct)
         + ". Linked: " + seq("latest", "primary", "mean linked level", f3)
         + ". Status: " + seq("latest", "primary", "status tier (linked)", pct), src),
    ]


# ── scenario dimensions and code timing (results/scenarios) ──────────────────────────────────────
def scenario_rows() -> list[tuple]:
    fmt = lambda v: "none in (0, 1]" if str(v) == "none in range" else ("1 (already at the primary)" if float(v) == 1.0 else f"{float(v):.2f}")
    tip = csv(SCN / "scenario_tipping_stream.csv")
    tv = lambda conc, axis, col: tip[tip.conclusion.str.contains(conc, regex=False) & (tip.axis == axis)].iloc[0][col]
    sob = {t: csv(SCN / f"sobol_{t}_stream.csv") for t in "ab"}
    so = lambda t, i, o: sob[t][(sob[t].input == i) & (sob[t].output == o)].iloc[0]
    J = csv(SCN / "timing_steps_s10ref.csv")
    jg = lambda part, var, pop, st: J[(J.part == part) & (J.variant == var) & (J.population == pop) & (J.statistic == st)].iloc[0]
    S = csv(EST / "sensitivities.csv")
    sv = lambda sid, st: S[(S.id == sid) & (S.statistic == st)].iloc[0]
    G = csv(SCN / "scenario_grid_stream.csv")
    below = G[(G.statistic == "below the status tier (linked)") & np.isclose(G.c, 1) & np.isclose(G.d, 0.05)].iloc[0]
    src = "results/scenarios/"
    tips = "; ".join(f"{lab} c: {fmt(tv(key, 'c', 'interval includes opposite (down)'))}, d: "
                     f"{fmt(tv(key, 'd', 'interval includes opposite (down)'))}"
                     for lab, key in (("H1", "(H1)"), ("H2b", "(H2b)"), ("small < mid", "small < mid"), ("mid < large", "mid-size < large"),
                                      ("H3", "(H4)")))
    rows = [
        ("C-226", "Scenario dimensions over (0, 1], description applied to non-BAS routes only (the design): 'most floor area reaches no "
                  "tier' (H1) under compliance c and description d",
         f"c: point flips at {fmt(tv('(H1)', 'c', 'point flips (down)'))}, interval includes the opposite at "
         f"{fmt(tv('(H1)', 'c', 'interval includes opposite (down)'))}; d: point flips at "
         f"{fmt(tv('(H1)', 'd', 'point flips (down)'))}, interval includes the opposite at "
         f"{fmt(tv('(H1)', 'd', 'interval includes opposite (down)'))}; below the status tier at d = 0.05: {pct(below)}",
         src + "scenario_tipping_stream.csv"),
        ("C-323", "H2b and size ranking under c and d over (0, 1], description on non-BAS routes only: value at which the interval first "
                  "includes the opposite", tips, src + "scenario_tipping_stream.csv"),
        ("C-228", "Sobol total-effect ranking for the mean linked level, description on non-BAS routes only; ranges (a) c, d in [0.5, 1] and "
                  "(b) d in [0.05, 1]; δ in [-3, 3]",
         "(a) " + ", ".join(f"{i} {so('a', i, 'mean linked level').ST:.2f}" for i in ("d", "delta", "c")) + "; (b) "
         + ", ".join(f"{i} {so('b', i, 'mean linked level').ST:.2f}" for i in ("d", "delta", "c")), src + "sobol_a_stream.csv, sobol_b_stream.csv"),
        ("C-324", "Sobol total effect of description d on the status / summary / diagnosis tier shares [95 %], description on non-BAS routes "
                  "only; ranges (a) d in [0.5, 1] and (b) d in [0.05, 1]",
         "; ".join(f"({t}) " + " / ".join(f"{so(t, 'd', o).ST:.2f} [{so(t, 'd', o)['ST lo']:.2f}–{so(t, 'd', o)['ST hi']:.2f}]" for o in TIERS)
                   for t in "ab"), src + "sobol_a_stream.csv, sobol_b_stream.csv"),
    ]
    for cid, var in (("C-326", "lag 1 year"), ("C-327", "lag 2 years"), ("C-328", "monitoring credit only under 90.1")):
        rows.append((cid, f"Code timing sensitivity, {var}: floor area under 90.1-2013 or later; that cohort's mean linked level and status "
                          "share; H3 (construction period minus building type) [95 %]; primary for comparison",
                     f"{pct(jg('timing', var, 'stock', 'share under 90.1-2013 or later'))}; cohort {f3(jg('timing', var, 'cohort 90.1-2013 or later', 'mean linked level'))}, "
                     f"status {pct(jg('timing', var, 'cohort 90.1-2013 or later', 'status tier (linked)'))}; H3 {sg(jg('timing', var, 'H3', 'construction period minus building type'))} "
                     f"(primary: {pct(jg('timing', 'primary', 'stock', 'share under 90.1-2013 or later'))}; cohort "
                     f"{f3(jg('timing', 'primary', 'cohort 90.1-2013 or later', 'mean linked level'))}, status "
                     f"{pct(jg('timing', 'primary', 'cohort 90.1-2013 or later', 'status tier (linked)'))}; H3 "
                     f"{sg(jg('timing', 'primary', 'H3', 'construction period minus building type'))})", src + "timing_steps_s10ref.csv"))
    for cid, sid in (("C-329", "S15"), ("C-330", "S16"), ("C-331", "S17"), ("C-333", "S18")):
        rows.append((cid, f"Sensitivity {sid}: mean linked level; status tier; below the status tier (H1); no linked data; H2b concentration "
                          "ratio; H3 (construction period minus building type) [95 %]",
                     f"{f3(sv(sid, 'mean linked level'))}; {pct(sv(sid, 'status tier (linked)'))}; {pct(sv(sid, 'below the status tier (linked)'))}; "
                     f"{pct(sv(sid, 'linked level zero'))}; {sv(sid, 'H2b concentration ratio').estimate:.2f} "
                     f"[{sv(sid, 'H2b concentration ratio').lo95:.2f}–{sv(sid, 'H2b concentration ratio').hi95:.2f}]; {sg(sv(sid, 'H4 vintage - type'))}",
                     "results/estimates/sensitivities.csv"))
    st = lambda s: jg("steps", "primary", "stock", s)
    rows.append(("C-336", "Floor area at linked level exactly 0 / 8/76 / 12/76 / 20/76, and at any of the four [95 %] (primary)",
                 " / ".join(pct(st(f"linked level exactly {k}")) for k in ("0", "8/76", "12/76", "20/76"))
                 + f"; together {pct(st('linked level at 0, 8/76, 12/76 or 20/76'))}", src + "timing_steps_s10ref.csv"))
    L = csv(SCN / "levels_required.csv")
    for cid, r in zip(("C-243", "C-244", "C-245"), L.itertuples()):
        rows.append((cid, f"Linked level {r.level}: what it requires", f"{r.means} ({r.streams}); answers {r._5}",
                     "atlas/canon.py; " + src + "levels_required.csv"))
    h2 = sv("S23", "H2b concentration ratio")
    rows.append(("C-351", "Sensitivity S23 (description d = 0.5 applied to every linked stream, BAS routes included): mean linked "
                          "level; status / summary / diagnosis tier; below the status tier (H1); no linked data; H2b concentration "
                          "ratio; H3 (construction period minus building type) [95 %]",
                 f"{f3(sv('S23', 'mean linked level'))}; {pct(sv('S23', 'status tier (linked)'))} / {pct(sv('S23', 'summary tier (linked)'))} / "
                 f"{pct(sv('S23', 'diagnosis tier (linked)'))}; {pct(sv('S23', 'below the status tier (linked)'))}; "
                 f"{pct(sv('S23', 'linked level zero'))}; {h2.estimate:.2f} [{h2.lo95:.2f}–{h2.hi95:.2f}]; {sg(sv('S23', 'H4 vintage - type'))}",
                 "results/estimates/sensitivities.csv"))
    GB = csv(SCN / "scenario_grid_bas_description.csv")
    gv = lambda d, st_: GB[np.isclose(GB.d, d) & (GB.statistic == st_)].iloc[0]
    TB = csv(SCN / "scenario_tipping_bas_description.csv")
    tb = lambda conc, col: TB[TB.conclusion.str.contains(conc, regex=False)].iloc[0][col]
    fb = lambda v: "none in (0, 1]" if str(v) == "none in range" else f"{float(v):.2f}"
    h1 = gv(0.05, "below the status tier (linked)")
    nat_lo = h1.estimate - Z * np.sqrt(h1["T"])
    nxt = GB[(GB.statistic == "below the status tier (linked)") & (GB.d > 0.05 + 1e-9) & (GB.lo95 <= 0.5)]
    rows.append(("C-271", "Description d applied to every linked stream (BAS routes included), c = 1, searched over (0, 1] down to "
                          "0.05: 'most floor area reaches no tier' (H1) — d at which the point estimate flips, d at which its interval "
                          "first includes the opposite; the same for H2b, small < mid-size, mid-size < large",
                 f"H1: point flips at {fb(tb('(H1)', 'point flips (down)'))}, interval includes the opposite at "
                 + ("none in (0, 1]" if nxt.empty else f"{nxt.d.max():.2f}")
                 + f" (at d = 0.05 the share below the status tier is {h1.estimate:.6f}, where the logit interval is "
                 f"uninformative; natural-scale 95 % interval [{nat_lo:.5f}, 1.00000]); H2b: point below 1 only at d = 0.05 "
                 f"(status-tier share {gv(0.05, 'status tier (linked)').estimate:.6f}), interval includes the opposite from "
                 f"{fb(tb('(H2b)', 'interval includes opposite (down)'))}; small < mid-size: interval from "
                 f"{fb(tb('small < mid', 'interval includes opposite (down)'))}; mid-size < large: "
                 f"{fb(tb('mid-size < large', 'interval includes opposite (down)'))}",
                 src + "scenario_tipping_bas_description.csv, scenario_grid_bas_description.csv"))
    return rows


# ── the question set's weighting (results/weighting) ─────────────────────────────────────────────
def weighting_rows() -> list[tuple]:
    S = csv(EST / "sensitivities.csv")
    sv = lambda sid, st: S[(S.id == sid) & (S.statistic == st)].iloc[0]
    rows = []
    for cid, sid, name in (("C-339", "S19", "each domain weighted equally"), ("C-340", "S20", "each intent weighted equally")):
        h2 = sv(sid, "H2b concentration ratio")
        rows.append((cid, f"Sensitivity {sid} ({name}): mean code-required / declared / linked level; status / summary / diagnosis "
                          "tier; below the status tier (H1); no linked data; H2b concentration ratio; H3 (construction period minus "
                          "building type) [95 %]",
                     f"{f3(sv(sid, 'mean code-required level'))} / {f3(sv(sid, 'mean declared level'))} / {f3(sv(sid, 'mean linked level'))}; "
                     f"{pct(sv(sid, 'status tier (linked)'))} / {pct(sv(sid, 'summary tier (linked)'))} / {pct(sv(sid, 'diagnosis tier (linked)'))}; "
                     f"{pct(sv(sid, 'below the status tier (linked)'))}; {pct(sv(sid, 'linked level zero'))}; "
                     f"{h2.estimate:.2f} [{h2.lo95:.2f}–{h2.hi95:.2f}]; {sg(sv(sid, 'H4 vintage - type'))}",
                     "results/estimates/sensitivities.csv"))
    SCN_ = (("primary", "primary"), ("domain_equal", "S19"), ("intent_equal", "S20"))
    Zr = csv(WGT / "size_ranking.csv")
    zg = lambda sc, st: Zr[(Zr.scoring == sc) & (Zr.statistic == st)].iloc[0]
    rows.append(("C-341", "Mean linked level by size class (≤ 5,000 / 5,001–24,999 / ≥ 25,000 ft²) and the contrasts small − mid-size and "
                          "mid-size − large [95 %], under the primary scoring, S19 and S20",
                 "; ".join(f"{lab}: {zg(sc, 'mean linked, <= 5,000').estimate:.3f} / {zg(sc, 'mean linked, 5,001-24,999').estimate:.3f} / "
                           f"{zg(sc, 'mean linked, >= 25,000').estimate:.3f}, small − mid {sg(zg(sc, 'size: small - mid'))}, mid − large "
                           f"{sg(zg(sc, 'size: mid - large'))}" for sc, lab in SCN_), "results/weighting/size_ranking.csv"))
    Ls = csv(WGT / "loss_shares.csv")
    lg = lambda sc, st: Ls[(Ls.scoring == sc) & (Ls.statistic == st)].iloc[0]
    rules = ("thermostat rule", "meter rule", "BAS and network route rule", "BAS linkage draw")
    rows.append(("C-342", "Declared-to-linked loss by recording rule (thermostat / meter / BAS and network route / BAS linkage draw), each "
                          "as a share of the total loss [95 %], under the primary scoring, S19 and S20",
                 "; ".join(f"{lab}: " + " / ".join(f"{100 * lg(sc, f'share of loss: {r}').estimate:.1f}%" for r in rules)
                           + f" (total loss {lg(sc, 'loss: total (declared minus linked)').estimate:.3f})" for sc, lab in SCN_)
                 + "; intervals in the source", "results/weighting/loss_shares.csv"))
    return rows


# ── the reference-set graphs that link none (S21, S22) ───────────────────────────────────────────
def link_none_rows() -> list[tuple]:
    A = csv(link_none.AUDIT)
    no, yes = A[A["any time-series reference"] == "no"], A[A["any time-series reference"] == "yes"]
    rows = [("C-260", "Reference-set knowledge graphs that link none of their declared streams: the number that publish no "
                      "time-series reference at all (of 9); the others and their points with a time-series reference",
             f"{len(no)} of {len(A)} ({', '.join(no.building)}); "
             + ", ".join(f"{r.building} {r._5} of {r.points} points" for r in yes.itertuples())
             + ", none of them on a stream that answers a question",
             "tables/R_linkage_none_audit.csv (Mortar graph files; atlas/link_none.py)")]
    S = csv(EST / "sensitivities.csv")
    sv = lambda sid, st: S[(S.id == sid) & (S.statistic == st)].iloc[0]
    for cid, sid, name in (("C-346", "S21", "the 9 graphs that link none treated as missing"),
                           ("C-347", "S22", "only the graphs that link none and publish no time-series reference treated as missing")):
        h2 = sv(sid, "H2b concentration ratio")
        rows.append((cid, f"Sensitivity {sid} ({name}): mean code-required / declared / linked level; status / summary / diagnosis "
                          "tier; below the status tier (H1); no linked data; H2b concentration ratio; H3 (construction period minus "
                          "building type) [95 %]",
                     f"{f3(sv(sid, 'mean code-required level'))} / {f3(sv(sid, 'mean declared level'))} / {f3(sv(sid, 'mean linked level'))}; "
                     f"{pct(sv(sid, 'status tier (linked)'))} / {pct(sv(sid, 'summary tier (linked)'))} / {pct(sv(sid, 'diagnosis tier (linked)'))}; "
                     f"{pct(sv(sid, 'below the status tier (linked)'))}; {pct(sv(sid, 'linked level zero'))}; "
                     f"{h2.estimate:.2f} [{h2.lo95:.2f}–{h2.hi95:.2f}]; {sg(sv(sid, 'H4 vintage - type'))}",
                     "results/estimates/sensitivities.csv"))
    return rows


# ── the assumption tables ────────────────────────────────────────────────────────────────────────
def table_rows() -> list[tuple]:
    V = csv(TAB / "V_states.csv")
    proj = V.source.str.contains("workbook|2026", case=False, na=False)   # the rows added to carry the edition record to 2025
    return [
        ("C-256", "Edition record: state-edition rows for the 2018 stock (all but the rows added for the 2025 projection); of them, "
                  "rows assumed for states with no formal statewide code (PNNL-19887 Table 7.1); rows added for the projection",
         f"{int((~proj).sum())}; {int(((V.kind == 'implicit') & ~proj).sum())} assumed; {int(proj.sum())} added", "tables/V_states.csv"),
        ("C-354", "Code-clause table: rows used by this study",
         f"{int((csv(TAB / 'C_clauses.csv').reading != 'not_used').sum())} rows", "tables/C_clauses.csv"),
    ]


# ── the code clauses: §8.4.3 exception f, the rows by use, §10.4.5, S24 ──────────────────────────────
def clause_rows() -> list[tuple]:
    from atlas import floors
    st = draws.stock()
    a, d = st.a, st.df
    share = lambda m: float(a[m].sum() / a.sum())
    ex = d.PBAPLUS.isin(floors.EXEMPT_843).to_numpy()
    big = (d.SQFT >= 25000).to_numpy()
    C = csv(TAB / "C_clauses.csv")
    ids = lambda tags: ", ".join(C[C.reading.isin(tags)].id)
    dist = ((d.CHWT == 1) | (d.STHW == 1)).to_numpy()
    new = (d.YRCONC == 9).to_numpy()
    c15 = floors._condition(d, "sqft>=25000&district").to_numpy()
    c08 = floors._condition(d, "electric&sqft>=25000&not-hotel-motel-restaurant").to_numpy()
    only = c15 & ~c08
    S = csv(EST / "sensitivities.csv")
    sv = lambda st_: S[(S.id == "S24") & (S.statistic == st_)].iloc[0]
    h2 = sv("H2b concentration ratio")
    return [
        ("C-272", "§8.4.3 exception f (hotels, motels and restaurants): survey activity codes exempted; their share of the stock's "
                  "floor area, all sizes and at least 25,000 ft² (records); the floor area where the clause applies by size and type "
                  "(the lever's scope), against all floor area at least 25,000 ft²",
         f"PBAPLUS {', '.join(str(c) for c in floors.EXEMPT_843)} (fast food, restaurant/cafeteria, hotel/resort, motel/inn/bed and "
         f"breakfast); {pct0(share(ex))} of floor area, {pct0(share(ex & big))} at least 25,000 ft² ({int((ex & big).sum())} records); "
         f"{pct0(share(mi.clause_scope()))} against {pct0(share(big))}",
         "tables/C_clauses.csv C08; atlas/floors.py EXEMPT_843; CBECS 2018 (PBAPLUS, SQFT, weights)"),
        ("C-353", "C15 (§10.4.5): records meeting its condition (at least 25,000 ft², district chilled water, steam or hot water) / "
                  "of them also meeting C08's / C15's alone (hotels, motels and restaurants among them) / C15's alone built "
                  "2013–2018 (the only 2018 buildings where it could add Sub that C08 does not already give at full compliance); "
                  "share of floor area",
         f"{int(c15.sum())} ({pct0(share(c15))}) / {int((c15 & c08).sum())} / {int(only.sum())} ({pct0(share(only))}; "
         f"{int((only & ex).sum())} hotels, motels or restaurants) / {int((only & new).sum())}",
         "tables/C_clauses.csv C08, C15; atlas/floors.py CONDITIONS; CBECS 2018 (SQFT, CHWT, STHW, ELUSED, PBAPLUS, YRCONC)"),
        ("C-355", "Code-clause table, rows by use: primary result / sensitivities / condition not in the survey / added and not used",
         f"{ids(['core', 'occ_late', 'meter_clause'])} / {ids(['occ_early', 'occ_alt'])} / {ids(['sensitivity'])} / "
         f"{ids(['not_used'])} ({len(C)} rows shown in the appendix table)", "tables/C_clauses.csv"),
        ("C-356", "90.1-2013 §10.4.5 whole-building monitoring of non-electric energy (C15): streams of the question set it would "
                  "record; floor area at least 25,000 ft², built 2013–2018, with district chilled water, steam or hot water (an upper "
                  "bound on where it could apply)",
         "natural gas, fuel oil and propane meters: none (Whole is electricity only, Pwr is equipment input power); district chilled "
         "water, steam or hot water: Sub, the thermal-energy end-use meter the enriched reading credits (row E13), so C15 is "
         f"credited for Sub; {pct0(share(dist & big & new))} of floor area ({int((dist & big & new).sum())} records)",
         "tables/C_clauses.csv C15; atlas/canon.py STREAMS; tables/E_controls.csv E13"),
        ("C-275", "Sensitivity S24 (occupancy sensing credited for lighting altered under 90.1-2016, C06): mean code-required / "
                  "declared / linked level; status / summary / diagnosis tier; below the status tier (H1); no linked data; H2b "
                  "concentration ratio; H3 (construction period minus building type) [95 %]",
         f"{f3(sv('mean code-required level'))} / {f3(sv('mean declared level'))} / {f3(sv('mean linked level'))}; "
         f"{pct(sv('status tier (linked)'))} / {pct(sv('summary tier (linked)'))} / {pct(sv('diagnosis tier (linked)'))}; "
         f"{pct(sv('below the status tier (linked)'))}; {pct(sv('linked level zero'))}; "
         f"{h2.estimate:.2f} [{h2.lo95:.2f}–{h2.hi95:.2f}]; {sg(sv('H4 vintage - type'))}",
         "results/estimates/sensitivities.csv"),
    ]


def build() -> list[tuple]:
    rows = [*method_rows(), *survey_rows(), *estimate_rows(), *scoring_rows(), *edition_rows(),
            *scenario_rows(), *weighting_rows(), *link_none_rows(), *table_rows(), *clause_rows()]
    return sorted(rows, key=lambda r: int(r[0][2:]))


def main() -> None:
    rows = build()
    L = ["# Results ledger", "",
         "Every number the paper cites, by its claim ID, regenerated from `results/` and `tables/` by "
         "`scripts/build_ledger.py`. Intervals are 95 % (CBECS jackknife within each imputation, Rubin's rules across the 200 "
         "imputations) unless stated.", "",
         "| ID | Claim | Number | Source |", "|----|-------|--------|--------|"]
    L += [f"| {i} | {c} | {n} | {s} |" for i, c, n, s in rows]
    (RES / "ledger.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    (RES / "ledger.json").write_text(json.dumps({i: {"claim": c, "number": n, "source": s} for i, c, n, s in rows},
                                                ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"results/ledger.md: {len(rows)} rows")


if __name__ == "__main__":
    main()
