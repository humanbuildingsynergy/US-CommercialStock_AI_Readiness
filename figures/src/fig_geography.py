"""Figure: geography, as two stacked maps. (a) Mean linked level of each census division (the survey identifies the
division, never the state, so the whole division takes one shade), labelled with the division's floor-area-weighted
advanced-meter share, expected over its buildings' imputed states. (b) Commercial advanced-meter share of each state in
2018 (Form EIA-861, observed by state), every state labelled, with division borders. No note: titles and a key carry it.
Alaska and Hawaii are moved and rescaled into insets; Puerto Rico is outside CBECS.
Sources: results/estimates/geography.csv (CENDIV), tables/L_ami_states.csv, tables/V_states.csv, and the Census
Bureau's 2018 cartographic state boundaries (data/sources/census/cb_2018_us_state_20m.zip)."""
from __future__ import annotations

import geopandas as gpd
import matplotlib as mpl
import pandas as pd
from shapely import affinity

from _common import RES, SMALL, figsize, plt, save

ROOT = RES.parent
G = pd.read_csv(RES / "estimates" / "geography.csv")
G = G[G.cut == "CENDIV"].pivot(index="segment", columns="statistic", values=["estimate", "lo95", "hi95"])
DIV = {1: "New England", 2: "Middle Atlantic", 3: "East North Central", 4: "West North Central", 5: "South Atlantic",
       6: "East South Central", 7: "West South Central", 8: "Mountain", 9: "Pacific"}
div_of = pd.read_csv(ROOT / "tables" / "V_states.csv").drop_duplicates("state").set_index("state").cendiv.map(DIV)
ami = pd.read_csv(ROOT / "tables" / "L_ami_states.csv").set_index("state").ami_share

st = gpd.read_file(f"zip://{ROOT / 'data' / 'sources' / 'census' / 'cb_2018_us_state_20m.zip'}")
st = st[st.STUSPS.isin(div_of.index)].to_crs("EPSG:5070")   # 50 states and DC, Albers equal-area


def inset(geom, scale, dx, dy):
    """Rescale a state about its centroid and move it to the lower left (Alaska, Hawaii)."""
    g = affinity.scale(geom, scale, scale, origin="centroid")
    return affinity.translate(g, dx - g.centroid.x, dy - g.centroid.y)


st.loc[st.STUSPS == "AK", "geometry"] = st.loc[st.STUSPS == "AK", "geometry"].apply(lambda g: inset(g, 0.35, -1.75e6, 0.45e6))
st.loc[st.STUSPS == "HI", "geometry"] = st.loc[st.STUSPS == "HI", "geometry"].apply(lambda g: inset(g, 1.0, -0.85e6, 0.35e6))
st["division"] = st.STUSPS.map(div_of)
st["ami"] = st.STUSPS.map(ami)
dv = st.dissolve(by="division").reset_index()
dv["linked"] = dv.division.map(G[("estimate", "mean linked level")])
dv["ami_div"] = dv.division.map(G[("estimate", "advanced-meter share (expected)")])

CLEAR = dict(facecolor="none", edgecolor="none")   # map labels sit on the fill, with no white backing
# each division's label sits on one of its states, chosen so that neighboring labels do not overlap
ANCHOR = {"New England": "ME", "Middle Atlantic": "PA", "East North Central": "WI", "West North Central": "NE",
          "South Atlantic": "SC", "East South Central": "KY", "West South Central": "TX", "Mountain": "UT", "Pacific": "CA"}
# small eastern states: labels set out in the Atlantic with a leader line, from north to south
CALLOUT = ["VT", "NH", "MA", "RI", "CT", "NJ", "DE", "MD", "DC"]
x_east = st[~st.STUSPS.isin(["AK", "HI"])].total_bounds[2] + 0.12e6
y_top = st.loc[st.STUSPS == "VT"].geometry.representative_point().y.iloc[0] + 0.05e6

fig, (ax, bx) = plt.subplots(2, 1, figsize=figsize("full", 1.22))
norm_a = mpl.colors.Normalize(0.08, 0.18)
dv.plot(column="linked", ax=ax, cmap="Blues", norm=norm_a, edgecolor="white", linewidth=0.8)
for d, a, b in zip(dv.division, dv.linked, dv.ami_div):
    p = st.loc[st.STUSPS == ANCHOR[d]].geometry.representative_point().iloc[0]
    ax.text(p.x, p.y, f"{a:.3f}\n{100 * b:.0f}%", ha="center", va="center", fontsize=SMALL, linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8))
norm_b = mpl.colors.Normalize(0, 100)
cmap_b = mpl.colormaps["Oranges"]
st.assign(pct=100 * st.ami).plot(column="pct", ax=bx, cmap=cmap_b, norm=norm_b, edgecolor="white", linewidth=0.3)
dv.boundary.plot(ax=bx, color="#333333", linewidth=0.7)
for r in st.itertuples():
    p, v = r.geometry.representative_point(), 100 * r.ami
    if r.STUSPS in CALLOUT:
        y = y_top - CALLOUT.index(r.STUSPS) * 0.115e6
        bx.plot([p.x, x_east - 0.02e6], [p.y, y], color="#555555", lw=0.4)
        bx.text(x_east, y, f"{r.STUSPS} {v:.0f}", ha="left", va="center", fontsize=5.6, bbox=CLEAR)
    else:
        bx.text(p.x, p.y, f"{v:.0f}", ha="center", va="center", fontsize=5.6, color="white" if v > 60 else "black", bbox=CLEAR)
for a, title in ((ax, "(a) Census divisions, the survey's finest geography: mean linked level and, below it,\n"
                      "the share of floor area with an advanced meter"),
                 (bx, "(b) States: commercial meters that are advanced (%), 2018, Form EIA-861")):
    a.set_title(title, loc="left")
    a.set_axis_off()
for a, cmap, norm, lab in ((ax, "Blues", norm_a, "Mean linked level"), (bx, "Oranges", norm_b, "Advanced (%)")):
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a, orientation="vertical", fraction=0.03, pad=0.01,
                      shrink=0.7)
    cb.set_label(lab, fontsize=SMALL)
    cb.ax.tick_params(labelsize=5.6)
x0, y0, x1, y1 = st.total_bounds
for a in (ax, bx):   # one frame for both maps, wide enough for the callouts
    a.set_xlim(x0 - 0.05e6, x_east + 0.42e6)
    a.set_ylim(y0 - 0.05e6, y1 + 0.05e6)
    a.set_aspect("equal", adjustable="box")
fig.tight_layout(h_pad=0.6)
half = (G[("hi95", "mean linked level")] - G[("lo95", "mean linked level")]) / 2
ax.text(0.99, 0.02, f"95% intervals of the level:\n±{half.min():.2f} to ±{half.max():.2f}", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=5.6, color="#333333")
save(fig, "fig_geography")
