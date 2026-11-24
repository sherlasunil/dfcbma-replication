"""
Replication code
"Digital-Focused Cross-Border M&A and Enterprise Innovation: Evidence from Chinese Firms"

Implements every analysis in the manuscript and in the reviewer-response tables:
  S1  Descriptive statistics                        (Table 2)
  S2  Baseline OLS / two-way FE                     (Eq. 1, Table 3)
  S3  Propensity score matching + diagnostics       (Table 4, Cmt 12)
  S4  PSM-DID                                       (Eq. 2, Table 5)
  S5  Event study / parallel trends                 (Eq. 3, Cmt 4 and 11, Fig R1, R2)
  S6  Robustness: drop 2020, lag, industry FE       (Tables 6-8)
      IPC/CPC digital definition, digital intensity (Cmt 2, 3, 18)
      alternative innovation measures               (Table 11, Cmt 7)
  S7  Endogeneity: 2SLS and Heckman                 (Cmt 5)
  S8  Staggered adoption: Callaway-Sant'Anna style  (Cmt 6)
  S9  Mediation: Baron-Kenny, Sobel, bootstrap,
      moderation                                    (Cmt 8, 14, Fig R3)
  S10 Heterogeneity: ownership, region, industry,
      power / MDE, non-state shareholding           (Tables 9, 10, 12, Cmt 9, 15)

INPUT  one firm-year CSV (default panel.csv). Required columns:
  firm_id, year, Inn, treated (1 = firm ever does a DFCBMA), event_year (year of its first
  DFCBMA; empty for never-treated), DLCR, CAP, Patent, TMTPay, HighTech, CEOHold, FIXED,
  Tangible, CTR, EPS, Myop, PB, SOE, region (East/Central/West), industry, manufacturing
Optional columns (steps are skipped when missing):
  digital (firm-year DFCBMA dummy; defaults to Treated x Post), Inn_ipc, digital_intensity (0-3),
  patent_count, rd_exp, rd_int, patent_quality, iv, heck_excl, nonstate_share

USAGE  python replication_dfcbma.py panel.csv outdir
Needs: numpy pandas scipy scikit-learn statsmodels matplotlib openpyxl
Standard errors are clustered by firm; df correction G/(G-1) * (N-1)/(N-K).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "panel.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
os.makedirs(OUT, exist_ok=True)
SEED = 2024
N_BOOT = 500
rng = np.random.default_rng(SEED)

CONTROLS = ["DLCR", "CAP", "Patent", "TMTPay", "HighTech", "CEOHold",
            "FIXED", "Tangible", "CTR", "EPS"]
TABLES = {}     # name -> DataFrame, written to one Excel workbook at the end


# =====================================================================
# 0. Variable-construction helpers (run once on raw data, if needed)
# =====================================================================
def inverse_hhi_breadth(patents):
    """Innovation breadth (Inn) from patent-level data.
    patents: DataFrame [firm_id, year, ptype ('invention'|'utility'), ipc].
    Breadth = 1 - sum_k s_k^2 (s_k = share of the firm-year's patents in IPC class k),
    computed separately for invention and utility-model patents, then averaged.
    Design patents are excluded. The manuscript also mentions a log transform; its
    reported range (0 to 0.943) implies none was applied, so none is applied here."""
    p = patents[patents["ptype"].isin(["invention", "utility"])]
    cnt = p.groupby(["firm_id", "year", "ptype", "ipc"]).size().rename("n").reset_index()
    cnt["s"] = cnt["n"] / cnt.groupby(["firm_id", "year", "ptype"])["n"].transform("sum")
    b = (1 - cnt.assign(s2=cnt["s"] ** 2).groupby(["firm_id", "year", "ptype"])["s2"].sum())
    return b.groupby(["firm_id", "year"]).mean().rename("Inn").reset_index()


def flag_digital_target(target_code, crosswalk):
    """1 if the target's Chinese industry code (GB/T 4754) falls in a digital sub-sector.
    crosswalk: list of code strings such as '6511', '3971-3979' (UK SIC -> GB/T table)."""
    code = str(target_code)
    for c in crosswalk:
        c = str(c).replace("–", "-")
        if "-" in c:
            lo, hi = c.split("-")
            if lo.isdigit() and hi.isdigit() and lo[:3] <= code[:3] <= hi[:3] and len(code) >= 3:
                return 1
        elif c.isdigit() and code.startswith(c[:3]):
            return 1
    return 0


# =====================================================================
# Core estimation tools
# =====================================================================
def _codes(s):
    return pd.factorize(s)[0]


def _demean(M, code_list, iters=200, tol=1e-10):
    """Absorb any number of fixed effects by alternating projections."""
    M = M.copy().astype(float)
    if not code_list:
        return M
    for _ in range(iters):
        prev = M.copy()
        for c in code_list:
            n = np.bincount(c)
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(c, weights=M[:, j]) / n)[c]
        if np.max(np.abs(M - prev)) < tol:
            break
    return M


class Fit:
    pass


def fe_ols(df, y, xs, fe=(), cluster="firm_id", extra_cols=()):
    """OLS with absorbed fixed effects and one-way cluster-robust SE."""
    xs = list(xs)
    d = df.dropna(subset=[y] + xs + list(fe) + [cluster]).copy()
    Y = d[y].to_numpy(float)
    X = d[xs].to_numpy(float)
    names = list(xs)
    codes = [_codes(d[f]) for f in fe]
    if not fe:                                   # intercept: centre the data
        M = np.column_stack([Y, X])
        M = M - M.mean(axis=0)
    else:
        M = _demean(np.column_stack([Y, X]), codes)
    Yd, Xd = M[:, 0], M[:, 1:]
    keep = Xd.std(axis=0) > 1e-9
    Xk = Xd[:, keep]
    kn = [n for n, k in zip(names, keep) if k]
    XtXi = np.linalg.pinv(Xk.T @ Xk)
    b = XtXi @ Xk.T @ Yd
    e = Yd - Xk @ b
    g = _codes(d[cluster])
    G, N, K = g.max() + 1, len(d), Xk.shape[1] + (0 if fe else 1)
    for f in fe:
        if f != cluster:
            K += d[f].nunique() - 1
    S = np.zeros((G, K if False else Xk.shape[1]))
    for j in range(Xk.shape[1]):
        S[:, j] = np.bincount(g, weights=Xk[:, j] * e, minlength=G)
    V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((N - 1) / max(N - K, 1))
    se = np.sqrt(np.diag(V))
    r = Fit()
    r.params = pd.Series(b, index=kn)
    r.se = pd.Series(se, index=kn)
    r.t = r.params / r.se
    r.p = pd.Series(2 * stats.t.sf(np.abs(r.t), G - 1), index=kn)
    crit = stats.t.ppf(0.975, G - 1)
    r.ci = pd.DataFrame({"lo": r.params - crit * r.se, "hi": r.params + crit * r.se})
    r.cov = pd.DataFrame(V, index=kn, columns=kn)
    r.n, r.G = N, G
    r.r2 = 1 - (e @ e) / np.sum((Y - Y.mean()) ** 2)
    r.rmse = np.sqrt(np.mean(e ** 2))
    r.ymean, r.ysd = Y.mean(), Y.std(ddof=1)
    return r


def two_way_beta(Y, X, f_codes, y_codes):
    """Fast point estimates for bootstrap loops: Y on X with firm and year FE absorbed."""
    M = _demean(np.column_stack([Y, X]), [f_codes, y_codes], iters=60, tol=1e-8)
    Xd, Yd = M[:, 1:], M[:, 0]
    keep = Xd.std(axis=0) > 1e-9
    b = np.zeros(X.shape[1])
    b[keep] = np.linalg.lstsq(Xd[:, keep], Yd, rcond=None)[0]
    return b


def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(r, v):
    if v not in r.params.index:
        return ""
    return f"{r.params[v]:.4f}{star(r.p[v])} ({r.se[v]:.4f})"


def table(name, cols, rows, extra=None):
    """cols: {col label: Fit}; rows: variables to show. SE in parentheses."""
    out = pd.DataFrame({k: [fmt(r, v) for v in rows] for k, r in cols.items()}, index=rows)
    out.loc["Observations"] = [f"{r.n:,}" for r in cols.values()]
    out.loc["R2"] = [f"{r.r2:.4f}" for r in cols.values()]
    out.loc["Firms (clusters)"] = [f"{r.G:,}" for r in cols.values()]
    if extra:
        for k, v in extra.items():
            out.loc[k] = v
    out.loc["Note"] = ["Standard errors in parentheses, clustered by firm. *** p<0.01 ** p<0.05 * p<0.1"] + [""] * (len(cols) - 1)
    TABLES[name] = out
    print(f"\n=== {name} ===\n{out.to_string()}")
    return out


def has(df, cols):
    return all(c in df.columns for c in cols)


# =====================================================================
# Load and prepare
# =====================================================================
df = pd.read_csv(CSV)
df = df.sort_values(["firm_id", "year"]).reset_index(drop=True)
df["post"] = ((df["treated"] == 1) & (df["year"] >= df["event_year"])).astype(int)
df["dt"] = df["treated"] * df["post"]                     # Treatment x Post
if "digital" not in df.columns:
    df["digital"] = df["dt"]
df["L_dt"] = df.groupby("firm_id")["dt"].shift(1)
df["ind_year"] = df["industry"].astype(str) + "_" + df["year"].astype(str)

# =====================================================================
# S1  Descriptive statistics (Table 2)
# =====================================================================
dvars = ["Inn", "digital"] + CONTROLS + [c for c in ["Myop", "PB"] if c in df]
desc = df[dvars].describe().T[["count", "mean", "std", "min", "max"]]
desc.insert(0, "Variable", desc.index)
TABLES["T2_descriptives"] = desc.round(4)
print("\n=== Table 2 Descriptive statistics ===\n", desc.round(4).to_string(index=False))

# =====================================================================
# S2  Baseline OLS / TWFE  (Eq. 1, Table 3)
# =====================================================================
m1 = fe_ols(df, "Inn", ["digital"])
m2 = fe_ols(df, "Inn", ["digital"] + CONTROLS)
m3 = fe_ols(df, "Inn", ["digital"] + CONTROLS, fe=["firm_id"])
m4 = fe_ols(df, "Inn", ["digital"] + CONTROLS, fe=["firm_id", "year"])
table("T3_baseline_OLS", {"(1)": m1, "(2)": m2, "(3) firm FE": m3, "(4) firm+year FE": m4},
      ["digital"] + CONTROLS)

# =====================================================================
# S3  Propensity score matching (Table 4 and the Cmt 12 diagnostics)
# =====================================================================
def psm(data, covs, caliper=0.05, k=4):
    d = data.dropna(subset=covs + ["treated"]).copy()
    lg = sm.Logit(d["treated"], sm.add_constant(d[covs])).fit(disp=0)
    d["pscore"] = lg.predict(sm.add_constant(d[covs]))
    tr, co = d[d.treated == 1], d[d.treated == 0]
    lo = max(tr.pscore.min(), co.pscore.min())
    hi = min(tr.pscore.max(), co.pscore.max())
    tr = tr[(tr.pscore >= lo) & (tr.pscore <= hi)]
    co = co[(co.pscore >= lo) & (co.pscore <= hi)]
    nn = NearestNeighbors(n_neighbors=k).fit(co[["pscore"]].to_numpy())
    dist, idx = nn.kneighbors(tr[["pscore"]].to_numpy())
    ok = dist <= caliper
    treated_kept = ok.any(axis=1)
    ctrl_pos = idx[ok]
    w = pd.Series(ctrl_pos).value_counts()                # match frequency (with replacement)
    matched_co = co.iloc[w.index.to_numpy()].copy()
    matched_co["w"] = w.to_numpy()
    matched_tr = tr[treated_kept].copy()
    matched_tr["w"] = 1
    diag = {
        "Matching method": f"1:{k} nearest-neighbour on logit propensity score, with replacement",
        "Caliper": caliper,
        "Common support region": f"{lo:.3f} to {hi:.3f}",
        "Treated obs before matching": int((d.treated == 1).sum()),
        "Control obs before matching": int((d.treated == 0).sum()),
        "Treated obs off common support": int((d.treated == 1).sum() - len(tr)),
        "Treated obs lost (no match within caliper)": int(len(tr) - treated_kept.sum()),
        "Treated obs retained": int(treated_kept.sum()),
        "Treated firms retained": int(matched_tr.firm_id.nunique()),
        "Distinct matched control obs": int(len(matched_co)),
    }
    # balance table
    rows = []
    for v in covs:
        a0, b0 = d.loc[d.treated == 1, v], d.loc[d.treated == 0, v]
        t0, p0 = stats.ttest_ind(a0, b0, equal_var=False)
        rep = matched_co.loc[matched_co.index.repeat(matched_co["w"]), v]
        a1 = matched_tr[v]
        t1, p1 = stats.ttest_ind(a1, rep, equal_var=False)
        rows.append([v, "Before", a0.mean(), b0.mean(), a0.mean() - b0.mean(), t0, p0])
        rows.append([v, "After", a1.mean(), rep.mean(), a1.mean() - rep.mean(), t1, p1])
    bal = pd.DataFrame(rows, columns=["Variable", "Sample", "Mean (Treated)", "Mean (Control)",
                                      "Difference", "t", "p"])
    return pd.concat([matched_tr, matched_co]), diag, bal, lg


matched_rows, diag, balance, logit_fit = psm(df, CONTROLS)
TABLES["T4_balance"] = balance.round(4)
TABLES["Cmt12_PSM_diagnostics"] = pd.DataFrame({"Diagnostic": list(diag), "Value": list(diag.values())})
TABLES["Cmt12_pscore_logit"] = pd.DataFrame({"coef": logit_fit.params, "z": logit_fit.tvalues,
                                             "p": logit_fit.pvalues}).round(4)
print("\n=== PSM diagnostics ===")
for k_, v_ in diag.items():
    print(f"  {k_}: {v_}")
print(balance.round(4).to_string(index=False))

# matched estimation sample = all rows of the matched firms' retained observations
ms = df.loc[matched_rows.index.unique()].copy()

# =====================================================================
# S4  PSM-DID (Eq. 2, Table 5)
# =====================================================================
d1 = fe_ols(ms, "Inn", ["dt", "treated", "post"])
d2 = fe_ols(ms, "Inn", ["dt", "treated", "post"] + CONTROLS)
d3 = fe_ols(ms, "Inn", ["dt"] + CONTROLS, fe=["firm_id", "year"])
table("T5_PSM_DID", {"(1)": d1, "(2)": d2, "(3)": d3}, ["dt"] + CONTROLS,
      {"Controls": ["No", "Yes", "Yes"], "Firm FE": ["No", "No", "Yes"],
       "Year FE": ["No", "No", "Yes"]})
BASE = d3

# =====================================================================
# S5  Event study and parallel trends (Eq. 3; Cmt 4 and 11)
# =====================================================================
K_MIN, K_MAX = -5, 5
ms["rel"] = np.where(ms["treated"] == 1, ms["year"] - ms["event_year"], np.nan)
ms["rel_b"] = ms["rel"].clip(K_MIN, K_MAX)                 # bin the tails
ev_cols = []
for k_ in range(K_MIN, K_MAX + 1):
    if k_ == -1:
        continue                                            # reference period
    c = f"D_{'m' if k_ < 0 else 'p'}{abs(k_)}"
    ms[c] = (ms["rel_b"] == k_).astype(int)
    ev_cols.append((k_, c))
es = fe_ols(ms, "Inn", [c for _, c in ev_cols] + CONTROLS, fe=["firm_id", "year"])
ev = pd.DataFrame({
    "event_time": [k for k, _ in ev_cols],
    "coef": [es.params[c] for _, c in ev_cols],
    "se": [es.se[c] for _, c in ev_cols],
    "t": [es.t[c] for _, c in ev_cols],
    "p": [es.p[c] for _, c in ev_cols],
    "ci_lo": [es.ci.loc[c, "lo"] for _, c in ev_cols],
    "ci_hi": [es.ci.loc[c, "hi"] for _, c in ev_cols]})
ev = pd.concat([ev, pd.DataFrame([{"event_time": -1, "coef": 0.0}])]).sort_values("event_time")
TABLES["Cmt4_11_event_study"] = ev.round(4)
pre = [c for k_, c in ev_cols if k_ < -1]
bpre = es.params[pre].to_numpy()
chi2 = float(bpre @ np.linalg.pinv(es.cov.loc[pre, pre].to_numpy()) @ bpre)
pj = float(stats.chi2.sf(chi2, len(pre)))
TABLES["Cmt11_pretrend_joint_test"] = pd.DataFrame(
    {"Test": ["Joint Wald test, all pre-treatment coefficients = 0"],
     "chi2": [round(chi2, 3)], "df": [len(pre)], "p": [round(pj, 4)]})
print("\n=== Event study ===\n", ev.round(4).to_string(index=False), f"\nJoint pre-trend: chi2={chi2:.3f} p={pj:.4f}")

# =====================================================================
# S6  Robustness
# =====================================================================
r6 = fe_ols(ms[ms.year != 2020], "Inn", ["dt"] + CONTROLS, fe=["firm_id", "year"])
r7 = fe_ols(ms, "Inn", ["L_dt"] + CONTROLS, fe=["firm_id", "year"])
r8 = fe_ols(ms, "Inn", ["dt"] + CONTROLS, fe=["industry", "year"])
r8b = fe_ols(ms, "Inn", ["dt"] + CONTROLS, fe=["ind_year"])
table("T6_T7_T8_robustness", {"Drop 2020": r6, "Lagged dt": r7,
                              "Industry + year FE": r8, "Industry x year FE": r8b},
      ["dt", "L_dt"])

if has(ms, ["Inn_ipc"]):                                   # Section 4.5.4, Cmt 2 and 18
    a1 = fe_ols(ms, "Inn_ipc", ["dt"])
    a2 = fe_ols(ms, "Inn_ipc", ["dt"] + CONTROLS)
    a3 = fe_ols(ms, "Inn_ipc", ["dt"] + CONTROLS, fe=["firm_id", "year"])
    table("Cmt2_IPC_CPC_robustness", {"(1)": a1, "(2) controls": a2, "(3) controls+FE": a3},
          ["dt"] + CONTROLS)
if has(ms, ["digital_intensity"]):                         # Cmt 3 (0-3 ordinal score)
    ms["dig_int_norm"] = ms["digital_intensity"] / 3.0
    i1 = fe_ols(ms, "Inn", ["digital_intensity"] + CONTROLS, fe=["firm_id", "year"])
    i2 = fe_ols(ms, "Inn", ["dig_int_norm"] + CONTROLS, fe=["firm_id", "year"])
    table("Cmt3_digital_intensity", {"Ordinal 0-3": i1, "Normalised 0-1": i2},
          ["digital_intensity", "dig_int_norm"])

# alternative innovation measures, with comparable effect sizes (Table 11, Cmt 7)
alt = {"Patent breadth (Inn)": "Inn", "Patent count": "patent_count", "R&D expenditure": "rd_exp",
       "R&D intensity": "rd_int", "Patent quality": "patent_quality"}
alt_fits, alt_rows = {}, []
for lab, col in alt.items():
    if col not in ms.columns:
        continue
    d_ = ms.copy()
    if col == "patent_count":
        d_[col] = np.log1p(d_[col])
    f = fe_ols(d_, col, ["dt"] + CONTROLS, fe=["firm_id", "year"])
    alt_fits[lab] = f
    alt_rows.append([lab, f.params["dt"], f.se["dt"], f.p["dt"], f.params["dt"] / f.ysd,
                     100 * f.params["dt"] / f.ymean, f.rmse / abs(f.ymean), f.n])
if alt_rows:
    TABLES["Cmt7_alt_innovation"] = pd.DataFrame(
        alt_rows, columns=["Outcome", "Coef(dt)", "SE", "p", "Effect in SD of outcome",
                           "Effect % of outcome mean", "Relative RMSE", "N"]).round(4)
    print("\n=== Alternative innovation measures ===\n", TABLES["Cmt7_alt_innovation"].to_string(index=False))

# =====================================================================
# S7  Endogeneity: 2SLS and Heckman  (Cmt 5)
# =====================================================================
def twfe_demean(d, cols):
    c = [_codes(d["firm_id"]), _codes(d["year"])]
    return pd.DataFrame(_demean(d[cols].to_numpy(float), c), columns=cols, index=d.index)


if has(ms, ["iv"]):
    from linearmodels.iv import IV2SLS
    d = ms.dropna(subset=["Inn", "dt", "iv"] + CONTROLS)
    z = twfe_demean(d, ["Inn", "dt", "iv"] + CONTROLS)
    fs = fe_ols(d, "dt", ["iv"] + CONTROLS, fe=["firm_id", "year"])
    ctl = [c for c in CONTROLS if z[c].std() > 1e-9]   # drop controls absorbed by the FE
    iv = IV2SLS(z["Inn"], z[ctl], z[["dt"]], z[["iv"]]).fit(
        cov_type="clustered", clusters=d["firm_id"])
    f_stat = float(fs.t["iv"] ** 2)
    TABLES["Cmt5_IV_2SLS"] = pd.DataFrame({
        "Stage": ["First stage: iv -> dt", "Second stage: dt -> Inn"],
        "Coef": [fs.params["iv"], iv.params["dt"]],
        "SE": [fs.se["iv"], iv.std_errors["dt"]],
        "t": [fs.t["iv"], iv.tstats["dt"]],
        "First-stage F (cluster-robust, t^2)": [f_stat, np.nan],
        "Weak-instrument flag (F<10)": ["WEAK" if f_stat < 10 else "ok", ""]}).round(4)
    print("\n=== 2SLS ===\n", TABLES["Cmt5_IV_2SLS"].to_string(index=False))

if has(ms, ["heck_excl"]):
    # Selection equation on firm-year observations (probit of ever-treated on the controls plus
    # an exclusion variable), so the inverse Mills ratio varies over time and is not absorbed
    # by the firm fixed effects.
    sel = ms.dropna(subset=CONTROLS + ["heck_excl", "treated"]).copy()
    Xs = sm.add_constant(sel[CONTROLS + ["heck_excl"]])
    pr = sm.Probit(sel["treated"], Xs).fit(disp=0)
    xb = Xs @ pr.params
    sel["imr"] = np.where(sel["treated"] == 1, stats.norm.pdf(xb) / stats.norm.cdf(xb),
                          -stats.norm.pdf(xb) / (1 - stats.norm.cdf(xb)))
    h = fe_ols(sel, "Inn", ["dt", "imr"] + CONTROLS, fe=["firm_id", "year"])
    TABLES["Cmt5_Heckman"] = pd.DataFrame({
        "Variable": ["dt", "Inverse Mills ratio"],
        "Coef": [h.params["dt"], h.params.get("imr", np.nan)],
        "SE": [h.se["dt"], h.se.get("imr", np.nan)],
        "p": [h.p["dt"], h.p.get("imr", np.nan)]}).round(4)
    print("\n=== Heckman ===\n", TABLES["Cmt5_Heckman"].to_string(index=False))

# =====================================================================
# S8  Staggered adoption (Cmt 6)
#     ATT(g,t) vs never-treated firms, base period g-1 (Callaway-Sant'Anna, unconditional,
#     never-treated comparison). For the covariate-adjusted doubly robust version use R `did`
#     or the Python package `csdid`; this version needs no extra dependency.
# =====================================================================
def cs_np(P, ev, years):
    """P: firms x years matrix of Inn (NaN allowed); ev: first-DFCBMA year (NaN = never treated)."""
    nev = P[np.isnan(ev)]
    rows = []
    for g in np.unique(ev[~np.isnan(ev)]):
        base = np.where(years == g - 1)[0]
        if len(base) == 0:
            continue
        grp = P[ev == g]
        for j, t in enumerate(years):
            if j == base[0]:
                continue
            da = grp[:, j] - grp[:, base[0]]
            db = nev[:, j] - nev[:, base[0]]
            da, db = da[~np.isnan(da)], db[~np.isnan(db)]
            if len(da) and len(db):
                rows.append((g, t, da.mean() - db.mean(), len(da)))
    a = pd.DataFrame(rows, columns=["g", "t", "att", "n"])
    post = a[a.t >= a.g]
    overall = np.average(post.att, weights=post.n)
    by_g = post.groupby("g").apply(lambda x: np.average(x.att, weights=x.n))
    a["e"] = a.t - a.g
    dyn = a.groupby("e").apply(lambda x: np.average(x.att, weights=x.n))
    return overall, by_g, dyn


Pw = ms.pivot_table(index="firm_id", columns="year", values="Inn")
EVw = ms.groupby("firm_id")["event_year"].first().reindex(Pw.index).to_numpy(float)
YRS = Pw.columns.to_numpy()
ov, by_g, dyn = cs_np(Pw.to_numpy(), EVw, YRS)
bo, bg_ = [], []
for _ in range(200):                                   # firm-level bootstrap
    ix = rng.integers(0, len(Pw), len(Pw))
    try:
        o_, g_, _d = cs_np(Pw.to_numpy()[ix], EVw[ix], YRS)
        bo.append(o_); bg_.append(g_)
    except Exception:
        pass
cs_se = np.std(bo, ddof=1)
cs_tab = pd.DataFrame({"Cohort (first DFCBMA year)": by_g.index, "ATT": by_g.values})
cs_tab["SE (firm bootstrap)"] = pd.concat(bg_, axis=1).std(axis=1, ddof=1).reindex(by_g.index).values
cs_tab.loc[len(cs_tab)] = ["Overall ATT", ov, cs_se]
cs_tab["95% CI"] = [f"[{a - 1.96 * s:.4f}, {a + 1.96 * s:.4f}]" for a, s in
                    zip(cs_tab["ATT"], cs_tab["SE (firm bootstrap)"])]
TABLES["Cmt6_CallawaySantAnna"] = cs_tab.round(4)
TABLES["Cmt6_CS_dynamic"] = pd.DataFrame({"event_time": dyn.index, "ATT": dyn.values}).round(4)
TABLES["Cmt6_TWFE_vs_CS"] = pd.DataFrame({
    "Estimator": ["TWFE PSM-DID (Table 5 col 3)", "Callaway-Sant'Anna overall ATT"],
    "Coefficient": [BASE.params["dt"], ov], "SE": [BASE.se["dt"], cs_se]}).round(4)
print("\n=== Staggered adoption ===\n", cs_tab.round(4).to_string(index=False))

# =====================================================================
# S9  Mediation: Baron-Kenny, Sobel, cluster bootstrap, moderation (Cmt 8, 14)
#     Myop = managerial myopia (R&D-based, as in the manuscript); PB = financing capacity.
#     Interpretation caveat: mediator effects are not identified causally by TWFE alone.
# =====================================================================
FE2 = ["firm_id", "year"]
med_rows, boot_rows, ind_effects = [], [], {}
for mname in [m for m in ["Myop", "PB"] if m in ms.columns]:
    c1 = fe_ols(ms, "Inn", ["dt"] + CONTROLS, fe=FE2)                      # total effect c
    c2 = fe_ols(ms, mname, ["dt"] + CONTROLS, fe=FE2)                      # path a
    c3 = fe_ols(ms, "Inn", ["dt", mname] + CONTROLS, fe=FE2)               # paths c' and b
    a_, b_ = c2.params["dt"], c3.params[mname]
    sa, sb = c2.se["dt"], c3.se[mname]
    ab = a_ * b_
    z_ = ab / np.sqrt(b_ ** 2 * sa ** 2 + a_ ** 2 * sb ** 2)
    table(f"Cmt14_mediation_{mname}", {"(1) Total: Inn": c1, f"(2) {mname}": c2,
                                       "(3) Direct: Inn": c3}, ["dt", mname])
    # cluster bootstrap of a*b (firms resampled with replacement)
    base = ms.dropna(subset=["Inn", mname, "dt"] + CONTROLS)
    fcode, ycode = _codes(base["firm_id"]), _codes(base["year"])
    Xa = base[["dt"] + CONTROLS].to_numpy(float)
    Xb = base[["dt", mname] + CONTROLS].to_numpy(float)
    Ym, Yi = base[mname].to_numpy(float), base["Inn"].to_numpy(float)
    rows_by_firm = [np.where(fcode == f)[0] for f in range(fcode.max() + 1)]
    bs_ab = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(rows_by_firm), len(rows_by_firm))
        idx = np.concatenate([rows_by_firm[f] for f in pick])
        newf = np.repeat(np.arange(len(pick)), [len(rows_by_firm[f]) for f in pick])
        try:
            ba = two_way_beta(Ym[idx], Xa[idx], newf, ycode[idx])
            bb = two_way_beta(Yi[idx], Xb[idx], newf, ycode[idx])
            bs_ab.append(ba[0] * bb[1])
        except Exception:
            pass
    lo_, hi_ = np.percentile(bs_ab, [2.5, 97.5])
    ind_effects[mname] = (ab, np.std(bs_ab, ddof=1), lo_, hi_)
    med_rows.append([f"Digital M&A -> {mname} -> Inn", a_, b_, ab, z_, 2 * stats.norm.sf(abs(z_)),
                     c3.params["dt"], c1.params["dt"], 100 * ab / c1.params["dt"]])
    boot_rows.append([mname, ab, np.std(bs_ab, ddof=1), lo_, hi_,
                      "Partial" if (lo_ > 0 or hi_ < 0) and abs(c3.params["dt"] / c3.se["dt"]) > 1.96 else
                      "Full" if (lo_ > 0 or hi_ < 0) else "None"])
if med_rows:
    TABLES["Cmt14_Sobel"] = pd.DataFrame(med_rows, columns=[
        "Pathway", "a", "b", "a*b", "Sobel Z", "p", "Direct c'", "Total c", "% mediated"]).round(4)
    TABLES["Cmt14_bootstrap"] = pd.DataFrame(boot_rows, columns=[
        "Mediator", "Indirect effect", "Bootstrap SE", "CI lower", "CI upper", "Mediation"]).round(4)
    print("\n=== Sobel ===\n", TABLES["Cmt14_Sobel"].to_string(index=False))
    print("\n=== Bootstrap ===\n", TABLES["Cmt14_bootstrap"].to_string(index=False))
if "Myop" in ms.columns:                                   # moderation (Cmt 14)
    ms["dt_x_Myop"] = ms["dt"] * ms["Myop"]
    mo = fe_ols(ms, "Inn", ["dt", "Myop", "dt_x_Myop"] + CONTROLS, fe=FE2)
    table("Cmt14_moderation_Myop", {"Inn": mo}, ["dt", "Myop", "dt_x_Myop"])

# =====================================================================
# S10 Heterogeneity (Tables 9, 10, 12; Cmt 9 and 15)
# =====================================================================
def split_fit(d, mask):
    return fe_ols(d[mask], "Inn", ["dt"] + CONTROLS, fe=FE2)


soe, non = split_fit(ms, ms.SOE == 1), split_fit(ms, ms.SOE == 0)
table("T9_ownership", {"SOE": soe, "Non-SOE": non}, ["dt"])
reg = {r: split_fit(ms, ms.region == r) for r in ["East", "Central", "West"] if (ms.region == r).sum() > 50}
table("T10_region", reg, ["dt"])
man, nman = split_fit(ms, ms.manufacturing == 1), split_fit(ms, ms.manufacturing == 0)
table("T12_industry", {"Manufacturing": man, "Non-manufacturing": nman}, ["dt"])

# interaction-term tests of the subgroup differences (Cmt 9)
ms["dt_x_SOE"] = ms["dt"] * ms["SOE"]
ms["dt_x_manuf"] = ms["dt"] * ms["manufacturing"]
ms["dt_x_East"] = ms["dt"] * (ms["region"] == "East").astype(int)
ms["dt_x_West"] = ms["dt"] * (ms["region"] == "West").astype(int)
i_soe = fe_ols(ms, "Inn", ["dt", "dt_x_SOE"] + CONTROLS, fe=FE2)
i_man = fe_ols(ms, "Inn", ["dt", "dt_x_manuf"] + CONTROLS, fe=FE2)
i_reg = fe_ols(ms, "Inn", ["dt", "dt_x_East", "dt_x_West"] + CONTROLS, fe=FE2)
table("Cmt9_interaction_tests", {"Ownership": i_soe, "Industry": i_man, "Region": i_reg},
      ["dt", "dt_x_SOE", "dt_x_manuf", "dt_x_East", "dt_x_West"])

# statistical power / minimum detectable effect (80% power, 5% two-sided)
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
pw = []
for lab, f in [("SOE", soe), ("Non-SOE", non), ("East", reg.get("East")), ("Central", reg.get("Central")),
               ("West", reg.get("West")), ("Manufacturing", man), ("Non-manufacturing", nman)]:
    if f is None:
        continue
    se_, b_ = f.se["dt"], f.params["dt"]
    mde = (z_a + z_b) * se_
    power = stats.norm.cdf(abs(b_) / se_ - z_a) + stats.norm.cdf(-abs(b_) / se_ - z_a)
    pw.append([lab, f.n, b_, se_, mde, power])
TABLES["Cmt9_power_MDE"] = pd.DataFrame(pw, columns=["Subgroup", "N", "Coef", "SE",
                                                     "MDE (80% power, 5% alpha) = 2.8 x SE",
                                                     "Observed power at estimate"]).round(4)
print("\n=== Power / MDE ===\n", TABLES["Cmt9_power_MDE"].to_string(index=False))

if has(ms, ["nonstate_share"]):                            # Cmt 15: direct test in SOE sample
    s = ms[ms.SOE == 1].copy()
    s["dt_x_ns"] = s["dt"] * s["nonstate_share"]
    ns = fe_ols(s, "Inn", ["dt", "nonstate_share", "dt_x_ns"] + CONTROLS, fe=FE2)
    table("Cmt15_SOE_nonstate_share", {"SOE sample": ns}, ["dt", "nonstate_share", "dt_x_ns"])

# =====================================================================
# Figures (requested plotting style)
# =====================================================================
fams = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"figure.figsize": (9, 7), "font.size": 18, "font.weight": "bold",
                     "font.family": "Times New Roman" if "Times New Roman" in fams else "Liberation Serif",
                     "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.grid": False,
                     "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
                     "xtick.color": "black", "ytick.color": "black"})
FS = 18


def finish(ax, xl, yl, ti, name):
    ax.set_xlabel(xl, fontsize=FS); ax.set_ylabel(yl, fontsize=FS); ax.set_title(ti, fontsize=FS)
    ax.tick_params(axis="both", labelsize=FS)
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_fontweight("bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=300); plt.close()


e = ev.sort_values("event_time")
e["ci_lo"] = e["ci_lo"].fillna(0); e["ci_hi"] = e["ci_hi"].fillna(0)
t_, b_ = e.event_time.to_numpy(), e.coef.to_numpy()
lo_, hi_ = e.ci_lo.to_numpy(), e.ci_hi.to_numpy()
fig, ax = plt.subplots()
ax.errorbar(t_, b_, yerr=[b_ - lo_, hi_ - b_], fmt="o-", color="#1f77b4", capsize=6, lw=2.2, ms=9)
ax.fill_between(t_, lo_, hi_, color="#1f77b4", alpha=.15)
ax.axhline(0, color="black", ls="--", lw=1.2); ax.axvline(-0.5, color="black", ls=":", lw=1.2)
ax.set_xticks(t_)
finish(ax, "Event time (years relative to DFCBMA)", "Coefficient on Innovation", "Event Study", "Fig_R1_event_study")

p_ = e[e.event_time < 0]
fig, ax = plt.subplots()
ax.errorbar(p_.event_time, p_.coef, yerr=[p_.coef - p_.ci_lo, p_.ci_hi - p_.coef], fmt="s-",
            color="#2ca02c", capsize=7, lw=2.2, ms=11)
ax.axhline(0, color="black", ls="--", lw=1.2)
ax.set_xticks(p_.event_time); ax.set_xlim(K_MIN - .6, -.4)
ax.set_ylim(p_.ci_lo.min() * 1.15, p_.ci_hi.max() + 0.35 * (p_.ci_hi.max() - p_.ci_lo.min()))
ax.annotate(f"Joint test p = {pj:.3f}", xy=(0.5, 0.93), xycoords="axes fraction", ha="center",
            fontsize=FS, color="black")
finish(ax, "Pre-treatment event time (years)", "Coefficient on Innovation", "Parallel Trends", "Fig_R2_parallel_trends")

if ind_effects:
    names = list(ind_effects)
    v = np.array([ind_effects[n][0] for n in names])
    l_ = np.array([ind_effects[n][2] for n in names]); h_ = np.array([ind_effects[n][3] for n in names])
    fig, ax = plt.subplots()
    ax.bar(range(len(v)), v, width=.5, color="#d62728", edgecolor="black", linewidth=1.5)
    ax.errorbar(range(len(v)), v, yerr=[v - l_, h_ - v], fmt="none", ecolor="black", capsize=10, elinewidth=2.2)
    ax.axhline(0, color="black", lw=1.2); ax.set_xticks(range(len(v))); ax.set_xticklabels(names)
    finish(ax, "Mediation pathway", "Indirect effect (a × b)", "Indirect Effects", "Fig_R3_mediation_effects")

# =====================================================================
# Save every table
# =====================================================================
with pd.ExcelWriter(f"{OUT}/replication_tables.xlsx") as xw:
    for k_, v_ in TABLES.items():
        v_.to_excel(xw, sheet_name=k_[:31])
print(f"\nDone. {len(TABLES)} tables -> {OUT}/replication_tables.xlsx ; figures -> {OUT}/")
