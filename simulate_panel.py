"""Synthetic firm-year panel with the column layout expected by replication_dfcbma.py.
FAKE DATA for testing that the pipeline runs. It says nothing about the paper's results.
Usage: python simulate_panel.py panel_synthetic.csv"""
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
out = sys.argv[1] if len(sys.argv) > 1 else "panel_synthetic.csv"
N, YEARS = 1500, np.arange(2012, 2023)
rows = []
for i in range(N):
    soe = rng.random() < .25
    region = rng.choice(["East", "Central", "West"], p=[.6, .25, .15])
    ind = rng.integers(1, 13)
    treated = rng.random() < (.30 if not soe else .15)
    ev = rng.choice(np.arange(2014, 2022)) if treated else np.nan
    a = rng.normal(0, .15)
    base = dict(DLCR=rng.beta(2, 6), CAP=rng.normal(.6, .3), TMTPay=rng.normal(15, 1),
                HighTech=float(rng.random() < .8), FIXED=rng.beta(2, 8), Tangible=rng.beta(8, 2))
    ns = rng.random() * .5 if soe else 0
    for y in YEARS:
        post = treated and y >= ev
        k = (y - ev) if treated else -9
        eff = 0 if not post else (0.20 if not soe else 0.0) * min(1, (k + 1) / 4)
        myop = .026 - .08 * post * .3 + rng.normal(0, .02)
        pb = 2 + .11 * post + rng.normal(0, .5)
        inn = .3 + a + eff - .05 * myop * 10 * post + .03 * (y - 2012) / 10 + rng.normal(0, .08)
        rows.append(dict(
            firm_id=i, year=y, Inn=inn, treated=int(treated), event_year=ev, **base,
            Patent=np.log1p(rng.poisson(30)), CEOHold=rng.normal(10, 5), CTR=rng.normal(.025, .01),
            EPS=rng.normal(.5, .4), Myop=myop, PB=pb, SOE=int(soe), region=region, industry=ind,
            manufacturing=int(ind <= 6), nonstate_share=ns,
            Inn_ipc=inn * .9 + rng.normal(0, .03), digital_intensity=int(post) * rng.integers(1, 4),
            patent_count=rng.poisson(20 * (1 + .3 * post)), rd_exp=.5 + .09 * post + rng.normal(0, .1),
            rd_int=.02 + .018 * post + rng.normal(0, .01), patent_quality=1 + .38 * post + rng.normal(0, .4),
            iv=rng.normal(0, 1) + .5 * treated, heck_excl=rng.normal(0, 1) + .3 * treated))
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, len(rows), "rows")
