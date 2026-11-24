# Replication code: Digital-Focused Cross-Border M&A and Enterprise Innovation

Code for "Digital-Focused Cross-Border Mergers and Acquisitions and Enterprise Innovation: Evidence from Chinese Firms".

## Files
- `replication_dfcbma.py`: full pipeline (descriptives, baseline FE, PSM, PSM-DID, event study, robustness, 2SLS, Heckman, Callaway-Sant'Anna, mediation, heterogeneity and power). Writes all tables to `replication_tables.xlsx` and the figures to PNG.
- `make_figures.py`: redraws the event-study, parallel-trends and mediation figures from `reviewer_response_tables.xlsx`.
- `simulate_panel.py`: creates a fake firm-year panel for testing the pipeline. Its numbers are meaningless.
- `requirements.txt`: Python packages.

## Run
```
pip install -r requirements.txt
python simulate_panel.py panel_synthetic.csv      # test data only
python replication_dfcbma.py panel_synthetic.csv results
python make_figures.py reviewer_response_tables.xlsx figures
```
For the real analysis, pass your own firm-year CSV. The required columns are listed at the top of `replication_dfcbma.py`.

## Data
Firm-level data come from the Zephyr and SDC M&A databases and Chinese listed-company financial data, which are licensed and cannot be redistributed here. DATA_ACCESS_NOTE: describe how to obtain them.
