# Kazakhstan Oil-Revenue Dependence Monitor

An **offline-capable local Streamlit dashboard** describing the relationship between Kazakhstan's annual oil-sector tax receipts into the National Fund and state-budget income. The repository contains annual observations for **2007–2025**, an explanatory first-difference regression, saved model results, and a comparison against a naive forecasting benchmark.

> **Important:** The indicator is a *constructed ratio*, not an official measure of Kazakhstan's total oil dependence. It does **not** include every category of oil-related government revenue. The regression reports statistical associations, not causal effects, and it did **not** beat the naive benchmark on held-out RMSE.

## 1. Quick start (Windows / PowerShell)

**Requirements:** Python 3.10+ (Python 3.10 was used in development), a browser, and the files from this repository. Internet access is needed to install dependencies initially; the dashboard uses local files after installation.

Open a terminal **in the repository's root folder** (the one containing `app.py` and `requirements.txt`), then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal (typically `http://localhost:8501`). To stop the server, press **Ctrl+C** in the terminal.

If `python` is not recognized, install Python and reopen the terminal. If the first launch fails, check that `data/processed/master_dataset.csv` and `model/artifact.json` exist.

## 2. Indicator definition

For each calendar year *t*:

```text
Y_t (%) = 100 × OilTaxNF_t / StateIncome_t
```

- **OilTaxNF:** direct taxes from oil-sector organizations received by the National Fund, in thousand KZT.
- **StateIncome:** reported state-budget income, in thousand KZT.
- **Y:** an annual **ratio**, reported in percent. Its annual difference is measured in **percentage points**.

The numerator and denominator do **not** represent a complete consolidated-budget oil/non-oil decomposition. Do not describe `Y` as the share of *all* oil revenue in *all* government revenue. Detailed provenance and limitations are recorded in [`SOURCES.md`](SOURCES.md).

## 3. Project structure

```text
app.py                                  Streamlit presentation; reads cached data and saved artifact
requirements.txt                        Pinned Python dependencies
README.md                               Installation, reproduction, and limitations
SOURCES.md                              Original sources, transformations, and qualifications
data/raw/                                Archived source materials and source-derived input CSVs
  raw_manual_input.csv                  Manually transcribed fiscal values with documented sources
  brent_usd_per_bbl_annual_avg.csv     Annual Brent series
  usd_kzt_annual_avg_input.csv          Annual FX series
  extra_material_oil_output.xlsx        Oil-production source table
data/processed/
  master_dataset.csv                    Canonical dataset used by the dashboard
  master_macro_dataset_2007_2025_UPDATED.xlsx  Earlier analysis dataset (used by model scripts)
src/
  ingest.py                              Rebuilds canonical CSV from source-derived inputs
  dataset.py                             Loads and validates canonical CSV for the dashboard
  build_artifact.py                      Fits explanatory difference regression; exports artifact
  difference_backtest.py                Creates held-out forecasts / comparison CSV
  model_comparison.py                   Other exploratory backtests
  stationarity.py, first_differences.py ADF / KPSS exploratory checks
  diagnostics.py, robustness.py         Other diagnostic / robustness checks
  ...                                   Additional exploration and verification scripts
model/artifact.json                      Saved coefficients, covariance, diagnostics, forecast scores
outputs/
  backtest_forecasts.csv
  backtest_scores.csv
  difference_backtest.csv
```

## 4. Refresh data and rebuild model

The committed snapshot is sufficient for normal dashboard use; **do not fit models inside `app.py`**.

To regenerate the *canonical* processed CSV from the committed source-derived inputs:

```powershell
python src/ingest.py
```

To regenerate the held-out difference-model forecast CSV and the JSON artifact:

```powershell
python src/difference_backtest.py
python src/build_artifact.py
```

**Current pipeline limitation:** `ingest.py` writes `data/processed/master_dataset.csv`, whereas `build_artifact.py` currently reads the *separate* `data/processed/master_macro_dataset_2007_2025_UPDATED.xlsx`. These two data representations agree for the current committed 2007–2025 series, but they are **not yet wired into a single end-to-end rebuild**. If values are updated, reconcile both datasets and rebuild the artifact before displaying new numbers; otherwise, the website could show a newer headline beside older coefficients. Do not claim that `ingest.py` alone rebuilds every result.

Source downloads in `data/raw/` are archival evidence; the present ingestion script uses source-derived CSV/XLSX inputs rather than automatically re-extracting every original PDF or budget report. The extraction and verification history is explained in `SOURCES.md`.

## 5. What the dashboard shows

The sidebar contains five sections:

1. **Overview:** latest year and ratio, yearly changes, historical chart, and selected source variables.
2. **Regression & Drivers:** saved first-difference model, coefficient uncertainty, fitted-versus-observed change decomposition, and available diagnostics.
3. **Scenario Analysis:** illustrative response to a hypothetical change in the annual Brent price, with a **95% confidence interval for the conditional mean** computed using the artifact's covariance matrix. This is **not** a future-outcome prediction interval or causal simulation.
4. **Model Comparison & Backtesting:** saved out-of-sample accuracy and forecasts for 2022–2025.
5. **Key Findings & Sources:** indicator definition, provenance, and interpretation limits.

The site reads local processed data and `model/artifact.json`; it **does not run regression estimation on page load**. It identifies the snapshot as cached. Some optional sections, including detailed stationarity and robustness tables, may display a missing-results notice because those fields are not yet part of the current artifact.

## 6. Model and evaluation

The saved explanatory model is:

```text
ΔY_t = α + β × ΔBrent_t + ε_t
```

It uses **18 annual differences (2008–2025)**, derived from 19 observations. Annual Brent changes are contemporaneous, so the coefficients **must not** be represented as a verified one-year-ahead forecast using future Brent prices.

For the separately evaluated **one-year-ahead** difference-model forecast, the code uses the *previous year's* Brent change to avoid test-year information leakage. The held-out period is **2022–2025**. Published saved results (percentage points):

| Method | MAE | RMSE |
|---|---:|---:|
| Naive: next Y equals previous Y | 8.5724 | **10.3384** |
| Lagged-Brent difference forecast | 8.5515 | 11.3004 |

The naive method achieved **lower held-out RMSE**; this negative finding is intentional and should be disclosed in the demonstration and technical note. Four test observations are insufficient for strong claims about long-run predictive superiority.

The original level regressions showed serial correlation; the first-difference regression's Breusch–Godfrey test did not reject the null at 5%, but this is not proof of independent errors. Tests for unit roots, structural changes, and specification stability are especially uncertain with such a small annual sample.

## 7. Offline use and error handling

After packages are installed, disconnecting the network should not prevent the app from reading the **committed cached CSV and JSON artifact**. It labels the data as an offline cached snapshot, not live government data.

Error-handling checks:

- Temporarily rename `model/artifact.json`: the app should show a clear missing-model message. Restore the name afterward.
- Temporarily rename `data/processed/master_dataset.csv`: the app should show a missing-dataset message. Restore the name afterward.
- The optional backtest detail CSV can be missing: summary scores should still be available from the artifact, with a warning.

These are **manual acceptance checks**; the repository should still be tested from a fresh ZIP / clone on a different computer before submission. The page-load-time target is **under three seconds from cached data**; measure this locally rather than assuming it is met.

## 8. Reproducibility checklist (before submission)

- [ ] Install pinned dependencies on a clean machine without manual package fixes.
- [ ] Start the app using the two commands in §1 within the instructor's time limit.
- [ ] Verify current ratio, unit, reference year, and latest change are visible without scrolling.
- [ ] Measure cached page rendering time (target: <3 seconds).
- [ ] Check three displayed model contributions against `model/artifact.json`.
- [ ] Move the scenario slider and verify output and uncertainty reflect the saved covariance.
- [ ] Disconnect the network and reload; cached data must be clearly labeled.
- [ ] Rename the model artifact and reload; a readable warning must appear.
- [ ] Verify there is no hard-coded headline in `app.py` and no regression call in the presentation path.
- [ ] Check Git history for credentials and personal/sensitive files; record the peer replication receipt.

Suggested numeric reproduction tolerance for the displayed indicator: **±0.01 percentage points** (subject to team confirmation).

## 9. Known limitations and outstanding submission work

- Indicator omits additional oil-related public revenues and is **not** the official aggregate oil-revenue share.
- Annual frequency and the **19-observation exception** were reportedly approved by the instructor; document that approval in the technical note.
- Some historical National Fund documents are labeled *draft* in the source archive (including 2008, 2009, 2013, 2017, and 2020). Verify vintage/status and document remaining uncertainty.
- The current source-derived fiscal table includes manually transcribed inputs; verify the original reports before relying on every year.
- Some regression diagnostics and robustness outputs are not yet serialized into `model/artifact.json`.
- Current source-to-model rebuild is not a single unified pipeline (see §4).
- A final technical note (maximum four pages), appendix of full diagnostics and robustness tests, APA references, AI-use disclosure, contribution statements, and independent replication receipt must still be assembled and validated before submission.

## 10. References, reuse, and authorship

See [`SOURCES.md`](SOURCES.md) for the official data sources and references. Python packages are listed in `requirements.txt` and their respective licenses apply. **Before submitting:** list any externally copied code with its original URL, license, and affected files/lines, where applicable. Each member should identify and sign their personal contributions. Declare substantive AI assistance and independently verified results in the technical-note annex as required by the instructor.

**Repository:** https://github.com/asylhanasem0906-lang/kz-oil-revenue-dependence
