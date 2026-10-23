"""Read-only Streamlit presentation of the team's committed dataset and model artifact."""
import json
from pathlib import Path
from statistics import NormalDist

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/processed/master_dataset.csv'
ARTIFACT = ROOT / 'model/artifact.json'
BACKTEST = ROOT / 'outputs/difference_backtest.csv'

st.set_page_config(page_title='Kazakhstan | Oil Revenue Monitor', page_icon='📊', layout='wide')
st.markdown('''<style>
.stApp {background:#f6f8fc;} h1,h2,h3 {color:#193b60;}
[data-testid="stMetric"] {background:white;border:1px solid #e1e7ef;border-radius:13px;padding:15px 18px;}
[data-testid="stMetricValue"] {color:#135c9b;}
.block-container {padding-top:2rem;max-width:1250px;}
[data-testid="stMetricLabel"] p {font-size:0.88rem;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
[data-testid="stSidebar"] {background:#edf2f8;}

</style>''', unsafe_allow_html=True)

st.sidebar.title('Oil Revenue Monitor')
section = st.sidebar.radio('Navigation', ['Overview', 'Regression & Drivers', 'Scenario Analysis', 'Model Comparison & Backtesting', 'Key Findings & Sources'])
st.sidebar.caption('Kazakhstan · Annual series · Offline snapshot')

if not DATA.is_file():
    st.error('Processed dataset is missing. Restore data/processed or run the documented data pipeline.')
    st.stop()
if not ARTIFACT.is_file():
    st.error('Model artifact is missing: model/artifact.json. Rebuild it using the documented model command; the website does not estimate models.')
    st.stop()
try:
    from src.dataset import load_dataset
    df = load_dataset()
    with ARTIFACT.open(encoding='utf-8') as f:
        art = json.load(f)
    needed = ['Year', 'Y (%)', 'Brent annual avg (USD/bbl)', 'Oil incl condensate (million t)']
    if any(c not in df.columns for c in needed):
        raise ValueError('Missing required data columns')
    for key in ['coefficients', 'covariance_matrix', 'forecast_test', 'estimated_at_utc']:
        if key not in art:
            raise ValueError(f'Missing artifact field: {key}')
except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
    st.error(f'Unable to load project data or model artifact: {e}')
    st.stop()

st.sidebar.success('Offline cached snapshot loaded')
st.sidebar.caption(f"Model estimated: {art['estimated_at_utc'][:10]} (UTC)")
first_year, last_year = int(df['Year'].min()), int(df['Year'].max())
last, previous = df.iloc[-1], df.iloc[-2]
current_y = float(last['Y (%)'])
previous_y = float(previous['Y (%)'])
change_y = current_y - previous_y


def annual_line_chart(frame, value_col, title, unit, color='#2166a5'):
    """Calendar-year x-axis without thousands separators."""
    data = frame[['Year', value_col]].copy()
    data['Year label'] = data['Year'].astype(int).astype(str)
    line = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=50), strokeWidth=2.8, color=color).encode(
        x=alt.X('Year label:O', title='Year', sort=list(data['Year label'])),
        y=alt.Y(f'{value_col}:Q', title=unit, scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('Year label:N', title='Year'), alt.Tooltip(f'{value_col}:Q', title=title, format='.2f')]
    )
    st.altair_chart(line.properties(height=295), use_container_width=True)


if section == 'Overview':
    st.title('Kazakhstan Oil Revenue Dependence')
    st.write('**Indicator:** direct oil-sector taxes paid into the National Fund ÷ state budget income × 100. This is a constructed research indicator, not an official oil-dependence statistic.')
    a,b,c = st.columns(3)
    a.metric(f'Latest value · {last_year}', f'{current_y:.2f}%', delta=f'{change_y:+.2f} pp vs {int(previous["Year"])}')
    b.metric('Annual Brent average', f'{float(last["Brent annual avg (USD/bbl)"]):.2f} USD/bbl')
    c.metric('Sample', f'{first_year}–{last_year}', f'{len(df)} annual observations', delta_color='off')
    years = st.slider('Historical range', first_year, last_year, (first_year, last_year))
    view = df[df['Year'].between(*years)]
    st.subheader('Historical indicator (%)')
    annual_line_chart(view, 'Y (%)', 'Indicator', 'Percent (%)')
    st.caption('Source: project processed dataset; original National Fund and budget reports documented in SOURCES.md. Annual observations; cached snapshot.')
    left,right = st.columns(2)
    with left:
        st.subheader('Brent oil price')
        annual_line_chart(view, 'Brent annual avg (USD/bbl)', 'Brent', 'USD per barrel')
    with right:
        st.subheader('Oil and condensate production')
        annual_line_chart(view, 'Oil incl condensate (million t)', 'Production', 'Million tonnes')
    with st.expander('Explore underlying annual data'):
        st.dataframe(view, hide_index=True, width='stretch')

elif section == 'Regression & Drivers':
    st.title('Regression & Drivers')
    st.write('**Saved specification:** annual change in indicator (percentage points) regressed on same-year annual change in Brent price (USD/bbl).')
    coeff = art['coefficients']
    a,b,c = st.columns(3)
    a.metric('R-squared (in sample)', f"{art['r_squared']:.4f}")
    b.metric('ΔBrent coefficient', f"{coeff['dBrent']:.4f}")
    c.metric('Usable observations', str(art['n_observations']))
    st.subheader('Saved coefficients and standard errors')
    st.dataframe(pd.DataFrame({'Term':list(coeff.keys()), 'Coefficient':list(coeff.values()), 'Standard error':[art['standard_errors'][k] for k in coeff]}), hide_index=True, width='stretch')
    dbrent = float(last['Brent annual avg (USD/bbl)'] - previous['Brent annual avg (USD/bbl)'])
    intercept = float(coeff['const'])
    contribution = float(coeff['dBrent']) * dbrent
    st.subheader(f'Accounting decomposition of fitted change · {last_year}')
    residual = change_y - (intercept + contribution)
    chart_data = pd.DataFrame({
        'Component': ['Intercept', 'Brent price change', 'Unexplained residual'],
        'Contribution (pp)': [intercept, contribution, residual],
    })
    chart_data['Label'] = chart_data['Contribution (pp)'].map(lambda v: f'{v:+.2f} pp')
    bars = alt.Chart(chart_data).mark_bar(size=27).encode(
        y=alt.Y('Component:N', sort=['Intercept', 'Brent price change', 'Unexplained residual'], title=None),
        x=alt.X('Contribution (pp):Q', title='Percentage points (pp)'),
        color=alt.condition(alt.datum['Contribution (pp)'] >= 0, alt.value('#26846b'), alt.value('#2166a5')),
        tooltip=['Component:N', alt.Tooltip('Contribution (pp):Q', format='+.2f')]
    )
    zero_line = alt.Chart(pd.DataFrame({'zero': [0]})).mark_rule(color='#7c8798').encode(x='zero:Q')
    st.altair_chart((bars + zero_line).properties(height=160), use_container_width=True)
    st.caption('Intercept + Brent contribution = fitted change; adding the residual reconciles the model with the actual observed change. The residual is not an identified economic driver.')
    st.write(f'**Intercept:** {intercept:+.2f} pp · **Brent contribution:** {contribution:+.2f} pp · **Residual:** {residual:+.2f} pp')
    st.write(f'Observed ΔBrent: **{dbrent:+.2f} USD/bbl** · Model-implied Δindicator: **{intercept + contribution:+.2f} pp** · Actual Δindicator: **{change_y:+.2f} pp**.')
    st.caption('These are conditional regression contributions, not identified causal effects. Production and exchange rate are not drivers in this saved model.')
    d = art.get('diagnostics', {})
    st.subheader('Selected saved diagnostics')
    st.dataframe(pd.DataFrame([{'Breusch–Godfrey p':d.get('breusch_godfrey_p'), 'Breusch–Pagan p':d.get('breusch_pagan_p'), 'Durbin–Watson':d.get('durbin_watson')}]).round(4), hide_index=True, width='stretch')
    with st.expander('Stationarity: ADF and KPSS', expanded=True):
        stationarity = art.get('stationarity', {})
        if stationarity:
            labels = {'indicator_level':'Indicator (level)', 'brent_level':'Brent (level)', 'indicator_difference':'ΔIndicator', 'brent_difference':'ΔBrent'}
            station_rows = []
            for key, label in labels.items():
                val = stationarity.get(key, {})
                if val:
                    station_rows.append({'Series':label, 'n':val.get('n_observations'), 'ADF p-value':val.get('adf_p'), 'KPSS p-value':val.get('kpss_p'), 'KPSS upper-bound flag':val.get('kpss_p_bounded', False)})
            st.dataframe(pd.DataFrame(station_rows).round(4), hide_index=True, use_container_width=True)
            st.caption('ADF null: unit root. KPSS null: stationarity. A KPSS p-value of 0.10 with an upper-bound flag means p ≥ 0.10, not exactly 0.10. With 18–19 annual points, these tests have limited power.')
        else:
            st.warning('Stationarity results are missing from the saved artifact.')
    with st.expander('Robustness and exploratory structural break', expanded=True):
        robustness = art.get('robustness', {})
        if robustness:
            robust_rows = []
            for key, label in [('full_sample','Full sample'), ('exclude_2020','Excluding 2020'), ('from_2012','2012 onward')]:
                val = robustness.get(key, {})
                if val:
                    robust_rows.append({'Sample':label, 'n':val.get('n'), 'Brent coefficient':val.get('brent_coefficient'), 'Brent p-value':val.get('brent_p'), 'R²':val.get('r_squared')})
            st.dataframe(pd.DataFrame(robust_rows).round(4), hide_index=True, use_container_width=True)
        structural = art.get('structural_break_exploratory', {})
        if structural:
            st.write(f"**Exploratory Chow split at {structural.get('split_year')}:** F = {structural.get('f_statistic',float('nan')):.3f}; p = {structural.get('p_value',float('nan')):.4f}. No statistically significant break detected at this split; this is not evidence that the coefficients are stable across all dates.")
            st.caption(f"Pre-split n = {structural.get('pre_n')}; post-split n = {structural.get('post_n')}. Results are exploratory given the very small sub-samples.")


elif section == 'Scenario Analysis':
    st.title('Scenario Analysis')
    st.write('Change the **hypothetical annual Brent price movement** and inspect the model-implied change in the indicator. The baseline level is the latest observed value.')
    delta = st.slider('Hypothetical change in annual Brent average (USD/bbl)', min_value=-60.0, max_value=60.0, value=0.0, step=1.0)
    beta = art['coefficients']
    covariance = np.asarray(art['covariance_matrix'], dtype=float)
    x = np.array([1.0, delta])
    predicted_change = float(beta['const'] + beta['dBrent'] * delta)
    se_mean = float(np.sqrt(max(0.0, x @ covariance @ x)))
    z = NormalDist().inv_cdf(0.975)
    lower = current_y + predicted_change - z * se_mean
    upper = current_y + predicted_change + z * se_mean
    a,b,c = st.columns(3)
    a.metric('Model-implied level', f'{current_y + predicted_change:.2f}%', delta=f'{predicted_change:+.2f} pp vs {last_year}')
    b.metric('95% CI lower', f'{lower:.2f}%')
    c.metric('95% CI upper', f'{upper:.2f}%')
    st.caption('The two bounds form a confidence interval for the conditional mean, not a prediction interval for a future observation.')
    historical_changes = df['Brent annual avg (USD/bbl)'].diff().dropna()
    min_change, max_change = float(historical_changes.min()), float(historical_changes.max())
    st.caption(f'Observed annual Brent changes in the dataset: {min_change:+.2f} to {max_change:+.2f} USD/bbl (historical context, not a validity guarantee).')
    if delta < min_change or delta > max_change:
        st.warning('Selected Brent change is outside the observed annual-change range: extrapolation risk is elevated.')
    if not 0 <= current_y + predicted_change <= 100:
        st.warning('The selected model-implied level is outside the feasible 0–100% ratio range. The linear model is not constrained to feasible levels.')
    values = np.linspace(-60,60,121)
    rows = []
    for d in values:
        v = np.array([1.0,d]); mean = current_y + beta['const'] + beta['dBrent']*d
        err = z*np.sqrt(max(0.0, v @ covariance @ v))
        rows.append({'ΔBrent (USD/bbl)':d,'Lower 95%':mean-err,'Model-implied level':mean,'Upper 95%':mean+err})
    scenario = pd.DataFrame(rows)
    x_axis = alt.X('ΔBrent (USD/bbl):Q', title='Hypothetical annual Brent change (USD/bbl)')
    y_axis = alt.Y('Indicator (%):Q', title='Model-implied indicator (%)', scale=alt.Scale(zero=False))
    band = alt.Chart(scenario).mark_area(opacity=0.18, color='#5b8fcb').encode(
        x=x_axis,
        y=alt.Y('Lower 95%:Q', title='Model-implied indicator (%)'),
        y2='Upper 95%:Q',
        tooltip=['ΔBrent (USD/bbl):Q', 'Lower 95%:Q', 'Upper 95%:Q']
    )
    mean_line = alt.Chart(scenario).mark_line(color='#1f5f9c', strokeWidth=3).encode(
        x=x_axis,
        y=alt.Y('Model-implied level:Q', title='Model-implied indicator (%)'),
        tooltip=['ΔBrent (USD/bbl):Q', 'Model-implied level:Q']
    )
    selected = pd.DataFrame({
        'ΔBrent (USD/bbl)': [delta],
        'Model-implied level': [current_y + predicted_change],
    })
    marker = alt.Chart(selected).mark_rule(color='#c44747', strokeDash=[5, 4]).encode(x=x_axis)
    point = alt.Chart(selected).mark_circle(size=145, color='#c44747').encode(
        x=x_axis,
        y=alt.Y('Model-implied level:Q', title='Model-implied indicator (%)'),
        tooltip=['ΔBrent (USD/bbl):Q', 'Model-implied level:Q']
    )
    st.altair_chart((band + mean_line + marker + point).properties(height=370), use_container_width=True)
    st.caption('Blue line: model-implied level · Shaded area: 95% confidence interval for the conditional mean · Red marker: selected scenario.')
    st.info(
        f'**How to read this:** With ΔBrent = 0, the saved regression still implies a '
        f'{float(beta["const"]):+.2f} pp annual change because its intercept is not zero. '
        f'The scenario starts from the observed {last_year} indicator ({current_y:.2f}%), '
        f'so the model-implied level need not equal the observed level even when the slider is zero. '
        f'Changing Brent by {delta:+.0f} USD/bbl changes the model-implied level by '
        f'{float(beta["dBrent"]) * delta:+.2f} pp relative to the zero-change scenario.'
    )
    if lower < 0 or upper > 100:
        st.caption('Some extrapolated scenario values fall outside the feasible 0–100% indicator range. They are left visible to flag model limitations, not presented as plausible measured shares.')
    st.warning('Illustrative conditional scenario, NOT a validated future forecast or causal effect. The interval reflects coefficient uncertainty only, not a prediction interval; it excludes future shocks and data uncertainty. The latest observed indicator is held fixed as the starting level.')

elif section == 'Model Comparison & Backtesting':
    st.title('Model Comparison & Backtesting')
    st.write('Held-out years and metrics are read from the saved model artifact, not recomputed when the page loads.')
    test = art['forecast_test']; naive = test['naive']; fitted = test['difference_model_lagged_brent']
    scores = pd.DataFrame([{'Model':'Naive benchmark', 'MAE (pp)':naive['mae'], 'RMSE (pp)':naive['rmse']}, {'Model':'Difference model (lagged Brent)', 'MAE (pp)':fitted['mae'], 'RMSE (pp)':fitted['rmse']}])
    st.dataframe(scores.round(3), hide_index=True, width='stretch')
    st.metric('Model / naive RMSE ratio', f"{fitted['rmse']/naive['rmse']:.3f}")
    st.write('**RMSE winner: naive benchmark.** The regression is slightly better on MAE, but worse on RMSE. This negative result is reported rather than hidden.')
    st.caption('The holdout contains only four annual years (about 21% of the 19-year sample); both error estimates are highly sensitive to individual years.')
    st.caption('Explanatory regression: contemporaneous ΔBrent; forecasting evaluation: lagged ΔBrent to avoid using same-year information not known at forecast time.')
    st.caption('Held-out years: '+', '.join(map(str,test['test_years']))+'. These saved forecasts are a separate lagged-Brent evaluation, not necessarily the same same-year explanatory regression displayed under Drivers.')
    if BACKTEST.exists():
        forecast = pd.read_csv(BACKTEST).set_index('Year')
        st.subheader('Held-out predictions')
        chart_df = forecast.reset_index()
        forecast_long = chart_df.melt(id_vars='Year', var_name='Series', value_name='Indicator (%)')
        forecast_long['Year label'] = forecast_long['Year'].astype(int).astype(str)
        fc_chart = alt.Chart(forecast_long).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X('Year label:O', title='Year', sort=sorted(forecast_long['Year label'].unique())),
            y=alt.Y('Indicator (%):Q', title='Indicator (%)'),
            color=alt.Color('Series:N', title='Series'),
            tooltip=['Year label:N', 'Series:N', alt.Tooltip('Indicator (%):Q', format='.2f')]
        )
        st.altair_chart(fc_chart.properties(height=320), use_container_width=True)
        st.dataframe(forecast.reset_index().round(2), hide_index=True, width='stretch')
    else:
        st.warning('Forecast detail CSV missing; summary metrics from artifact remain available.')

else:
    st.title('Key Findings & Sources')
    st.subheader('What the data show')
    st.write(f'The indicator is **{current_y:.2f}% in {last_year}**, a change of **{change_y:+.2f} percentage points** from {int(previous["Year"])}.')
    st.write('The saved first-difference regression shows a positive same-year statistical association between Brent price changes and the indicator change. This does not establish causality.')
    st.write('In the saved held-out evaluation, the naive benchmark has the lower RMSE. The model should not be presented as outperforming a simple forecast.')
    st.subheader('Indicator definition')
    st.latex(r'Y_t = 100 \times \frac{\mathrm{OilTaxNF}_t}{\mathrm{StateIncome}_t}')
    st.caption('Numerator: direct taxes from oil-sector enterprises credited to the National Fund. Denominator: state-budget income. Both are expressed in thousands of KZT before taking the ratio. Unit: percent. This is not an official government indicator.')
    st.subheader('Provenance')
    st.write(f"**Indicator:** {art.get('indicator_name','Oil-tax-to-budget-income ratio')} · **Frequency:** annual · **Data window:** {first_year}–{last_year} · **Model window:** {art['sample_start']}–{art['sample_end']} · **Model estimation (UTC):** {art['estimated_at_utc']}.")
    st.write('**Primary fiscal data:** Ministry of Finance of Kazakhstan, annual National Fund formation/use reports (oil-sector direct taxes) and state-budget execution reports (state-budget income). Individual year, source document, retrieval and table/row references: `SOURCES.md`.')
    st.write('**Brent:** Federal Reserve Bank of St. Louis, FRED series [DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU), annual average in USD/bbl. **Oil production:** Bureau of National Statistics of Kazakhstan, annual industrial-production tables; oil including condensate in million tonnes.')
    st.caption('These are source descriptions transcribed from the project documentation. Source-document values and retrieval dates still require independent verification before submission.')
    st.write('**Reproduction:** `python src/ingest.py` → `python src/difference_backtest.py` → `python src/build_artifact.py` → `python -m streamlit run app.py`. The app reads committed local data and a saved model artifact; it does not fit regressions during page loading.')
    st.info('Offline mode: the app uses a committed local snapshot. No network refresh occurs on page load. “Latest” means latest observation in this snapshot, not necessarily latest published data.')
    st.subheader('What this indicator cannot tell you')
    for limitation in art.get('limitations',[]):
        st.write('• '+limitation)
    st.write('It is not the total share of oil in government finances: the numerator excludes other petroleum-linked receipts and transfers. The small annual sample limits statistical power and external validity.')
    st.subheader('Decision use')
    st.write('**Intended user:** fiscal-risk analyst. A large change in this ratio can trigger a closer review of National Fund oil-tax receipts and state-budget income before fiscal-planning discussions. The ratio is a screening signal, not a policy rule or an automatic warning threshold.')
    st.caption('The small sample (19 annual observations; 18 changes) limits statistical power. Any exception to the course minimum sample requirement should be documented with actual instructor approval, not assumed.')
