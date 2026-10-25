from pathlib import Path
from html import escape
import base64
import json
import joblib
import pandas as pd
import streamlit as st
from src.forecast_inputs import historical_arrivals, lags

root = Path(__file__).resolve().parent
st.set_page_config(page_title='Sri Lanka Tourist Arrivals Forecast', page_icon='✈️', layout='wide')
st.html(f'<style>{(root / "assets/app.css").read_text(encoding="utf-8-sig")}</style>')


@st.cache_resource
def load_model():
    model = joblib.load(root / 'models/deployment_model.joblib')
    info = json.loads((root / 'models/model_info.json').read_text(encoding='utf-8'))
    return model, info


@st.cache_data
def load_history():
    return pd.read_csv(root / 'data/Tourism_Arrivals.csv')


@st.cache_data
def banner_image():
    return base64.b64encode((root / 'assets/coast.jpg').read_bytes()).decode()


model, info = load_model()
data = load_history()
default_target = pd.Period(info['default_target'], freq='M')
training_start, training_end = [pd.Period(value, freq='M').strftime('%B %Y')
                                for value in info['training_period'].split(' to ')]
month_options = pd.period_range(data.Year_Month.min(), default_target + 12, freq='M').astype(str).tolist()
if 'market' not in st.session_state:
    st.session_state.market = info['default_inputs']['Country']
    st.session_state.target_month = str(default_target)
    st.session_state.custom_history = None


def reset_history():
    st.session_state.custom_history = None
    for lag in lags:
        st.session_state.pop(f'edit_lag_{lag}', None)


def choose_market(country):
    st.session_state.market = country
    reset_history()


def move_month(direction):
    index = month_options.index(st.session_state.target_month)
    st.session_state.target_month = month_options[index + direction]
    reset_history()


st.html(f'''
<header class="sea-banner">
  <img src="data:image/jpeg;base64,{banner_image()}" alt="" />
  <div class="banner-copy">
    <h1>Sri Lanka Tourist Arrivals Forecast</h1>
    <p>Select a market and month to estimate tourist arrivals.</p>
    <p class="banner-source">Monthly arrivals by origin market · Source: SLTDA</p>
  </div>
</header>
''')

setup, result = st.columns([.92, 1.08], gap='medium')
with setup:
    with st.container(border=True, key='setup_card'):
        st.subheader('Forecast setup')
        with st.container(key='market_picker'):
            st.selectbox('Origin market', info['countries'], key='market', on_change=reset_history)
        with st.container(key='quick_picks'):
            picks = [('India', 'India'), ('UK', 'United Kingdom'),
                     ('Russia', 'Russian Federation'), ('Germany', 'Germany')]
            for column, (label, country) in zip(st.columns(4, gap='small'), picks):
                column.button(label, key=f'pick_{label}', use_container_width=True,
                              type='primary' if st.session_state.market == country else 'secondary',
                              on_click=choose_market, args=(country,))

        st.markdown('<p class="field-label">Target month</p>', unsafe_allow_html=True)
        with st.container(key='month_navigation'):
            previous, selected, following = st.columns([1, 5, 1], gap='small')
            previous.button('‹', key='previous_month', help='Previous month', use_container_width=True,
                            disabled=st.session_state.target_month == month_options[0],
                            on_click=move_month, args=(-1,))
            selected.selectbox('Target month', month_options, key='target_month',
                               format_func=lambda value: pd.Period(value, freq='M').strftime('%B %Y'),
                               label_visibility='collapsed', on_change=reset_history)
            following.button('›', key='next_month', help='Next month', use_container_width=True,
                             disabled=st.session_state.target_month == month_options[-1],
                             on_click=move_month, args=(1,))

        country = st.session_state.market
        target = pd.Period(st.session_state.target_month, freq='M')
        published = historical_arrivals(data, country, target)
        custom = st.session_state.custom_history is not None
        counts = st.session_state.custom_history if custom else published
        complete = all(value is not None for value in counts.values())
        status = 'Entered counts' if custom else ('SLTDA counts' if complete else 'Incomplete inputs')
        st.html(f'<div class="history-heading"><h3>Historical inputs</h3><span class="status-badge">{status}</span></div>')
        st.caption(f'One estimate uses these four counts, the market and month. The model learned from {info["training_rows"]:,} records across {training_start}–{training_end}.')
        descriptions = {3: '3 months earlier', 2: '2 months earlier',
                        1: '1 month earlier', 12: 'Same month last year'}
        rows = []
        for lag in lags:
            value = counts[f'lag_{lag}']
            display = 'Not in dataset' if value is None else f'{value:,}'
            rows.append(f'<div class="history-row"><div><strong>{(target - lag).strftime("%b %Y")}</strong>'
                        f'<small>{descriptions[lag]}</small></div><span>{display}</span></div>')
        st.html('<div class="history-list">' + ''.join(rows) + '</div>')

        with st.container(key='history_actions'):
            hint, edit = st.columns([1.6, 1], vertical_alignment='center')
            hint.caption('Change the market or month and these update.')
            with edit.popover('Edit values', type='tertiary', icon=':material/edit:', key='edit_values'):
                st.markdown('**Enter counts for this estimate**')
                st.caption('These values affect this estimate only. The source dataset stays unchanged.')
                with st.form('edit_history'):
                    edits = {}
                    for lag in lags:
                        edits[f'lag_{lag}'] = st.number_input(descriptions[lag], min_value=0,
                            value=counts[f'lag_{lag}'], step=1, key=f'edit_lag_{lag}')
                    if st.form_submit_button('Apply values', key='apply_values', type='primary'):
                        if any(value is None for value in edits.values()):
                            st.error('Enter all four counts before applying them.')
                        else:
                            st.session_state.custom_history = edits
                            st.rerun()
                st.button('Use dataset counts', key='use_published', on_click=reset_history)
        st.caption(f'SLTDA data available through {pd.Period(data.Year_Month.max(), freq="M").strftime("%B %Y")}.')

with result:
    with st.container(border=True, key='result_card'):
        market_label = escape(country)
        in_training_period = target <= pd.Period(info['training_period'].split(' to ')[1], freq='M')
        estimate_type = ('No estimate' if not complete else 'Custom scenario' if custom
                         else 'Historical estimate' if in_training_period else 'Forecast')
        st.html(f'<div class="result-heading"><div><h2>Estimated arrivals</h2>'
                f'<p>{market_label} · {target.strftime("%B %Y")}</p></div>'
                f'<span class="estimate-badge">{estimate_type}</span></div>')
        if not complete:
            st.info('The available dataset is missing one or more required counts. Choose another market or month, or enter known counts under “Edit values”.')
        else:
            inputs = {'Country': country, **counts, 'month_number': target.month}
            estimate = max(0, float(model.predict(pd.DataFrame([inputs])[info['features']])[0]))
            st.metric(f'{country} · {target.strftime("%B %Y")}', f'{estimate:,.0f}', label_visibility='collapsed')
            changes = []
            for lag in [12, 1]:
                previous_count = counts[f'lag_{lag}']
                if previous_count > 0:
                    change = 100 * (estimate / previous_count - 1)
                    direction = 'above' if change >= 0 else 'below'
                    style = 'increase' if change >= 0 else 'decrease'
                    period = (target - lag).strftime('%b %Y')
                    label = f'About the same as {period}' if abs(change) < .05 else f'{abs(change):.1f}% {direction} {period}'
                    changes.append(f'<span class="change {style}">{label}</span>')
            if changes:
                st.html('<p class="change-label">Estimated change from earlier counts</p><div class="changes">' + ''.join(changes) + '</div>')

            comparison = [(target - lag, counts[f'lag_{lag}']) for lag in lags] + [(target, estimate)]
            largest = max(1, *(value for _, value in comparison))
            bars, labels = [], []
            for index, (period, value) in enumerate(comparison):
                is_forecast = index == 4
                bars.append(f'<div class="bar-column"><span class="bar-value">{value:,.0f}</span>'
                            f'<div class="bar{" forecast" if is_forecast else ""}" style="height:{value / largest * 175:.1f}px"></div></div>')
                labels.append(f'<span><strong>{period.strftime("%b")}</strong><br>{period.year}{" est." if is_forecast else ""}</span>')
            legend = 'Entered counts' if custom else 'SLTDA counts'
            st.html('<section class="comparison" aria-label="Historical inputs compared with the estimate">'
                    f'<div class="chart-heading"><h3>History and estimate</h3><div class="legend"><span>{legend}</span><span>Estimate</span></div></div>'
                    '<div class="bar-chart">' + ''.join(bars) + '</div><div class="bar-labels">'
                    + ''.join(labels) + '</div></section>')
            if in_training_period:
                st.caption('The selected month is within the model’s training period. This estimate does not measure performance on unseen data.')

        with st.container(key='result_footer'):
            note, details = st.columns([1.7, 1], vertical_alignment='center')
            with note:
                st.html('<p class="result-note">An estimate based on the selected market and arrival history. Not an official figure.</p>')
            with details.popover('How this is estimated', type='tertiary', key='model_details'):
                st.write(f'Extra Trees averages 150 decision trees. The deployed model was trained on {info["training_rows"]:,} monthly records from {training_start} to {training_end}. It uses the market, calendar month and four historical counts to estimate one month’s arrivals.')
                st.write('Settings were chosen using 2023 validation. Training on 2019–2023 gave MAE 258 arrivals per market/month and R² 0.904 on 2024 validation. Random Forest had lower error. These scores describe the evaluation model. The deployed refit has no independent evaluation.')
                st.caption('A retrospective check on previously explored 2025–September 2026 data gave MAE 293 for Extra Trees and 220 for the seasonal rule. This was not an untouched test. R² is not percentage accuracy.')
                st.caption('History comes from the available SLTDA dataset. Comparison percentages describe estimated changes, not accuracy. Entered counts affect this estimate only. Delayed or revised data and sudden travel disruptions can affect forecasts. No prediction interval is calculated.')
                st.link_button('View SLTDA data', 'https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries')
