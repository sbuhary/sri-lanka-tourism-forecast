from pathlib import Path
from html import escape
import base64
import calendar
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
first_month = pd.Period(data.Year_Month.min(), freq='M')
last_month = pd.Period(year=default_target.year + 1, month=12, freq='M')
years = list(range(last_month.year, first_month.year - 1, -1))
if 'market' not in st.session_state:
    st.session_state.market = info['default_inputs']['Country']
    st.session_state.target_month = str(default_target)
if 'overrides' not in st.session_state:
    st.session_state.overrides = {}
    st.session_state.editing_lag = None


def reset_history():
    st.session_state.overrides = {}
    st.session_state.editing_lag = None
    for lag in lags:
        st.session_state.pop(f'draft_{lag}', None)


def sync_picker():
    st.session_state.pop('picker_year', None)
    st.session_state.pop('picker_month', None)


def choose_market(country):
    st.session_state.market = country
    reset_history()
    sync_picker()


def move_month(direction):
    st.session_state.target_month = str(pd.Period(st.session_state.target_month, freq='M') + direction)
    reset_history()
    sync_picker()


def choose_month():
    st.session_state.target_month = str(pd.Period(
        year=st.session_state.picker_year, month=st.session_state.picker_month, freq='M'))
    reset_history()


def begin_edit(lag):
    st.session_state.editing_lag = lag
    st.session_state.pop(f'draft_{lag}', None)


def reset_value(lag):
    overrides = dict(st.session_state.overrides)
    overrides.pop(f'lag_{lag}', None)
    st.session_state.overrides = overrides
    st.session_state.editing_lag = None



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
        current_target = pd.Period(st.session_state.target_month, freq='M')
        with st.container(key='month_navigation'):
            previous, selected, following = st.columns([1, 5, 1], gap='small')
            previous.button('‹', key='previous_month', help='Previous month', use_container_width=True,
                            disabled=current_target == first_month, on_click=move_month, args=(-1,))
            with selected.popover(current_target.strftime('%B %Y'), key='month_picker', use_container_width=True):
                with st.form('choose_month'):
                    st.selectbox('Year', years, index=years.index(current_target.year), key='picker_year')
                    st.selectbox('Month', range(1, 13), index=current_target.month - 1, key='picker_month',
                                 format_func=lambda month: calendar.month_name[month])
                    st.form_submit_button('Go to month', key='go_to_month', type='primary', on_click=choose_month)
            following.button('›', key='next_month', help='Next month', use_container_width=True,
                             disabled=current_target == last_month, on_click=move_month, args=(1,))

        country = st.session_state.market
        target = pd.Period(st.session_state.target_month, freq='M')
        published = historical_arrivals(data, country, target)
        custom = bool(st.session_state.overrides)
        counts = {**published, **st.session_state.overrides}
        complete = all(value is not None for value in counts.values())
        status = 'Incomplete inputs' if not complete else 'Edited inputs' if custom else 'SLTDA counts'
        st.html(f'<div class="history-heading"><h3>Historical inputs</h3><span class="status-badge">{status}</span></div>')
        st.caption(f'One estimate uses these four counts, the market and month. The model learned from {info["training_rows"]:,} records across {training_start}–{training_end}.')
        descriptions = {3: '3 months earlier', 2: '2 months earlier',
                        1: '1 month earlier', 12: 'Same month last year'}
        for lag in lags:
            field = f'lag_{lag}'
            value = counts[field]
            period_label = (target - lag).strftime('%b %Y')
            with st.container(key=f'history_row_{lag}'):
                if st.session_state.editing_lag == lag:
                    label, editor = st.columns([1, 1.7], gap='small')
                    label.html(f'<div class="history-label"><strong>{period_label}</strong><small>{descriptions[lag]}</small></div>')
                    with editor.form(f'edit_history_{lag}'):
                        edited_value = st.number_input(f'{period_label} arrivals', min_value=0,
                            value=value, step=1, key=f'draft_{lag}', label_visibility='collapsed')
                        save, cancel = st.columns(2, gap='small')
                        save_clicked = save.form_submit_button('Save', key=f'save_{lag}', type='primary')
                        cancel_clicked = cancel.form_submit_button('Cancel', key=f'cancel_{lag}')
                        if cancel_clicked:
                            st.session_state.editing_lag = None
                            st.rerun()
                        if save_clicked:
                            if edited_value is None:
                                st.error('Enter a known count. Missing data is not zero.')
                            else:
                                overrides = dict(st.session_state.overrides)
                                if edited_value == published[field]:
                                    overrides.pop(field, None)
                                else:
                                    overrides[field] = edited_value
                                st.session_state.overrides = overrides
                                st.session_state.editing_lag = None
                                st.rerun()
                else:
                    is_edited = field in st.session_state.overrides
                    widths = [1.45, 1, .25, .25] if is_edited else [1.45, 1, .25]
                    row = st.columns(widths, gap='small', vertical_alignment='center')
                    label, number, pencil = row[:3]
                    label.html(f'<div class="history-label"><strong>{period_label}</strong><small>{descriptions[lag]}</small></div>')
                    display = 'Not in dataset' if value is None else f'{value:,}'
                    edited = '<small>Edited</small>' if field in st.session_state.overrides else ''
                    number.html(f'<div class="history-value">{display}{edited}</div>')
                    pencil.button('', icon=':material/edit_square:', key=f'edit_{lag}', type='tertiary', help=f'Edit {period_label} count',
                                  on_click=begin_edit, args=(lag,))
                    if field in st.session_state.overrides:
                        row[3].button('', icon=':material/restart_alt:', key=f'reset_{lag}', type='tertiary', help=f'Restore {period_label} dataset count',
                                     on_click=reset_value, args=(lag,))
        if custom:
            st.button('Reset all to dataset values', key='reset_all', type='tertiary', on_click=reset_history)
        st.caption('Changing the market or month loads matching data and clears edits.')
        st.caption('One month ahead: the previous three monthly counts and the same month last year must be available.')
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
            st.info('The available dataset is missing one or more required counts. Choose another market or month, or use the pencil beside a missing count to enter a known value.')
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
                bars.append(f'<div class="bar-column{" forecast-column" if is_forecast else ""}"><span class="bar-value">{value:,.0f}</span>'
                            f'<div class="bar{" forecast" if is_forecast else ""}" style="height:{value / largest * 175:.1f}px"></div></div>')
                suffix = ' est.' if is_forecast else ' · edited' if f'lag_{lags[index]}' in st.session_state.overrides else ''
                labels.append(f'<span><strong>{period.strftime("%b")}</strong><br>{period.year}{suffix}</span>')
            legend = 'Input counts' if custom else 'SLTDA counts'
            st.html('<section class="comparison" aria-label="Historical inputs compared with the estimate">'
                    f'<div class="chart-heading"><h3>History and estimate</h3><div class="legend"><span>{legend}</span><span>Estimate</span></div></div>'
                    '<div class="bar-chart">' + ''.join(bars) + '</div><div class="bar-labels">'
                    + ''.join(labels) + '</div></section>')
            st.caption('Selected input months are shown; gaps between them are not to scale.')
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
                st.caption('History comes from the available SLTDA dataset. Comparison percentages describe estimated changes, not accuracy. Edited counts affect this estimate only. Delayed or revised data and sudden travel disruptions can affect forecasts. No prediction interval is calculated.')
                st.link_button('View SLTDA data', 'https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries')
