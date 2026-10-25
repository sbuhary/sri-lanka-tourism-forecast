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
    <p>Choose a market and a month. Recent arrivals fill in for you.</p>
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

        st.markdown('<p class="field-label">Month to forecast</p>', unsafe_allow_html=True)
        with st.container(key='month_navigation'):
            previous, selected, following = st.columns([1, 5, 1], gap='small')
            previous.button('‹', key='previous_month', help='Previous month', use_container_width=True,
                            disabled=st.session_state.target_month == month_options[0],
                            on_click=move_month, args=(-1,))
            selected.selectbox('Month to forecast', month_options, key='target_month',
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
        status = 'Custom inputs' if custom else ('Auto-filled from SLTDA' if complete else 'Missing history')
        st.html(f'<div class="history-heading"><h3>Arrivals used</h3><span class="status-badge">{status}</span></div>')
        descriptions = {3: '3 months earlier', 2: '2 months earlier',
                        1: '1 month earlier', 12: 'Same month last year'}
        rows = []
        for lag in lags:
            value = counts[f'lag_{lag}']
            display = 'Not available' if value is None else f'{value:,}'
            rows.append(f'<div class="history-row"><div><strong>{(target - lag).strftime("%b %Y")}</strong>'
                        f'<small>{descriptions[lag]}</small></div><span>{display}</span></div>')
        st.html('<div class="history-list">' + ''.join(rows) + '</div>')

        with st.container(key='history_actions'):
            hint, edit = st.columns([1.6, 1], vertical_alignment='center')
            hint.caption('Change the market or month and these update.')
            with edit.popover('Edit values', type='tertiary', icon=':material/edit:', key='edit_values'):
                st.markdown('**Edit historical arrivals**')
                st.caption('Custom counts apply only to the selected market and month.')
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
                st.button('Use published values', key='use_published', on_click=reset_history)
        st.caption(f'Included SLTDA data through {pd.Period(data.Year_Month.max(), freq="M").strftime("%B %Y")}.')

with result:
    with st.container(border=True, key='result_card'):
        market_label = escape(country)
        st.html(f'<div class="result-heading"><div><h2>Estimated arrivals</h2>'
                f'<p>{market_label} · {target.strftime("%B %Y")}</p></div>'
                '<span class="estimate-badge">Estimate</span></div>')
        if not complete:
            st.info('Published history is incomplete for this market and month. Choose another month or add the missing counts under “Edit values”.')
        else:
            inputs = {'Country': country, **counts, 'month_number': target.month}
            estimate = max(0, float(model.predict(pd.DataFrame([inputs])[info['features']])[0]))
            st.metric(f'{country} · {target.strftime("%B %Y")}', f'{estimate:,.0f}', label_visibility='collapsed')
            changes = []
            for lag in [12, 1]:
                previous_count = counts[f'lag_{lag}']
                if previous_count > 0:
                    change = 100 * (estimate / previous_count - 1)
                    direction = 'Up' if change >= 0 else 'Down'
                    style = 'increase' if change >= 0 else 'decrease'
                    changes.append(f'<span class="change {style}">{direction} {abs(change):.1f}% vs {(target - lag).strftime("%b %Y")}</span>')
            if changes:
                st.html('<div class="changes">' + ''.join(changes) + '</div>')

            comparison = [(target - lag, counts[f'lag_{lag}']) for lag in lags] + [(target, estimate)]
            largest = max(1, *(value for _, value in comparison))
            bars, labels = [], []
            for index, (period, value) in enumerate(comparison):
                is_forecast = index == 4
                bars.append(f'<div class="bar-column"><span class="bar-value">{value:,.0f}</span>'
                            f'<div class="bar{" forecast" if is_forecast else ""}" style="height:{value / largest * 175:.1f}px"></div></div>')
                labels.append(f'<span><strong>{period.strftime("%b")}</strong><br>{period.year}{" est." if is_forecast else ""}</span>')
            legend = 'Custom inputs' if custom else 'Published'
            st.html('<section class="comparison" aria-label="Historical inputs compared with the estimate">'
                    f'<div class="chart-heading"><h3>How it compares</h3><div class="legend"><span>{legend}</span><span>Estimate</span></div></div>'
                    '<div class="bar-chart">' + ''.join(bars) + '</div><div class="bar-labels">'
                    + ''.join(labels) + '</div></section>')
            if target <= pd.Period(info['training_period'].split(' to ')[1], freq='M'):
                st.caption('This month overlaps the training data; this is not a held-out evaluation.')

        with st.container(key='result_footer'):
            note, details = st.columns([1.7, 1], vertical_alignment='center')
            with note:
                st.html('<p class="result-note">An estimate based on the selected market and arrival history. Not an official figure.</p>')
            with details.popover('How this is estimated', type='tertiary', key='model_details'):
                st.write('Extra Trees averages 150 trees with random split thresholds. It uses the origin, calendar month and four historical counts.')
                training_end = pd.Period(info['training_period'].split(' to ')[1], freq='M').strftime('%B %Y')
                st.write(f'Settings were checked using 2023, then validated on 2024: MAE 258 arrivals per origin/month and R² 0.904. Random Forest had lower validation error. The deployed model was retrained through {training_end}; those scores describe the earlier model.')
                st.caption('In the previously explored later period, Extra Trees had MAE 293 and the seasonal rule had MAE 220. It was not an independent final test.')
                st.caption('Published counts can be delayed or revised. No prediction interval is provided. Year labels the period and is not a model input.')
                st.link_button('View SLTDA data', 'https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries')
