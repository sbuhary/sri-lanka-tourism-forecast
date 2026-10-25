from pathlib import Path
from html import escape
import base64
import json
import joblib
import pandas as pd
import streamlit as st
from src.forecast_inputs import historical_arrivals, forecast_outlook, lags

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
if 'market' not in st.session_state:
    st.session_state.market = info['default_inputs']['Country']
if 'overrides' not in st.session_state:
    st.session_state.overrides = {}
    st.session_state.editing_lag = None


def reset_history():
    st.session_state.overrides = {}
    st.session_state.editing_lag = None
    for lag in lags:
        st.session_state.pop(f'draft_{lag}', None)


def show_forecast():
    st.session_state.scroll_to_forecast = True
    st.session_state.scroll_request = st.session_state.get('scroll_request', 0) + 1


def change_market():
    reset_history()
    show_forecast()


def choose_market(country):
    st.session_state.market = country
    change_market()


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
    <p>Estimate up to three months of tourist arrivals by origin market.</p>
    <p class="banner-source">Monthly arrivals by origin market · Source: SLTDA</p>
  </div>
</header>
''')

with st.container(border=True, key='setup_card'):
    market_controls, horizon_controls = st.columns([1.6, 1], gap='large')
    with market_controls:
        with st.container(key='market_picker'):
            st.selectbox('Origin market', info['countries'], key='market', on_change=change_market)
        with st.container(key='quick_picks'):
            picks = [('India', 'India'), ('UK', 'United Kingdom'),
                     ('Russia', 'Russian Federation'), ('Germany', 'Germany')]
            for column, (label, country) in zip(st.columns(4, gap='small'), picks):
                column.button(label, key=f'pick_{label}', width='stretch',
                              type='primary' if st.session_state.market == country else 'secondary',
                              on_click=choose_market, args=(country,))
    with horizon_controls:
        months_ahead = st.radio('Forecast horizon', [1, 2, 3], index=2, horizontal=True, key='horizon', on_change=show_forecast,
                               format_func=lambda value: f'{value} month' if value == 1 else f'{value} months')
        st.caption(f'Data through {training_end} · Forecast starts {default_target.strftime("%B %Y")}')

country = st.session_state.market
target = default_target
published = historical_arrivals(data, country, target)
custom = bool(st.session_state.overrides)
counts = {**published, **st.session_state.overrides}
complete = all(value is not None for value in counts.values())
outlook, missing = forecast_outlook(model, info['features'], data, country, target, months_ahead, st.session_state.overrides)

with st.container(border=True, key='result_card'):
    estimate_type = 'No forecast' if not outlook else 'Custom scenario' if custom else 'Forecast'
    end = target + len(outlook) - 1
    period_label = target.strftime('%B %Y') if len(outlook) <= 1 else f'{target.strftime("%B")}–{end.strftime("%B %Y")}'
    st.html(f'<div class="result-heading"><div><h2>Monthly arrivals forecast</h2>'
            f'<p>{escape(country)} · {period_label}</p></div>'
            f'<span class="estimate-badge">{estimate_type}</span></div>')
    if not complete:
        st.info('Required starting history is missing. Choose another market, or open “View or adjust starting inputs” and enter a known count.')
    else:
        for column, row in zip(st.columns(len(outlook)), outlook):
            column.metric(f'{country} · {pd.Period(row["Month"], freq="M").strftime("%B %Y")}',
                          f'{row["Estimated arrivals"]:,.0f}')
        if months_ahead > 1:
            st.caption('Later months reuse earlier forecasts, so uncertainty increases. No prediction interval is available.')
        # Compare the first estimate with published counts, even when inputs are edited.
        changes = []
        for lag in [1, 12]:
            previous = published[f'lag_{lag}']
            if previous is not None and previous > 0:
                change = 100 * (outlook[0]['Estimated arrivals'] / previous - 1)
                period = (target - lag).strftime('%b %Y')
                direction = 'above' if change >= 0 else 'below'
                changes.append(f'<span class="change">{abs(change):.1f}% {direction} {period} actual</span>')
        if changes:
            st.html(f'<p class="change-label">{target.strftime("%B")} estimate compared with published arrivals</p>'
                    '<div class="changes">' + ''.join(changes) + '</div>')

        history = data.loc[data.Country.eq(country)].set_index('Year_Month').Arrivals
        recent = [(target - lag, history.get(str(target - lag))) for lag in range(12, 0, -1)]
        comparison = recent + [(pd.Period(row['Month'], freq='M'), row['Estimated arrivals']) for row in outlook]
        largest = max(1, *(value for _, value in comparison if pd.notna(value)))
        bars, labels = [], []
        for index, (period, value) in enumerate(comparison):
            is_forecast = index >= 12
            boundary = ' forecast-column' if index == 12 else ''
            bar_class = 'bar forecast' if is_forecast else 'bar'
            value_label = f'{value:,.0f}' if pd.notna(value) else 'Missing'
            height = value / largest * 175 if pd.notna(value) else 0
            detail = f'{period.strftime("%B %Y")}: {value_label} arrivals · {"Forecast" if is_forecast else "Published"}'
            visible_value = value_label if is_forecast or index in [0, 11] else ''
            bars.append(f'<div class="bar-column{boundary}" tabindex="0" title="{detail}" aria-label="{detail}"><span class="bar-value">{visible_value}</span>'
                        f'<div class="{bar_class}" style="height:{height:.1f}px"></div></div>')
            boundary_label = ' class="forecast-label"' if index == 12 else ''
            suffix = ' est.' if is_forecast else ''
            labels.append(f'<span{boundary_label}><strong>{period.strftime("%b")}</strong><br>{period.year}{suffix}</span>')
        st.html('<section class="comparison" aria-label="Twelve consecutive historical months followed by forecasts">'
                '<div class="chart-heading"><h3>Recent arrivals and forecast</h3><div class="legend"><span>Published arrivals</span><span>Forecast</span></div></div>'
                '<div class="chart-scroll"><div class="chart-inner"><div class="bar-chart">' + ''.join(bars)
                + '</div><div class="bar-labels">' + ''.join(labels) + '</div></div></div></section>')
        st.caption('Twelve months of published history. Hover over a bar for its exact count; scroll the chart on smaller screens.')
        if custom:
            st.caption('The published history is unchanged. Forecasts reflect your edited starting inputs.')
        forecast_table = pd.DataFrame(outlook).rename(columns={'Estimated arrivals': 'Forecast arrivals'})
        forecast_table.insert(0, 'Country', country)
        forecast_table['Forecast arrivals'] = forecast_table['Forecast arrivals'].round().astype(int)
        st.download_button('Download forecast CSV', forecast_table.to_csv(index=False),
                           file_name='tourism_forecast.csv', mime='text/csv', type='tertiary')
        if missing:
            st.warning('The outlook stops because a required historical count is missing: ' + ', '.join(missing) + '.')
    st.caption('Planning estimates, not official SLTDA figures. Publication delays and sudden travel disruptions can affect results.')

with st.expander('View or adjust starting inputs', expanded=not complete):
    st.caption('These four counts start the outlook. Edits affect this scenario only; the dataset is unchanged. Changing the market clears edits.')
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
                    with st.container(horizontal=True, gap='small'):
                        save_clicked = st.form_submit_button('Save', key=f'save_{lag}', type='primary')
                        cancel_clicked = st.form_submit_button('Cancel', key=f'cancel_{lag}')
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
                pencil.button('', icon=':material/edit:', key=f'edit_{lag}', type='tertiary', help=f'Edit {period_label} count',
                              on_click=begin_edit, args=(lag,))
                if field in st.session_state.overrides:
                    row[3].button('', icon=':material/restart_alt:', key=f'reset_{lag}', type='tertiary', help=f'Restore {period_label} dataset count',
                                 on_click=reset_value, args=(lag,))
    if custom:
        st.button('Reset all to dataset values', key='reset_all', type='tertiary', on_click=reset_history)

with st.expander('About the model and evaluation'):
    st.write(f'Extra Trees averages 150 decision trees. The deployed model was trained on {info["training_rows"]:,} monthly records from {training_start} to {training_end}. It uses the market, calendar month and four historical counts. The outlook runs this model up to three times, feeding earlier estimates into later months.')
    st.write('Settings were chosen using 2023 validation. Training on 2019–2023 gave MAE 258 arrivals per market/month and R² 0.904 on 2024 validation. Random Forest had lower error. These scores describe the evaluation model. The deployed refit has no independent evaluation.')
    st.caption('A retrospective check on previously explored 2025–September 2026 data gave MAE 293 for Extra Trees and 220 for the seasonal rule. This was not an untouched test. R² is not percentage accuracy.')
    st.caption('History comes from the available SLTDA dataset. Comparison percentages describe estimated changes, not accuracy. Edited counts affect this outlook only. Delayed or revised data and sudden travel disruptions can affect forecasts. No prediction interval is calculated.')
    st.link_button('View SLTDA data', 'https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries')

# Move to the results after a selection, without moving the initial page load.
if st.session_state.pop('scroll_to_forecast', False):
    st.html(f'''<script data-request="{st.session_state.scroll_request}">
      requestAnimationFrame(() => {{
        const panel = document.querySelector('.st-key-result_card');
        if (!panel) return;
        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        panel.scrollIntoView({{behavior: reducedMotion ? 'instant' : 'smooth', block: 'start'}});
        const heading = panel.querySelector('h2');
        heading?.setAttribute('tabindex', '-1');
        heading?.focus({{preventScroll: true}});
      }});
    </script>''', unsafe_allow_javascript=True)
