from pathlib import Path
from html import escape
import base64
import calendar
import json
import joblib
import pandas as pd
import streamlit as st

root = Path(__file__).resolve().parent
folder = root / 'models'
st.set_page_config(page_title='Sri Lanka Tourist Arrivals Forecast', page_icon='✈️', layout='wide')
st.html(f'<style>{(root / "assets/app.css").read_text(encoding="utf-8-sig")}</style>')


@st.cache_data
def banner_image():
    return base64.b64encode((root / 'assets/coast.jpg').read_bytes()).decode()


st.html(f'''
<header class="sea-banner">
  <img src="data:image/jpeg;base64,{banner_image()}" alt="" />
  <div class="banner-copy">
    <h1>Sri Lanka Tourist Arrivals Forecast</h1>
    <p>Estimate monthly arrivals using recent counts from each origin market.</p>
    <p class="banner-source">Monthly arrivals by origin market · Source: SLTDA</p>
  </div>
</header>
''')


@st.cache_resource
def load_model():
    model = joblib.load(folder / 'deployment_model.joblib')
    info = json.loads((folder / 'model_info.json').read_text(encoding='utf-8'))
    return model, info


if not all((folder / name).exists() for name in ['deployment_model.joblib', 'model_info.json']):
    st.error('Run python src/train_model.py once before starting the app.')
    st.stop()

model, info = load_model()
defaults = info['default_inputs']
default_target = pd.Period(info['default_target'], freq='M')
setup, result = st.columns([1, 1.2], gap='large')

with setup:
    with st.form('forecast'):
        st.subheader('Forecast setup')
        country = st.selectbox('Origin market', info['countries'], index=info['countries'].index(defaults['Country']))
        left, right = st.columns(2)
        year = left.number_input('Year to predict', min_value=2019, max_value=2100, value=default_target.year, step=1)
        month = right.selectbox('Month to predict', range(1, 13),
                               index=default_target.month - 1, format_func=lambda number: calendar.month_name[number])
        st.divider()
        st.markdown('#### Historical arrivals')
        st.caption(f'Prefilled for {defaults["Country"]}, {default_target.strftime("%B %Y")}. If you change the market or month, update all four counts to match.')
        left, right = st.columns(2)
        lag_1 = left.number_input('1 month earlier', min_value=0, value=int(defaults['lag_1']), step=1)
        lag_2 = right.number_input('2 months earlier', min_value=0, value=int(defaults['lag_2']), step=1)
        lag_3 = left.number_input('3 months earlier', min_value=0, value=int(defaults['lag_3']), step=1)
        lag_12 = right.number_input('Same month last year', min_value=0, value=int(defaults['lag_12']), step=1)
        submitted = st.form_submit_button('Estimate arrivals', type='primary', use_container_width=True)

# Keep the result tied to the last submitted inputs while the form is being edited.
if 'forecast_result' not in st.session_state:
    st.session_state.forecast_result = (dict(defaults), str(default_target))
if submitted:
    st.session_state.forecast_result = (
        {'Country': country, 'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3,
         'lag_12': lag_12, 'month_number': month}, str(pd.Period(year=int(year), month=month, freq='M')))

inputs, target_label = st.session_state.forecast_result
target = pd.Period(target_label, freq='M')
estimate = max(0, float(model.predict(pd.DataFrame([inputs])[info['features']])[0]))

with result:
    with st.container(border=True, key='result_card'):
        st.subheader('Estimated arrivals')
        st.metric(f'{inputs["Country"]} · {target.strftime("%B %Y")}', f'{estimate:,.0f}')
        st.caption('visitors')
        if target <= pd.Period(info['training_period'].split(' to ')[1], freq='M'):
            st.caption('This period overlaps the training data; this estimate is not a held-out evaluation.')

        comparison = [(target - lag, inputs[f'lag_{lag}']) for lag in [3, 2, 1, 12]]
        comparison.append((target, estimate))
        largest = max(1, *(value for _, value in comparison))
        bars, labels = [], []
        for index, (period, value) in enumerate(comparison):
            is_forecast = index == 4
            height = value / largest * 155
            bars.append(f'<div class="bar-column"><span class="bar-value">{value:,.0f}</span>'
                        f'<div class="bar{" forecast" if is_forecast else ""}" style="height:{height:.1f}px"></div></div>')
            label = escape(period.strftime('%b %Y'))
            labels.append(f'<span>{label}<br>{"<strong>Estimate</strong>" if is_forecast else "Input count"}</span>')
        st.html('<section class="comparison" aria-label="Historical inputs compared with the estimate">'
                '<h3>How it compares</h3><div class="bar-chart">' + ''.join(bars)
                + '</div><div class="bar-labels">' + ''.join(labels) + '</div></section>')
        st.html('<p class="result-note">An estimate based on the origin market, calendar month and historical counts. '
                'Sudden changes in travel demand may affect the result. No prediction interval is provided.</p>')

with st.expander('About the estimate'):
    st.write('Extra Trees averages 150 trees with random split thresholds. It uses the origin, calendar month and four historical counts.')
    training_end = pd.Period(info['training_period'].split(' to ')[1], freq='M').strftime('%B %Y')
    st.write(f'Settings were checked using 2023, then validated on 2024: MAE 258 arrivals per origin/month and R² 0.904. Random Forest had lower validation error. The deployed model was retrained through {training_end}; those scores describe the earlier model.')
    st.caption('In a later-period check, a model trained through 2024 had MAE 293, while the seasonal rule had MAE 220. That period had already been explored, so it is not an independent final test.')
    st.caption('Published counts can be delayed or revised. Year labels the period and is not a model input.')
    st.link_button('View SLTDA data', 'https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries')
