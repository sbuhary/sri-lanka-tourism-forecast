from pathlib import Path
import calendar
import json
import joblib
import pandas as pd
import streamlit as st

folder = Path(__file__).resolve().parent
st.set_page_config(page_title='Sri Lanka Tourist Arrivals Forecast', page_icon='✈️')
st.title('Sri Lanka Tourist Arrivals Forecast')
st.write('Estimate monthly arrivals from an origin market using its recent arrival counts.')

@st.cache_resource
def load_model():
    model = joblib.load(folder / 'deployment_model.joblib')
    info = json.loads((folder / 'model_info.json').read_text(encoding='utf-8'))
    return model, info

if not all((folder / name).exists() for name in ['deployment_model.joblib', 'model_info.json']):
    st.error('Run python train_model.py once before starting the app.')
    st.stop()

model, info = load_model()
defaults = info['default_inputs']
default_target = pd.Period(info['default_target'], freq='M')
st.caption('Monthly arrivals by origin market · Source: SLTDA')
with st.form('forecast'):
    country = st.selectbox('Origin market', info['countries'], index=info['countries'].index(defaults['Country']))
    left, right = st.columns(2)
    year = left.number_input('Year to predict', min_value=2019, max_value=2100, value=default_target.year, step=1)
    month = right.selectbox('Month to predict', range(1, 13),
                            index=default_target.month - 1, format_func=lambda number: calendar.month_name[number])
    st.subheader('Historical arrivals')
    st.caption('Prefilled for India, October 2026 using published historical counts. If you change the origin or target month, update all four counts to match.')
    left, right = st.columns(2)
    lag_1 = left.number_input('1 month earlier', min_value=0, value=int(defaults['lag_1']), step=1)
    lag_2 = right.number_input('2 months earlier', min_value=0, value=int(defaults['lag_2']), step=1)
    lag_3 = left.number_input('3 months earlier', min_value=0, value=int(defaults['lag_3']), step=1)
    lag_12 = right.number_input('Same month last year', min_value=0, value=int(defaults['lag_12']), step=1)
    submitted = st.form_submit_button('Estimate arrivals', type='primary', use_container_width=True)

counts = {'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3, 'lag_12': lag_12}
target = pd.Period(year=int(year), month=month, freq='M')
if submitted:
    inputs = pd.DataFrame([{'Country': country, **counts, 'month_number': month}])
    estimate = max(0, float(model.predict(inputs[info['features']])[0]))
    with st.container(border=True):
        st.metric(f'Estimated arrivals · {country} · {target.strftime("%B %Y")}', f'{estimate:,.0f}')
        if target <= pd.Period(info['training_period'].split(' to ')[1], freq='M'):
            st.caption('This period overlaps the training data; this estimate is not a held-out evaluation.')
        st.caption('Planning estimate only. Sudden changes in travel demand may affect the result.')


with st.expander('About the estimate'):
    st.write('A Random Forest combines predictions from 150 decision trees. It uses the origin market, calendar month and the four historical counts.')
    st.write('Random Forest was selected by comparing models trained on 2019–2023 data and validated on 2024 data. Its validation MAE was about 242 arrivals per origin/month. The selected model was then retrained on all eligible data from 2019 through September 2026 for deployment.')
    st.caption('The retrained deployment model has not been separately tested. Published counts can be delayed or revised. The year identifies the historical months; it is not a model input.')
    st.link_button('View SLTDA data', 'https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries')
