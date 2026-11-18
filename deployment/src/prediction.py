# Import libraries
import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st

# Load the model
with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

def run():
    # Membuat title
    st.title('Smartphone Price Range Classification')

    # Penjelasan class rentang harga
    st.markdown("""
    | price_range | class |
    |---|---|
    | 0 | Low price |
    | 1 | Medium price |
    | 2 | High price |
    | 3 | Very high price |
    """)

    # Membuar form
    with st.form('smartphone-price-predict'):
        name = st.text_input('Smartphone Name', value='Name')
        st.markdown('---')

        # Performa
        ram = st.number_input('RAM (MB)', min_value=256, max_value=4000, step=64, help='Enter the RAM capacity', value=2000)
        int_memory = st.number_input('Internal Memory (GB)', min_value=2, max_value=64, step=1, help='Enter the internal storage', value=32)
        battery_power = st.number_input('Battery Power (mAh)', min_value=500, max_value=2000, step=50, help='Enter the battery capacity', value=1500)
        talk_time = st.slider('Talk Time (hours)', min_value=2, max_value=20, help='Enter the maximum talk time', value=10)
        clock_speed = st.slider('Clock Speed', min_value=0.5, max_value=3.0, step=0.1, help='Enter the processor speed', value=1.5)
        n_cores = st.slider('Number of Cores', min_value=1, max_value=8, help='Enter the number of processor cores', value=4)
        st.markdown('---')

        # Kamera
        primary_camera = st.number_input('Primary Camera (MP)', min_value=0, max_value=20, step=1, help='Enter the primary camera resolution', value=12)
        front_camera = st.number_input('Front Camera (MP)', min_value=0, max_value=19, step=1, help='Enter the front camera resolution', value=5)
        st.markdown('---')

        # Layar
        pixel_width = st.number_input('Pixel Width', min_value=500, max_value=2000, step=10, help='Enter the horizontal screen resolution', value=1200)
        pixel_height = st.number_input('Pixel Height', min_value=0, max_value=1960, step=10, help='Enter the vertical screen resolution', value=800)
        screen_height = st.slider('Screen Height (cm)', min_value=5, max_value=19, help='Enter the physical screen height', value=12)
        screen_width = st.slider('Screen Width (cm)', min_value=0, max_value=18, help='Enter the physical screen width', value=6)
        st.markdown('---')

        # Fisik
        mobile_weight = st.slider('Weight (gram)', min_value=80, max_value=200, help='Enter the smartphone weight', value=140)
        mobile_depth = st.slider('Thickness (cm)', min_value=0.1, max_value=1.0, step=0.1, help='Enter the smartphone thickness', value=0.5)
        st.markdown('---')

        # Fitur ya/tidak
        bluetooth = st.radio('Bluetooth', options=['No', 'Yes'], horizontal=True, index=1)
        dual_sim = st.radio('Dual SIM', options=['No', 'Yes'], horizontal=True, index=1)
        three_g = st.radio('3G', options=['No', 'Yes'], horizontal=True, index=1)
        four_g = st.radio('4G', options=['No', 'Yes'], horizontal=True, index=1)
        touch_screen = st.radio('Touch Screen', options=['No', 'Yes'], horizontal=True, index=1)
        wifi = st.radio('WiFi', options=['No', 'Yes'], horizontal=True, index=1)
        st.markdown('---')

        submitted = st.form_submit_button('Predict')

        # Create a new data
        yes_no = {'No': 0, 'Yes': 1}
        data_inf = pd.DataFrame([{
            'battery_power': battery_power,
            'bluetooth': yes_no[bluetooth],
            'clock_speed': clock_speed,
            'dual_sim': yes_no[dual_sim],
            'front_camera': front_camera,
            'four_g': yes_no[four_g],
            'int_memory': int_memory,
            'mobile_depth': mobile_depth,
            'mobile_weight': mobile_weight,
            'n_cores': n_cores,
            'primary_camera': primary_camera,
            'pixel_height': pixel_height,
            'pixel_width': pixel_width,
            'ram': ram,
            'screen_height': screen_height,
            'screen_width': screen_width,
            'talk_time': talk_time,
            'three_g': yes_no[three_g],
            'touch_screen': yes_no[touch_screen],
            'wifi': yes_no[wifi],
        }])

        # Inferensi
        if submitted:
            
            pred = model.predict(data_inf)[0]
            price_ranges = ['Low price', 'Medium price', 'High price', 'Very high price']

            st.subheader('Prediction Result')
            st.write(f'**{name}** predicted into range:')
            st.write(f'Class {pred} - {price_ranges[pred]}')

if __name__ == '__main__':
    run()