# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import kagglehub
import os
from kagglehub import KaggleDatasetAdapter
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run():
    # Membuat title
    st.title('Smartphone Price Range Classification')

    st.write('Author: Chesta')

    # Memasukan gambar
    image = Image.open(os.path.join(BASE_DIR, 'img_assets', 'phone.webp'))
    st.image(image, caption='Smartphone, source: *Wired Italia.*')

    # Membuat sub-header
    st.subheader('This page contains Exploratory Data Analysis (EDA) of the dataset')
    st.markdown('---')

    # Memasukan dataset
    st.write('Feature data:')
    file_path = "train.csv"
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "iabhishekofficial/mobile-price-classification", file_path,)

    # Rename kolom dataset
    columns = {
    'blue': 'bluetooth',
    'fc': 'front_camera',
    'int_memory': 'internal_memory',
    'm_dep': 'mobile_depth',
    'mobile_wt': 'mobile_weight',
    'pc': 'primary_camera',
    'px_height': 'pixel_height',
    'px_width': 'pixel_width',
    'sc_h': 'screen_height',
    'sc_w': 'screen_width'
    }
    df.rename(columns=columns, inplace=True)

    # Tampilkan dataset
    st.dataframe(df)

    # Penjelasan data
    st.write('Feature data explanation:')
    st.markdown("""
    | Column Name | Description |
    |---|---|
    | `battery_power` | Battery capacity (mAh) |
    | `bluetooth` | Whether the phone supports Bluetooth. 0 = No / 1 = Yes |
    | `clock_speed` | Microprocessor clock speed |
    | `dual_sim` | Whether the phone supports dual SIM. 0 = No / 1 = Yes |
    | `front_camera` | Front camera resolution (MP) |
    | `four_g` | Whether the phone supports 4G. 0 = No / 1 = Yes |
    | `int_memory` | Internal storage capacity (GB) |
    | `mobile_depth` | Phone thickness (cm) |
    | `mobile_weight` | Phone weight (grams) |
    | `n_cores` | Number of processor cores |
    | `primary_camera` | Primary camera resolution (MP) |
    | `pixel_height` | Vertical screen resolution (pixels) |
    | `pixel_width` | Horizontal screen resolution (pixels) |
    | `ram` | RAM capacity (MB) |
    | `screen_height` | Physical screen height (cm) |
    | `screen_width` | Physical screen width (cm) |
    | `talk_time` | Maximum battery life during continuous usage (hours) |
    | `three_g` | Whether the phone supports 3G. 0 = No / 1 = Yes |
    | `touch_screen` | Whether the phone has a touchscreen. 0 = No / 1 = Yes |
    | `wifi` | Whether the phone supports Wi-Fi. 0 = No / 1 = Yes |
    | `price_range` | Smartphone price category (0–3) |
    """)

    # Memasukan gambar
    image = Image.open(os.path.join(BASE_DIR, 'img_assets', 'phone-2.jpg'))
    st.image(image, caption='Nothing Phone 2, source: *GSMArena*')

    # Distribusi data 'price_range'
    st.write("Class Price Range Distribution")
    fig = plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=df['price_range'])
    st.pyplot(fig)

    # Boxplot
    df_eda = df.copy()
    df_eda["pixel_total"] = df["pixel_width"] * df["pixel_height"]
    df_eda["ram_per_core"] = df["ram"] / df["n_cores"]

    st.write("Box Plot Data")
    fig = plt.figure(figsize=(5, 4))
    option = st.selectbox('Select a column to plot', ('ram', 'battery_power', 'pixel_total', 'ram_per_core'))
    sns.boxplot(data=df_eda, x=df['price_range'], y=option)
    st.pyplot(fig)

    # Scatter plot
    variable = [
    ('ram', 'battery_power'),
    ('ram', 'price_range'),
    ('pixel_width', 'pixel_height'),
    ('primary_camera', 'front_camera'),
    ('screen_height', 'screen_width'),
    ]

    st.write("Scatter Plot Data")
    fig = fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Membuat scatter plot menggunakan perulangan
    for ax, (x_col, y_col) in zip(axes.flat, variable):
        x = df[x_col].values
        y = df[y_col].values

        ax.scatter(x, y, alpha=0.5)

        ax.set_title(f'{x_col} vs {y_col}')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.3)

    # Menonaktifkan subplot yang tidak digunakan
    for ax in axes.flat[len(variable):]:
        ax.axis("off")
    
    st.pyplot(fig)

if __name__ == '__main__':
    run()