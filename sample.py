import streamlit as st
import pandas as pd
from time import sleep, time
import time

st.write("Wemp womp!")
st.header("Dataset Sample")
df = pd.read_csv('../Datasets/CleanedKaggleDataSets.csv')
st.table(df.iloc[0:10])

progress_text = 'Opeartion in progress. Please Wait.'
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(2)
    my_bar.progress(percent_complete +1, text=progress_text)
    # my_bar.empty()
st.balloons()
st.snow()
st.button('Rerun')
