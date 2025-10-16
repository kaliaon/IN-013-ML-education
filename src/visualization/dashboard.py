import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Education Analytics Dashboard", layout="wide")

st.title("Аналитика обучения: кластеры и прогнозы")

uploaded = st.file_uploader("Загрузите датасет (CSV)", type=["csv"])
if uploaded is not None:
	df = pd.read_csv(uploaded)
	st.write("Предпросмотр", df.head())

	# Простая визуализация распределений
	for col in df.select_dtypes(include="number").columns[:4]:
		fig = px.histogram(df, x=col)
		st.plotly_chart(fig, use_container_width=True)

else:
	st.info("Загрузите CSV для визуализации.")


