import streamlit as st
import pandas as pd
import numpy as np

st.title("👋 나의 첫 Streamlit 앱")
st.write("이것이 Streamlit의 전부입니다!")

# 슬라이더 위젯
n = st.slider("데이터 포인트 수", min_value=10, max_value=200, value=50)

# 데이터 생성 및 표시
df = pd.DataFrame(np.random.randn(n, 3), columns=["A", "B", "C"])
st.dataframe(df)
st.line_chart(df)