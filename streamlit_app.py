# streamlit_app.py

import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

# FastAPIエンドポイントのURL
FASTAPI_URL = "http://localhost:8000/segment"

st.title('Heart MRI Semantic Segmentation')

uploaded_file = st.file_uploader("Choose an MRI image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width="auto")
    
    # ユーザーがアップロードした画像をAPIに送信
    if st.button('Segment Image'):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(FASTAPI_URL, files=files)
        
        if response.status_code == 200:
            # セグメンテーションの結果を受け取り表示
            result_data = response.json()
            result_bytes = base64.b64decode(result_data['result'])
            result_image = Image.open(BytesIO(result_bytes))
            st.image(result_image, caption='Segmented Image', use_column_width="auto")
        else:
            st.error('Error during segmentation')
