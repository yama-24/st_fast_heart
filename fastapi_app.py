# fastapi_app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import base64
import torch
from network_model import Net, transform
import numpy as np

app = FastAPI()

# 訓練済みモデル読み込み
# 事前訓練済みU-Netモデルをロード
net = Net().cpu().eval()
net.load_state_dict(torch.load('heart.pt', map_location=torch.device('cpu')))

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # 画像読み込み
    image = Image.open(BytesIO(await file.read())).convert("L")

    # 画像の前処理
    img = transform(image)
    input_batch = img.unsqueeze(0)  # バッチ次元を作成

    # 推論
    y = net(input_batch)

    # モデル出力をnumpy配列に変換
    y_numpy = y.detach().cpu().numpy()

    # 出力から余分な次元を削除
    y_numpy = y_numpy.squeeze()

    # スケーリング
    result_image = y_numpy.astype(np.uint8)

    # numpy配列からPIL Imageを作成
    result = Image.fromarray(result_image)

    # セグメンテーションの結果をPIL画像に変換
    result_bytes = BytesIO()
    result.save(result_bytes, format='PNG')
    result_bytes = result_bytes.getvalue()
    encoded_img = base64.b64encode(result_bytes).decode('utf-8')

    return JSONResponse(content={"result": encoded_img}, media_type="application/json")
