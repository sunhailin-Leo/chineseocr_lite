import time
import uuid

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

from config import *
from model import text_predict, crnn_handle
from apphelper.image import base64_to_PIL
from application import trainTicket, idcard

file_lock = "file.lock"
if os.path.exists(file_lock):
    os.remove(file_lock)

app = Flask(__name__)


@app.route("/ocr", methods=["POST"])
def ocr():
    if request.method == "POST":
        request_start_time = time.time()
        uid_job = uuid.uuid1().__str__()
        data = request.json

        # 模型参数
        bill_model = data.get("billModel", "")

        # 文字检测
        # text_angle = data.get('textAngle', False)

        # 只进行单行识别
        text_line = data.get("textLine", False)

        img_str = data["imgString"].encode().split(b";base64,")[-1]
        img = base64_to_PIL(img_str)
        if img is None:
            response_time = time.time() - request_start_time
            return jsonify({"res": [], "timeTake": round(response_time, 4)})
        else:
            img = np.array(img)
            h, w = img.shape[:2]

            final_result: list = []
            while time.time() - request_start_time <= TIMEOUT:
                if os.path.exists(file_lock):
                    continue
                else:
                    with open(file_lock, "w") as f:
                        f.write(uid_job)
                    if text_line:
                        # 单行识别
                        part_img = Image.fromarray(img)
                        text = crnn_handle.predict(part_img)
                        final_result = [
                            {"text": text, "name": "0", "box": [0, 0, w, 0, w, h, 0, h]}
                        ]
                        os.remove(file_lock)
                        break
                    else:
                        result = text_predict(img)
                        if bill_model == "" or bill_model == "通用OCR":
                            final_result = [
                                {
                                    "text": x["text"],
                                    "name": str(i),
                                    "box": {
                                        "cx": x["cx"],
                                        "cy": x["cy"],
                                        "w": x["w"],
                                        "h": x["h"],
                                        "angle": x["degree"],
                                    },
                                }
                                for i, x in enumerate(result)
                            ]
                        elif bill_model == "火车票":
                            train_ticket_result = trainTicket.trainTicket(result)
                            result = train_ticket_result.res
                            final_result = [
                                {"text": result[key], "name": key, "box": {}}
                                for key in result
                            ]
                        elif bill_model == "身份证":
                            id_card_result = idcard.idcard(result)
                            result = id_card_result.res
                            final_result = [
                                {"text": result[key], "name": key, "box": {}}
                                for key in result
                            ]
                        os.remove(file_lock)
                        break

            response_time = time.time() - request_start_time
            return jsonify({"res": final_result, "timeTake": round(response_time, 4)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9001, debug=True)
