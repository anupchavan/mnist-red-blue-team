"""Small HTTP API: POST a digit image to /predict and get the predicted class back."""
import io

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

from common import load_model

app = Flask(__name__)
model = load_model("advtrained")


def to_tensor(data):
    img = Image.open(io.BytesIO(data)).convert("L").resize((28, 28))
    x = np.asarray(img, dtype=np.float32)[None, None] / 255.0
    assert x.shape == (1, 1, 28, 28)
    return torch.from_numpy(x)


@app.route("/predict", methods=["POST"])
def predict():
    x = to_tensor(request.files["image"].read())
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
    return jsonify({"digit": int(probs.argmax()), "probabilities": probs.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
