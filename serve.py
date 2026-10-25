"""Small HTTP API: POST a digit image to /predict and get the predicted class back."""
import hashlib
import io
import logging
import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

from common import load_model

Image.MAX_IMAGE_PIXELS = 1_000_000  # refuse decompression bombs before decoding

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 << 20  # 1 MB per request
logging.basicConfig(filename=os.environ.get("MNIST_API_LOG", "predictions.log"), level=logging.INFO,
                    format="%(asctime)s %(message)s")
model = load_model(os.environ.get("MNIST_MODEL", "advtrained"))


class BadInput(ValueError):
    pass


def to_tensor(data):
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ("PNG", "JPEG"):
            raise BadInput("only PNG or JPEG images are accepted")
        img = img.convert("L").resize((28, 28))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise BadInput("not a readable image") from e
    x = np.asarray(img, dtype=np.float32)[None, None] / 255.0
    return torch.from_numpy(x)


@app.route("/predict", methods=["POST"])
def predict():
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "send the image in a form field called 'image'"}), 400
    data = upload.read()
    try:
        x = to_tensor(data)
    except BadInput as e:
        return jsonify({"error": str(e)}), 400
    with torch.no_grad():
        digit = int(model(x).argmax(1))
    # Only the label leaves the server: full probability vectors make model theft and inversion cheaper.
    logging.info("client=%s sha256=%s digit=%d", request.remote_addr, hashlib.sha256(data).hexdigest(), digit)
    return jsonify({"digit": digit})


if __name__ == "__main__":
    app.run(host=os.environ.get("MNIST_API_HOST", "127.0.0.1"), port=5000)
