from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
import torch
import cv2
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'output')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MiDaS model (will download on first run)
model_type = "DPT_Large"  # options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
print(f"Loading MiDaS model ({model_type}) to device: {device}")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    depthmap_filename = None

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            # Read image with OpenCV and convert to RGB
            img = cv2.imread(image_path)
            if img is None:
                flash('Invalid image')
                return redirect(request.url)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Transform and prepare batch
            input_batch = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # Normalize to 0-255 and save as 8-bit PNG
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map = depth_map.astype(np.uint8)

            depthmap_filename = f"depthmap_{timestamp}.png"
            depthmap_path = os.path.join(OUTPUT_FOLDER, depthmap_filename)
            Image.fromarray(depth_map).save(depthmap_path)

            return render_template('index.html', depthmap_filename=depthmap_filename)

        else:
            flash('Allowed file types: png, jpg, jpeg')
            return redirect(request.url)

    return render_template('index.html', depthmap_filename=None)


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
