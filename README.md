# MiDaS Depthmap Web App

This repository contains a Flask web app that generates depthmaps using the MiDaS model.

Quick local run (recommended for development):

1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

2. Install Python dependencies

The repository includes a requirements.txt for non-PyTorch packages. Install PyTorch separately because you must choose the correct wheel for CPU vs CUDA.

CPU example:

```bash
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

3. Run the app

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

Docker (quick public deployment)

1. Build the image (this installs CPU PyTorch by default):

```bash
docker build -t midas-app .
```

2. Run the container:

```bash
docker run -p 5000:5000 midas-app
```

Deploy to Render (recommended for quick public URL)

1. Push this repo to GitHub.
2. Sign in to Render (https://render.com) and create a new Web Service.
3. Connect your GitHub repo and select the branch.
4. Select Docker as the environment (the repo contains a Dockerfile). Start the service.

Notes

- The app uses `MiDaS_small` by default for CPU-friendly inference. For higher-quality depthmaps switch `model_type` in `app.py` to `DPT_Large` and deploy to a GPU-backed host.
- For public uploads, use moderation/approval workflows and set upload size limits.
