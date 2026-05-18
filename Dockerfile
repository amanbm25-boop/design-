FROM python:3.10-slim

# System deps for OpenCV and general usage
RUN apt-get update && \
    apt-get install -y build-essential libgl1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
# Install CPU PyTorch wheel; change if you need CUDA-enabled PyTorch
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install pure-Python requirements
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
