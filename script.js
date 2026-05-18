// script.js
// Assumptions:
// - Model is ONNX and expects input shape [1,3,H,W] (NCHW).
// - Using midas_v21_small_256 by default (input size 256).
// - Model URL must be CORS-enabled.

let session = null;
let modelInputSize = 256; // updated for midas_v21_small_256
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];

const imageInput = document.getElementById('imageInput');
const loadModelBtn = document.getElementById('loadModelBtn');
const modelUrlInput = document.getElementById('modelUrl');
const statusDiv = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const downloadBtn = document.getElementById('downloadBtn');

const origCanvas = document.getElementById('origCanvas');
const depthCanvas = document.getElementById('depthCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');

let loadedImage = null;

imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    loadedImage = img;
    drawOriginal(img);
    runBtn.disabled = !session;
  };
  img.src = URL.createObjectURL(file);
});

loadModelBtn.addEventListener('click', async () => {
  const url = modelUrlInput.value.trim();
  if (!url) { alert('Model URL daalein'); return; }
  statusDiv.textContent = 'Loading model...';
  try {
    // use WebGL or WASM backend automatically chosen by ort
    session = await ort.InferenceSession.create(url);
    statusDiv.textContent = 'Model loaded';
    runBtn.disabled = !loadedImage;
  } catch (err) {
    console.error(err);
    statusDiv.textContent = 'Model load error: ' + err.message;
    alert('Model load failed (CORS or URL issue). See console.');
  }
});

runBtn.addEventListener('click', async () => {
  if (!session) { alert('Model not loaded'); return; }
  if (!loadedImage) { alert('Image upload karo'); return; }
  statusDiv.textContent = 'Running inference...';
  try {
    const inputTensor = preprocess(loadedImage, modelInputSize);
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    const depthData = postprocessDepth(output.data, output.dims, loadedImage.width, loadedImage.height);
    drawDepth(depthData.map, loadedImage.width, loadedImage.height);
    drawOverlay(loadedImage, depthData.map, loadedImage.width, loadedImage.height, 0.6);
    statusDiv.textContent = 'Done';
    downloadBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusDiv.textContent = 'Inference error: ' + err.message;
  }
});

downloadBtn.addEventListener('click', () => {
  const a = document.createElement('a');
  a.href = depthCanvas.toDataURL('image/png');
  a.download = 'depth.png';
  a.click();
});

// Helpers

function drawOriginal(img) {
  origCanvas.width = img.width;
  origCanvas.height = img.height;
  depthCanvas.width = img.width;
  depthCanvas.height = img.height;
  overlayCanvas.width = img.width;
  overlayCanvas.height = img.height;
  const ctx = origCanvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
}

function preprocess(image, size) {
  // resize to square (size x size) preserving aspect by letterbox or center-crop
  const tmp = document.createElement('canvas');
  tmp.width = size;
  tmp.height = size;
  const tctx = tmp.getContext('2d');
  // draw image scaled to fit
  tctx.drawImage(image, 0, 0, size, size);
  const imgData = tctx.getImageData(0, 0, size, size);
  const data = imgData.data;
  // convert to Float32 NCHW with normalization
  const float32 = new Float32Array(1 * 3 * size * size);
  // imageData is RGBA
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const r = data[i] / 255;
      const g = data[i + 1] / 255;
      const b = data[i + 2] / 255;
      // normalize
      const rn = (r - mean[0]) / std[0];
      const gn = (g - mean[1]) / std[1];
      const bn = (b - mean[2]) / std[2];
      // NCHW
      const idx = y * size + x;
      float32[idx] = rn; // channel 0
      float32[size * size + idx] = gn; // channel 1
      float32[2 * size * size + idx] = bn; // channel 2
    }
  }
  // Create ONNX tensor
  const tensor = new ort.Tensor('float32', float32, [1, 3, size, size]);
  return tensor;
}

function postprocessDepth(outputData, dims, origW, origH) {
  // Output may be [1,1,H,W] or [1,H,W]; convert to 2D and resize to original
  let outW = dims[dims.length - 1];
  let outH = dims[dims.length - 2];
  // copy into Canvas then resize
  const tmp = document.createElement('canvas');
  tmp.width = outW;
  tmp.height = outH;
  const tctx = tmp.getContext('2d');
  const img = tctx.createImageData(outW, outH);
  // find min/max
  let min = Infinity, max = -Infinity;
  const arr = new Float32Array(outW * outH);
  // outputData length may be outW*outH
  for (let i = 0; i < outW * outH; i++) {
    const v = outputData[i];
    arr[i] = v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  // normalize to 0..255
  for (let i = 0; i < outW * outH; i++) {
    const v = (arr[i] - min) / (max - min + 1e-6);
    const c = Math.max(0, Math.min(255, Math.round(v * 255)));
    img.data[i * 4 + 0] = c;
    img.data[i * 4 + 1] = c;
    img.data[i * 4 + 2] = c;
    img.data[i * 4 + 3] = 255;
  }
  tctx.putImageData(img, 0, 0);
  // scale to original size
  const final = document.createElement('canvas');
  final.width = origW;
  final.height = origH;
  const fctx = final.getContext('2d');
  fctx.imageSmoothingEnabled = true;
  fctx.drawImage(tmp, 0, 0, origW, origH);
  // get final pixel values
  const finalData = fctx.getImageData(0, 0, origW, origH).data;
  // build float map normalized 0..1
  const map = new Float32Array(origW * origH);
  for (let i = 0; i < origW * origH; i++) {
    map[i] = finalData[i * 4] / 255.0;
  }
  return { map };
}

function drawDepth(map, width, height) {
  const ctx = depthCanvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const v = map[i];
      const [r, g, b] = jetColor(v);
      img.data[i * 4 + 0] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawOverlay(img, map, width, height, alpha=0.6) {
  // draw original on overlay then blend depth colormap
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0,0,width,height);
  ctx.drawImage(img, 0, 0, width, height);
  // draw depth as imageData
  const depthCtx = document.createElement('canvas').getContext('2d');
  depthCtx.canvas.width = width;
  depthCtx.canvas.height = height;
  const imgData = depthCtx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = map[i];
    const [r,g,b] = jetColor(v);
    imgData.data[i*4+0] = r;
    imgData.data[i*4+1] = g;
    imgData.data[i*4+2] = b;
    imgData.data[i*4+3] = Math.round(255*alpha);
  }
  depthCtx.putImageData(imgData, 0, 0);
  ctx.drawImage(depthCtx.canvas, 0, 0);
}

function jetColor(v) {
  // simple jet colormap for v in [0,1]
  const fourValue = 4 * v;
  const r = Math.min(255, Math.max(0, Math.round(255 * Math.min(fourValue - 1.5, -fourValue + 4.5))));
  const g = Math.min(255, Math.max(0, Math.round(255 * Math.min(fourValue - 0.5, -fourValue + 3.5))));
  const b = Math.min(255, Math.max(0, Math.round(255 * Math.min(fourValue + 0.5, -fourValue + 2.5))));
  return [r,g,b];
}
