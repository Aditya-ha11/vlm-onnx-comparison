# 🚀 Vision-Language Captioning (PyTorch vs ONNX)

## 📌 Overview
This project compares image captioning performance using:
- PyTorch (accurate)
- ONNX (fast)

## ⚡ Results
| Backend | Time | Output |
|--------|------|--------|
| PyTorch | ~1.3 sec | Accurate caption |
| ONNX | ~0.6 sec | Faster but less accurate |

Speedup: ~2–3×

## 🧠 Key Insight
ONNX improves speed but lacks autoregressive decoding, leading to lower-quality captions.

## ▶️ Run

```bash
pip install -r requirements.txt
python app.py