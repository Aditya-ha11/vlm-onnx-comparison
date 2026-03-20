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

## ⚡ Results

### PyTorch (Accurate)
![PyTorch Output](demo1.png)

### ONNX (Fast)
![ONNX Output](demo2.png)

### Comparison
![Comparison](demo3.png)

## 🧠 Key Insight
ONNX improves speed but lacks autoregressive decoding, leading to lower-quality captions.

## ▶️ Run

```bash
pip install -r requirements.txt
python app.py

## ⚠️ Note on ONNX Output

ONNX inference uses simplified greedy decoding instead of autoregressive generation,
which leads to faster inference but lower-quality captions.

This demonstrates the trade-off between performance and accuracy.

