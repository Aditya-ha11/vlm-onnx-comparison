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


![PyTorch Output Vs ONNX](https://github.com/Aditya-ha11/vlm-onnx-comparison/blob/main/Screenshot%202026-03-20%20131722.png)

![PyTorch Output Vs ONNX](https://github.com/Aditya-ha11/vlm-onnx-comparison/blob/main/Screenshot%202026-03-20%20131746.png)

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

