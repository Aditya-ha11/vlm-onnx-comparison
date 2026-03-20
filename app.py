import gradio as gr
import numpy as np
import onnxruntime as ort
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------------
# Load Models
# -------------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
pt_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

onnx_session = ort.InferenceSession("blip.onnx")


# -------------------------------
# Clean ONNX output (important)
# -------------------------------
def clean_caption(text):
    words = text.split()
    cleaned = []

    for w in words:
        if len(cleaned) == 0 or w != cleaned[-1]:
            cleaned.append(w)

    return " ".join(cleaned)


# -------------------------------
# Main Function
# -------------------------------
def caption_image(img):
    try:
        if img is None:
            return "No image provided"

        # 🔥 FIX: ensure RGB (webcam fix)
        img = img.convert("RGB")

        # =============================
        # PyTorch Inference (Accurate)
        # =============================
        start_pt = time.time()

        inputs = processor(img, return_tensors="pt")
        out = pt_model.generate(**inputs)

        pt_caption = processor.decode(out[0], skip_special_tokens=True)

        end_pt = time.time()

        # =============================
        # ONNX Inference (Fast)
        # =============================
        start_onnx = time.time()

        pixel_values = inputs["pixel_values"].numpy()

        # Dummy input_ids (required by ONNX graph)
        input_ids = np.ones((1, 10), dtype=np.int64)

        onnx_inputs = {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }

        outputs = onnx_session.run(None, onnx_inputs)
        logits = outputs[0]

        # Greedy decoding
        predicted_ids = np.argmax(logits, axis=-1)

        onnx_caption = processor.tokenizer.decode(
            predicted_ids[0],
            skip_special_tokens=True
        )

        # Clean repeated words
        onnx_caption = clean_caption(onnx_caption)

        end_onnx = time.time()

        # =============================
        # Timing + Speedup
        # =============================
        pt_time = end_pt - start_pt
        onnx_time = end_onnx - start_onnx

        speedup = round(pt_time / onnx_time, 2) if onnx_time > 0 else "N/A"

        # =============================
        # Output UI
        # =============================
        return f"""
🟢 PyTorch (Accurate):
{pt_caption}
⏱ Time: {round(pt_time, 3)} sec

🔵 ONNX (Fast):
{onnx_caption}
⏱ Time: {round(onnx_time, 3)} sec

⚡ Speedup: {speedup}× faster

⚠️ Note: ONNX uses simplified decoding (lower quality)
"""

    except Exception as e:
        return f"❌ Error: {str(e)}"


# -------------------------------
# Gradio UI
# -------------------------------
demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🚀 Vision-Language Captioning (PyTorch vs ONNX)",
    description="Compare accuracy (PyTorch) vs speed (ONNX)"
)

demo.launch()