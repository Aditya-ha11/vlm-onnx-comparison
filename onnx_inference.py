import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import BlipProcessor

# Load processor (same as before)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_fast=False
)


session = ort.InferenceSession("blip.onnx")

# Load image
image = Image.open("C:/Users/Nitro/Downloads/test.jpg").convert("RGB")

# Preprocess
inputs = processor(image, return_tensors="pt")

onnx_inputs = {
    "pixel_values": inputs["pixel_values"].numpy(),
    "input_ids": np.ones((1, 10), dtype=np.int64)
}

# Run inference
import time

start = time.time()

outputs = session.run(None, onnx_inputs)

end = time.time()

print("ONNX Time:", end - start)



# 🔥 ADD FROM HERE
logits = outputs[0]

predicted_ids = np.argmax(logits, axis=-1)

caption = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print("Caption:", caption)
# 🔥 END HERE

print("ONNX output shape:", outputs[0].shape)