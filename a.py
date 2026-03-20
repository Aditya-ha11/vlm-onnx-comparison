from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image
image = Image.open("C:/Users/Nitro/Downloads/test.jpg").convert("RGB")

# Process image
inputs = processor(image, return_tensors="pt")

# 🔥 Timing start
start = time.time()

# Generate caption
out = model.generate(**inputs)

# 🔥 Timing end
end = time.time()

# Decode
caption = processor.decode(out[0], skip_special_tokens=True)

print("Caption:", caption)
print("PyTorch Time:", end - start)