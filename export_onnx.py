from transformers import BlipForConditionalGeneration
import torch

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Dummy inputs (correct shape for BLIP)
pixel_values = torch.randn(1, 3, 384, 384)
input_ids = torch.ones((1, 10), dtype=torch.long)

torch.onnx.export(
    model,
    (pixel_values, input_ids),
    "blip.onnx",
    input_names=["pixel_values", "input_ids"],
    output_names=["logits"],
    opset_version=18
)

print("ONNX model exported!")