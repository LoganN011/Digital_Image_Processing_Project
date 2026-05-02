import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(
            model_name,
            use_fast=False,
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=False,
        ).eval()

    def generate_caption(self, pil_image):
        inputs = self.processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_length=50)
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
