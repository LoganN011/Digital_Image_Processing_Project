from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class ImageCaptioner:
    def __init__(self):
        model_name = "Salesforce/blip-image-captioning-base"

        self.processor = BlipProcessor.from_pretrained(model_name)

        # Use PyTorch .bin weights for compatibility with the current torch/transformers setup.
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=False
        )

        self.model.eval()

    def generate_caption(self, pil_image):
        inputs = self.processor(pil_image, return_tensors="pt")

        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)

        return self.processor.decode(out[0], skip_special_tokens=True)