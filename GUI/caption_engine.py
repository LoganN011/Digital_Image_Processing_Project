from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:
    def __init__(self):
        # Load the pretrained BLIP model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_caption(self, pil_image):
        """Generates a short description of the poster image."""
        inputs = self.processor(pil_image, return_tensors="pt")
        out = self.model.generate(**inputs)

        return self.processor.decode(out[0], skip_special_tokens=True)