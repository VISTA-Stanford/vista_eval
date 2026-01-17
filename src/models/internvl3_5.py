import torch
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
from huggingface_hub import snapshot_download
from .base import BaseVLMAdapter

class InternVL35Adapter(BaseVLMAdapter):
    def load(self):
        snapshot_download(
            repo_id=self.model_name,
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False
        )

        engine_cfg = PytorchEngineConfig(
            session_len=32768,
            tp=1,
            dtype="bfloat16"
        )

        pipe = pipeline(
            self.cache_dir, 
            backend_config=engine_cfg,
            device=self.device
        )

        return pipe, None

    def create_template(self, item):
        content = []
        # Handle images: text only, one image, or multiple images
        image = item.get("image")
        if image is not None:
            # Check if image is a list (multiple images)
            if isinstance(image, list):
                # Add all images from the list
                for img in image:
                    content.append({"type": "image", "image": img})
            else:
                # Single image
                content.append({"type": "image", "image": image})
        # Always add the text question
        content.append({"type": "text", "text": item["question"]})
        
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        return conversation

    def prepare_inputs(self, messages, processor, model):
        """
        Prepare inputs for LMDeploy pipeline.
        Supports both image+text and text-only inputs.
        For text-only, we pass just the text string instead of a tuple.
        """
        prompts = []
        for msg in messages:
            content = msg[0]["content"]
            
            # Extract text (should always be present)
            text = None
            image = None
            
            # Find text and image in content
            for item in content:
                if item["type"] == "text":
                    text = item["text"]
                elif item["type"] == "image":
                    image = item["image"]
            
            if text is None:
                raise ValueError("Text prompt is required in message content")
            
            # Load image if present, otherwise pass only text for text-only
            if image is not None:
                try:
                    loaded_image = load_image(image)
                    prompts.append((text, loaded_image))
                except Exception as e:
                    print(f"Warning: Could not load image, using text-only: {e}")
                    # For text-only, pass just the text string
                    prompts.append(text)
            else:
                # Text-only input: pass just the text string (not a tuple)
                prompts.append(text)

        return prompts

    def infer(self, pipe, processor, inputs, max_new_tokens):
        """
        LMDeploy pipeline returns objects with a `.text` field.
        """
        with torch.inference_mode():
            outputs = pipe(
                inputs,
                max_new_tokens=max_new_tokens
            )

        decoded = [o.text for o in outputs]
        return decoded