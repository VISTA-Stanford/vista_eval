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
            # session_len=32768,
            session_len=50000,
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
                # Add all images from the list, filtering out None values
                for img in image:
                    if img is not None:
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
        LMDeploy supports multiple images: (text, image) for single, (text, [img1, img2, ...]) for multi.
        Experiments like all_image (3 views), axial_all_image (10 slices), etc. provide multiple images.
        """
        prompts = []
        for msg in messages:
            content = msg[0]["content"]

            # Extract text and collect ALL valid images from content
            text = None
            images = []

            for item in content:
                if item["type"] == "text":
                    text = item["text"]
                elif item["type"] == "image":
                    img = item.get("image")
                    if img is not None:
                        images.append(img)

            if text is None:
                raise ValueError("Text prompt is required in message content")

            if not images:
                # Text-only input: pass just the text string (not a tuple)
                prompts.append(text)
                continue

            # Load all images; on any failure, fall back to text-only for this item
            try:
                loaded = [load_image(img) for img in images]
            except Exception as e:
                print(f"Warning: Could not load image(s), using text-only: {e}")
                prompts.append(text)
                continue

            # LMDeploy: (text, image) for single, (text, [img1, img2, ...]) for multi
            if len(loaded) == 1:
                prompts.append((text, loaded[0]))
            else:
                prompts.append((text, loaded))

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