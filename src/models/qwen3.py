# -------------------------------
# Qwen3 Adapter
# -------------------------------
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base import BaseVLMAdapter

class Qwen3Adapter(BaseVLMAdapter):
    def load(self):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2",
            cache_dir=self.cache_dir
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
                
        return model, processor

    def create_template(self, item):
        """
        Return a single-sample conversation for Qwen3-VL.
        Each sample must be a list of dicts (conversation turns).
        Supports text-only, one image, or multiple images.
        """
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
        
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    # def prepare_inputs(self, messages, processor, model):
    #     """
    #     Qwen2_5-VL requires:
    #     - text prompts (strings)
    #     - images as a list of PIL Image objects
    #     """
    #     texts = processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     image_inputs, _ = process_vision_info(messages)

    #     inputs = processor(
    #         text=texts,
    #         images=image_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     ).to(model.device)

    #     return inputs

    def prepare_inputs(self, messages_batch, processor, model):
        """
        messages_batch: list of message dicts for each sample.
        Supports both image+text and text-only inputs.
        """
        texts = [
            processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
    
        # Process images for each message in the batch
        # Ensure we never pass None - use empty list for items without images
        image_inputs = []
        has_any_images = False
        for msgs in messages_batch:
            try:
                vision_info = process_vision_info(msgs)
                # vision_info is a tuple: (image_list, image_sizes) or None
                if vision_info is not None and isinstance(vision_info, (list, tuple)) and len(vision_info) > 0:
                    img_list = vision_info[0]
                    if img_list is not None and len(img_list) > 0:
                        image_inputs.append(img_list)
                        has_any_images = True
                    else:
                        image_inputs.append([])
                else:
                    image_inputs.append([])
            except (ValueError, IndexError, AttributeError, TypeError):
                # No images in this message - use empty list instead of None
                image_inputs.append([])
        
        # Only pass images parameter if at least one item has images
        # For text-only batches, don't pass images parameter at all
        if has_any_images:
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding="longest",
                return_tensors="pt"
            ).to(model.device)
        else:
            # Text-only processing - don't pass images parameter
            inputs = processor(
                text=texts,
                padding="longest",
                return_tensors="pt"
            ).to(model.device)
    
        return inputs

    def infer(self, model, processor, inputs, max_new_tokens):
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        outputs = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return outputs