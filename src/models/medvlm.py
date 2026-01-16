# -------------------------------
# MedVLM-r1 Adapter
# -------------------------------
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base import BaseVLMAdapter

class MedVLM_Adapter(BaseVLMAdapter):
    def load(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        processor = AutoProcessor.from_pretrained(self.model_name)
        return model, processor

    def create_template(self, item):
        QUESTION_TEMPLATE = """
        {Question} 
        Your task: 
        1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
        2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
        3. No extra information or text outside of these tags.
        """
        content = []
        # Only add image if it exists
        if item.get("image") is not None:
            content.append({"type": "image", "image": item["image"]})
        content.append({"type": "text", "text": QUESTION_TEMPLATE.format(Question=item["question"])})
        
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        return conversation

    def prepare_inputs(self, messages_batch, processor, model):
        """
        messages_batch: list of message dicts for each sample.
        """
        texts = [
            processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
    
        # Process images only if they exist in the messages
        image_inputs = []
        has_images = False
        for msgs in messages_batch:
            try:
                vision_info = process_vision_info(msgs)
                if vision_info and len(vision_info) > 0 and vision_info[0] is not None:
                    image_inputs.append(vision_info[0])
                    has_images = True
                else:
                    image_inputs.append(None)
            except (ValueError, IndexError, AttributeError):
                # No images in this message
                image_inputs.append(None)
        
        # Only pass images parameter if at least one image exists
        if has_images:
            inputs = processor(
                text=texts,
                images=image_inputs,
                padding="longest",
                return_tensors="pt"
            ).to(model.device)
        else:
            # Text-only processing
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
                use_cache=True,
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