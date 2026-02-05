# # -------------------------------
# # OctoMed Adapter
# # -------------------------------
# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from .base import BaseVLMAdapter

# class OctoMedAdapter(BaseVLMAdapter):
#     def load(self):
#         model = AutoModelForVision2Seq.from_pretrained(
#             self.model_name,
#             torch_dtype=torch.bfloat16,
#             device_map=self.device,
#             attn_implementation="flash_attention_2",
#             cache_dir=self.cache_dir
#         )
#         processor = AutoProcessor.from_pretrained(self.model_name)
#         if hasattr(processor, "tokenizer"):
#             processor.tokenizer.padding_side = "left"
#             if processor.tokenizer.pad_token is None:
#                 processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
#         return model, processor

#     def create_template(self, item):
#         content = []
#         # Handle images: text only, one image, or multiple images
#         image = item.get("image")
#         if image is not None:
#             # Check if image is a list (multiple images)
#             if isinstance(image, list):
#                 # Add all images from the list, filtering out None values
#                 for img in image:
#                     if img is not None:
#                         content.append({"type": "image", "image": img})
#             else:
#                 # Single image
#                 content.append({"type": "image", "image": image})
#         # Always add the text question
#         content.append({"type": "text", "text": item["question"]})
        
#         return [
#             {
#                 "role": "system",
#                 "content": [
#                     {"type": "text", "text": "You are a helpful assistant. Please reason step-by-step, and put your final answer within \\boxed{}."}
#                 ]
#             },
#             {
#                 "role": "user",
#                 "content": content,
#             }
#         ]

#     def prepare_inputs(self, messages, processor, model):
#         """
#         messages_batch: list of message dicts for each sample.
#         """
#         inputs = processor.apply_chat_template(
#         	messages,
#         	add_generation_prompt=True,
#         	tokenize=True,
#         	return_dict=True,
#         	return_tensors="pt",
#             padding=True,
#         ).to(model.device)
    
#         return inputs

#     def infer(self, model, processor, inputs, max_new_tokens):
#         with torch.inference_mode():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=False
#             )

#         trimmed = [
#             out_ids[len(in_ids):] 
#             for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]

#         outputs = processor.batch_decode(
#             trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )

#         return outputs

import torch
from typing import List, Dict, Any, Union
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from .base import BaseVLMAdapter

class OctoMedAdapter(BaseVLMAdapter):
    def load(self):
        # Initialize vLLM engine
        llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            limit_mm_per_prompt={"image": 100, "video": 0}, 
            max_model_len=100000,
        )
        
        # Load processor for chat template formatting
        processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Configure tokenizer padding
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        
        return llm, processor

    def create_template(self, item):
        """
        Standardizes input into a list of messages.
        Images are kept as objects (PIL) in the list for now; 
        prepare_inputs will separate them later.
        """
        content = []
        
        # 1. Handle Images
        image_input = item.get("image")
        if image_input is not None:
            if isinstance(image_input, list):
                # Multiple images
                for img in image_input:
                    if img is not None:
                        content.append({"type": "image", "image": img})
            else:
                # Single image
                content.append({"type": "image", "image": image_input})
        
        # 2. Handle Text
        question = item.get("question", "")
        if question and str(question).strip():
            content.append({"type": "text", "text": str(question)})
        
        # 3. Fallback for empty content
        if not content:
            content.append({"type": "text", "text": "Please analyze the provided information."})
        
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def prepare_inputs(self, messages, processor, model):
        """
        Separates Images and Text for vLLM consumption.
        Returns a list of input dictionaries containing 'prompt' and 'multi_modal_data'.
        """
        all_inputs = []
        
        for msg in messages:
            # Each 'msg' is a conversation (list of dicts)
            pixel_values = []
            
            # 1. Extract images and clean messages for the processor
            # We create a message structure for the chat template that preserves
            # {"type": "image"} placeholders but removes actual PIL objects
            clean_messages = []
            
            for message in msg:
                role = message["role"]
                content = message["content"]
                
                clean_content = []
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            # Extract the actual image object for vLLM
                            img = item.get("image")
                            if img is not None:
                                pixel_values.append(img)
                            
                            # Keep placeholder for template formatting
                            clean_content.append({"type": "image"}) 
                        else:
                            clean_content.append(item)
                else:
                    # String content (legacy format)
                    clean_content = content

                clean_messages.append({"role": role, "content": clean_content})

            # 2. Apply Chat Template to get the Text Prompt
            # This inserts the necessary system prompts and <image> tokens
            prompt_text = processor.apply_chat_template(
                clean_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 3. Construct multi_modal_data
            # vLLM expects: None (0 images), PIL Image (1 image), or List[PIL Image] (>1 images)
            multi_modal_data = {}
            if len(pixel_values) > 0:
                if len(pixel_values) == 1:
                    multi_modal_data["image"] = pixel_values[0]
                else:
                    multi_modal_data["image"] = pixel_values
            
            # 4. Final Input Dict
            input_dict = {
                "prompt": prompt_text,
                "multi_modal_data": multi_modal_data if multi_modal_data else None
            }
            
            all_inputs.append(input_dict)
            
        return all_inputs

    def stack_inputs(self, input_list, model):
        """
        Pass-through method for vLLM.
        Unlike HF models which require tensor stacking, vLLM takes a list of input dicts.
        """
        return input_list

    def infer(self, model, processor, inputs, max_new_tokens):
        """
        Run inference using vLLM's generate API.
        """
        # 1. Standardize inputs to list
        if isinstance(inputs, dict):
            request_list = [inputs]
        elif isinstance(inputs, list):
            request_list = inputs
        else:
            raise TypeError(f"Expected inputs to be dict or list, got {type(inputs)}")

        # 2. Sampling Params
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        
        # 3. Generate
        # request_list is [{"prompt": "...", "multi_modal_data": {...}}, ...]
        outputs = model.generate(
            request_list,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        # 4. Extract Text
        results = []
        for output in outputs:
            if hasattr(output, 'outputs') and len(output.outputs) > 0:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("")
        
        return results