# -------------------------------
# Qwen3 Adapter (vLLM)
# -------------------------------
import torch
from typing import List, Dict, Any, Union
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from .base import BaseVLMAdapter

class Qwen3Adapter(BaseVLMAdapter):
    def load(self):
        # Initialize vLLM engine
        llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 100, "video": 0},
            mm_processor_cache_gb=0,
            max_model_len=16000,
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
        content = []
        
        # Handle images
        image = item.get("image")
        if image is not None:
            if isinstance(image, list):
                for img in image:
                    if img is not None:
                        content.append({"type": "image", "image": img})
            else:
                content.append({"type": "image", "image": image})
        
        # Handle text
        question = item.get("question", "")
        if question and str(question).strip():
            content.append({"type": "text", "text": str(question)})
        
        # Fallback
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
        Prepare inputs for vLLM.
        """
        all_inputs = []
        for msg in messages:
            # 1. Extract Images for vLLM payload
            # AND prepare a clean message list for the Prompt Template
            images = []
            clean_messages = []
            
            for message in msg:
                role = message["role"]
                content = message["content"]
                
                clean_content = []
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            # A. Save the actual image for multi_modal_data
                            img = item.get("image")
                            if img is not None:
                                if isinstance(img, list):
                                    images.extend(img)
                                else:
                                    images.append(img)
                            
                            # B. Keep the placeholder for the prompt generator!
                            # CRITICAL FIX: Do NOT remove this. 
                            # Qwen needs {"type": "image"} to insert <|vision_start|> tokens.
                            # We just remove the heavy PIL object to prevent serialization issues.
                            clean_content.append({"type": "image"})
                        else:
                            clean_content.append(item)
                else:
                    clean_content = content

                clean_messages.append({"role": role, "content": clean_content})
            
            # 2. Generate Text Prompt (WITH Image Tokens)
            # using the processor's chat template
            prompt = processor.apply_chat_template(
                clean_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 3. Prepare multi_modal_data
            multi_modal_data = {}
            if images:
                if len(images) == 1:
                    multi_modal_data["image"] = images[0]
                else:
                    multi_modal_data["image"] = images
            
            # 4. Create Input Dict
            input_dict = {
                "prompt": prompt,
                "multi_modal_data": multi_modal_data if multi_modal_data else None,
            }
            all_inputs.append(input_dict)
        
        return all_inputs

    def infer(self, model, processor, inputs, max_new_tokens):
        """
        Run inference using vLLM generate() API.
        """
        # 1. Standardize to list
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
            # Qwen models sometimes need explicit stop tokens
            stop_token_ids=[151645, 151643] 
        )
        
        # 3. Run Inference
        # FIX: Pass the 'request_list' directly to generate.
        # DO NOT unpack it into separate prompts/multi_modal_data lists.
        # vLLM expects a list of dicts: [{"prompt": "...", "multi_modal_data": ...}, ...]
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

    def stack_inputs(self, input_list, model):
        return input_list