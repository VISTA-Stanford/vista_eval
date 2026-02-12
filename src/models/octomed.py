# # -------------------------------
# # OctoMed Adapter
# # -------------------------------
import torch
from typing import List, Dict, Any, Union
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoProcessor
from .base import BaseVLMAdapter, serialize_logprobs

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
            max_model_len=128000,
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

    # def prepare_inputs(self, messages, processor, model):
    #     all_inputs = []
        
    #     for msg in messages:
    #         # 1. Generate Prompt String WITH Images
    #         # We must pass the images to apply_chat_template so it can calculate 
    #         # the correct number of <|image_pad|> tokens based on resolution.
    #         prompt_text = processor.apply_chat_template(
    #             msg,
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )

    #         # 2. Extract Images for vLLM
    #         pixel_values = []
    #         for message in msg:
    #             for content_item in message["content"]:
    #                 if content_item["type"] == "image":
    #                     pixel_values.append(content_item["image"])

    #         # 3. Construct multi_modal_data
    #         multi_modal_data = {}
    #         if len(pixel_values) > 0:
    #             if len(pixel_values) == 1:
    #                 multi_modal_data["image"] = pixel_values[0]
    #             else:
    #                 multi_modal_data["image"] = pixel_values
            
    #         all_inputs.append({
    #             "prompt": prompt_text,
    #             "multi_modal_data": multi_modal_data if multi_modal_data else None
    #         })
            
    #     return all_inputs

    def infer(self, model, processor, inputs, max_new_tokens, constrained_choices=None):
        """
        Run inference using vLLM's generate API.
        When constrained_choices is provided (e.g. ["Yes", "No"] for binary tasks),
        uses vLLM structured outputs to force the model to output exactly one of them.
        """
        # Qwen2.5-VL Specific Stop Tokens
        # <|im_end|>, <|endoftext|>
        stop_token_ids = [151645, 151643]

        sampling_kwargs = {
            "temperature": 0.1,
            "max_tokens": max_new_tokens,
            "repetition_penalty": 1.1,
            "stop_token_ids": stop_token_ids,
        }
        if constrained_choices:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                choice=constrained_choices
            )
            sampling_kwargs["logprobs"] = 2  # Return logprobs for Yes/No confidence
        sampling_params = SamplingParams(**sampling_kwargs)
        
        outputs = model.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        results = []
        for output in outputs:
            if hasattr(output, 'outputs') and len(output.outputs) > 0:
                co = output.outputs[0]
                text = co.text.strip()
                cum_lp = getattr(co, 'cumulative_logprob', None)
                lp = getattr(co, 'logprobs', None)
                logprobs_str = serialize_logprobs(lp) if lp is not None else None
                results.append({
                    "text": text,
                    "cumulative_logprob": cum_lp,
                    "log_probs": logprobs_str,
                })
            else:
                results.append({"text": "", "cumulative_logprob": None, "log_probs": None})
        
        return results

    def stack_inputs(self, input_list, model):
        """
        Pass-through method for vLLM.
        Unlike HF models which require tensor stacking, vLLM takes a list of input dicts.
        """
        return input_list

    # def infer(self, model, processor, inputs, max_new_tokens):
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