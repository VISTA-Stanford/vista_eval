# # -------------------------------
# # GEMMA 3 Adapter
# # -------------------------------
# import torch
# from transformers import Gemma3ForConditionalGeneration, AutoProcessor
# from .base import BaseVLMAdapter

# class Gemma3Adapter(BaseVLMAdapter):
#     def load(self):
#         model = Gemma3ForConditionalGeneration.from_pretrained(
#             self.model_name,
#             device_map=self.device,
#             cache_dir=self.cache_dir
#         ).eval()
#         processor = AutoProcessor.from_pretrained(self.model_name)

#         if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
#             processor.tokenizer.padding_side = "left"
            
#         return model, processor

#     def prepare_inputs(self, messages, processor, model):
#         # messages is list[list[dict]] from build_messages()
#         inputs = processor.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt"
#         ).to(model.device, dtype=torch.bfloat16)
#         return inputs

#     def infer(self, model, processor, inputs, max_new_tokens):
#         # Calculate actual input length for each item in the batch using attention mask
#         # Since padding is on the left, we count the number of 1s in the attention mask
#         if "attention_mask" in inputs:
#             input_lens = inputs["attention_mask"].sum(dim=-1).cpu().tolist()
#         else:
#             # Fallback: use the full sequence length for all items
#             input_lens = [inputs["input_ids"].shape[-1]] * inputs["input_ids"].shape[0]
        
#         # Get tokenizer settings to prevent generating pad tokens
#         pad_token_id = getattr(model.config, "pad_token_id", None) if hasattr(model, "config") else None
#         eos_token_id = getattr(model.config, "eos_token_id", None) if hasattr(model, "config") else None
        
#         generate_kwargs = {
#             **inputs,
#             "max_new_tokens": max_new_tokens,
#             "do_sample": False
#         }
        
#         # Set pad_token_id and eos_token_id if available
#         if pad_token_id is not None:
#             generate_kwargs["pad_token_id"] = pad_token_id
#         if eos_token_id is not None:
#             generate_kwargs["eos_token_id"] = eos_token_id
        
#         with torch.inference_mode():
#             generation = model.generate(**generate_kwargs)
            
#         # Get pad_token_id for filtering
#         pad_token_id = getattr(model.config, "pad_token_id", None) if hasattr(model, "config") else None
        
#         outputs = []
#         for g, input_len in zip(generation, input_lens):
#             # Trim to only generated tokens (skip input)
#             g_trimmed = g[input_len:]
            
#             # Filter out pad tokens if pad_token_id is set
#             if pad_token_id is not None:
#                 g_trimmed = g_trimmed[g_trimmed != pad_token_id]
            
#             # Decode with proper settings to avoid pad tokens
#             decoded = processor.decode(g_trimmed, skip_special_tokens=True)
#             outputs.append(decoded)
#         return outputs

#     def stack_inputs(self, input_list, model):    
#         # Get pad_token_id from model config (most reliable source)
#         pad_token_id = getattr(model.config, "pad_token_id", None) if hasattr(model, "config") else None
#         if pad_token_id is None:
#             pad_token_id = 0  # Fallback
        
#         # 1. Stack Text (Standard)
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             [inp["input_ids"].squeeze(0) for inp in input_list],
#             batch_first=True, padding_value=pad_token_id
#         )
#         attn_mask = torch.nn.utils.rnn.pad_sequence(
#             [inp["attention_mask"].squeeze(0) for inp in input_list],
#             batch_first=True, padding_value=0
#         )

#         batch = {
#             "input_ids": input_ids.to(model.device),
#             "attention_mask": attn_mask.to(model.device),
#         }

#         # 2. Conditional Stacking for Vision Data
#         # Only stack if pixel_values exists in the FIRST item of the batch
#         if "pixel_values" in input_list[0]:
#             pixel_values = torch.stack([inp["pixel_values"].squeeze(0) for inp in input_list], dim=0)
#             batch["pixel_values"] = pixel_values.to(model.device, dtype=torch.bfloat16)
        
#         # Repeat for image_sizes or pixel_attention_mask if Gemma 3 processor provides them
#         if "image_sizes" in input_list[0]:
#             batch["image_sizes"] = torch.stack([inp["image_sizes"].squeeze(0) for inp in input_list], dim=0).to(model.device)

#         return batch

import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from .base import BaseVLMAdapter

class Gemma3Adapter(BaseVLMAdapter):
    def load(self):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=self.device,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16, # Use bfloat16 for Gemma 3
            attn_implementation="flash_attention_2",
        ).eval()
        
        processor = AutoProcessor.from_pretrained(self.model_name)

        # Force left padding for generation consistency
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
            
        return model, processor

    def create_template(self, item):
        """
        Create message template following Gemma3 format.
        Supports multiple images as shown in HuggingFace docs:
        - Images can be PIL Images or base64-encoded data URIs (data:image/jpeg;base64,...)
        - Format: instruction text, then all images in sequence, then query text
        - No slice labels between images (as per user preference)
        """
        content = []
        # Handle images: text only, one image, or multiple images
        image = item.get("image")
        if image is not None:
            # Check if image is a list (multiple images)
            if isinstance(image, list):
                # Add all images from the list in sequence (no labels between them)
                # Images can be PIL Images or base64-encoded data URIs
                for img in image:
                    if img is not None:
                        content.append({"type": "image", "image": img})
            else:
                # Single image
                content.append({"type": "image", "image": image})
        # Always add the text question at the end
        content.append({"type": "text", "text": item["question"]})
        
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": content,
            }
        ]

    def prepare_inputs(self, messages, processor, model):
        # messages is list[list[dict]] from build_messages()
        # Workaround for Gemma3 processor bug with multiple images in apply_chat_template
        # Process each message individually and batch manually to avoid unpacking errors
        
        # Process each message separately to handle multiple images correctly
        all_inputs = []
        for msg in messages:
            try:
                # Try standard processing first
                single_input = processor.apply_chat_template(
                    [msg],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                all_inputs.append(single_input)
            except (ValueError, TypeError) as e:
                error_msg = str(e)
                # If we get the unpacking error, process images separately
                if "too many values to unpack" in error_msg or ("expected" in error_msg.lower() and "unpack" in error_msg.lower()):
                    # Extract images and text from the message
                    images = []
                    text_only_msg = []
                    for msg_dict in msg:
                        if msg_dict.get("role") == "user" and isinstance(msg_dict.get("content"), list):
                            text_items = []
                            for item in msg_dict["content"]:
                                if item.get("type") == "image":
                                    images.append(item["image"])
                                elif item.get("type") == "text":
                                    text_items.append(item)
                            text_only_msg.append({**msg_dict, "content": text_items})
                        else:
                            text_only_msg.append(msg_dict)
                    
                    # Process text first
                    text_inputs = processor.apply_chat_template(
                        [text_only_msg],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    
                    # Process images using the image processor
                    # The processor can handle both PIL Images and base64-encoded data URIs
                    if images:
                        image_processor = getattr(processor, 'image_processor', processor)
                        # Process all images for this message
                        # Images can be PIL Images or base64 strings (data:image/jpeg;base64,...)
                        image_inputs = image_processor(images, return_tensors="pt")
                        
                        # Combine text and image inputs
                        combined_inputs = {
                            **text_inputs,
                            "pixel_values": image_inputs["pixel_values"],
                        }
                        if "image_sizes" in image_inputs:
                            combined_inputs["image_sizes"] = image_inputs["image_sizes"]
                        all_inputs.append(combined_inputs)
                    else:
                        all_inputs.append(text_inputs)
                else:
                    # Re-raise if it's a different error
                    raise
        
        # If we have multiple inputs, we need to batch them
        # This will be handled by stack_inputs in the calling code
        if len(all_inputs) == 1:
            inputs = all_inputs[0]
            # Move to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            else:
                inputs = inputs.to(model.device)
        else:
            # Return list of inputs - stack_inputs will handle batching
            # But we still need to move each to device
            for i, inp in enumerate(all_inputs):
                if isinstance(inp, dict):
                    all_inputs[i] = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inp.items()}
                else:
                    all_inputs[i] = inp.to(model.device)
            inputs = all_inputs
        
        return inputs

    def infer(self, model, processor, inputs, max_new_tokens):
        # Standardization: Generated tokens always start after the full input width
        input_width = inputs["input_ids"].shape[-1]
        
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        with torch.inference_mode():
            generation = model.generate(**generate_kwargs)
        
        outputs = []
        for g in generation:
            # Slice from the end of the total input tensor width
            g_trimmed = g[input_width:]
            
            # Decode while skipping special tokens (handles EOS and Pad)
            decoded = processor.decode(g_trimmed, skip_special_tokens=True).strip()
            outputs.append(decoded)
        return outputs

    def stack_inputs(self, input_list, model):    
        """
        Manually implements Left Padding to ensure compatibility with 
        Gemma 3 generation logic.
        """
        pad_token_id = getattr(model.config, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0

        # Find the maximum sequence length in the batch
        max_len = max([inp["input_ids"].shape[-1] for inp in input_list])
        
        all_input_ids = []
        all_masks = []
        
        for inp in input_list:
            ids = inp["input_ids"].squeeze(0)
            mask = inp["attention_mask"].squeeze(0)
            
            # Calculate how much padding is needed
            pad_len = max_len - ids.shape[-1]
            
            # Create Left-Padded Tensors: [Pad, Pad, ..., Prompt]
            # Padding value for attn_mask is always 0
            new_ids = torch.cat([
                torch.full((pad_len,), pad_token_id, device=ids.device, dtype=ids.dtype), 
                ids
            ])
            new_mask = torch.cat([
                torch.zeros(pad_len, device=mask.device, dtype=mask.dtype), 
                mask
            ])
            
            all_input_ids.append(new_ids)
            all_masks.append(new_mask)

        # Standard Batch Dictionary
        batch = {
            "input_ids": torch.stack(all_input_ids).to(model.device),
            "attention_mask": torch.stack(all_masks).to(model.device),
        }

        # 2. Multimodal Data Stacking (if present)
        # Check if ALL items have pixel_values before stacking
        # Some items may not have pixel_values if images failed to load
        has_pixel_values = all("pixel_values" in inp for inp in input_list)
        if has_pixel_values:
            pixel_values = torch.stack([inp["pixel_values"].squeeze(0) for inp in input_list], dim=0)
            batch["pixel_values"] = pixel_values.to(model.device, dtype=torch.bfloat16)
        
        # Check if ALL items have image_sizes before stacking
        has_image_sizes = all("image_sizes" in inp for inp in input_list)
        if has_image_sizes:
            batch["image_sizes"] = torch.stack([inp["image_sizes"].squeeze(0) for inp in input_list], dim=0).to(model.device)

        return batch