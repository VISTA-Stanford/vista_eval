# -------------------------------
# InternVL3.5 Adapter (vLLM)
# -------------------------------
import json
import os
import tempfile
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoProcessor
from .base import BaseVLMAdapter, serialize_logprobs


class InternVL35Adapter(BaseVLMAdapter):
    def load(self):
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        target_seq_len = 120000
        from huggingface_hub import snapshot_download

        temp_dir = tempfile.mkdtemp(prefix="internvl_patched_")
        model_path = snapshot_download(
            self.model_name,
            local_dir=temp_dir,
            local_dir_use_symlinks=True,  # Symlinks to cache; only config.json is patched
        )

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # InternVL nests LLM config under text_config
        text_cfg = config.get("text_config", config)
        original_max = text_cfg.get("max_position_embeddings", 40960)
        factor = target_seq_len / original_max

        text_cfg["max_position_embeddings"] = target_seq_len
        text_cfg["rope_scaling"] = {"type": "dynamic", "factor": round(factor, 2)}

        # Write patched config (replace symlink to avoid modifying shared HF cache)
        if os.path.islink(config_path):
            os.unlink(config_path)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"⚡ Patched InternVL config: {original_max} -> {target_seq_len} (factor={factor:.2f})")

        llm = LLM(
            model=model_path,
            tokenizer=self.model_name,
            max_model_len=target_seq_len,
            rope_scaling={"type": "dynamic", "factor": round(factor, 2)},
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            limit_mm_per_prompt={"image": 100, "video": 0},
            enforce_eager=True,  # Bypass torch.compile cache compiled with 40960
        )

        processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
            processor.tokenizer.model_max_length = target_seq_len

        return llm, processor

    def create_template(self, item):
        content = []
        # Handle images: text only, one image, or multiple images
        image = item.get("image")
        if image is not None:
            if isinstance(image, list):
                for img in image:
                    if img is not None:
                        content.append({"type": "image", "image": img})
            else:
                content.append({"type": "image", "image": image})
        # Always add the text question
        content.append({"type": "text", "text": item["question"]})

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def prepare_inputs(self, messages, processor, model):
        """
        Prepare inputs for vLLM.
        Returns a list of input dictionaries containing 'prompt' and 'multi_modal_data'.
        """
        all_inputs = []

        for msg in messages:
            pixel_values = []
            clean_messages = []

            for message in msg:
                role = message["role"]
                content = message["content"]

                clean_content = []
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            img = item.get("image")
                            if img is not None:
                                pixel_values.append(img)
                            clean_content.append({"type": "image"})
                        else:
                            clean_content.append(item)
                else:
                    clean_content = content

                clean_messages.append({"role": role, "content": clean_content})

            prompt_text = processor.apply_chat_template(
                clean_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Construct multi_modal_data for vLLM
            multi_modal_data = {}
            if pixel_values:
                multi_modal_data["image"] = pixel_values[0] if len(pixel_values) == 1 else pixel_values

            input_dict = {
                "prompt": prompt_text,
                "multi_modal_data": multi_modal_data if multi_modal_data else None
            }
            all_inputs.append(input_dict)

        return all_inputs

    def stack_inputs(self, input_list, model):
        """Pass-through for vLLM; takes list of input dicts."""
        return input_list

    def infer(self, model, processor, inputs, max_new_tokens, constrained_choices=None):
        """
        Run inference using vLLM's generate API.
        When constrained_choices is provided (e.g. ["Yes", "No"] for binary tasks),
        uses vLLM structured outputs to force the model to output exactly one of them.
        """
        if isinstance(inputs, dict):
            request_list = [inputs]
        elif isinstance(inputs, list):
            request_list = inputs
        else:
            raise TypeError(f"Expected inputs to be dict or list, got {type(inputs)}")

        sampling_kwargs = {
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
        }
        if constrained_choices:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                choice=constrained_choices
            )
            sampling_kwargs["logprobs"] = 2  # Return logprobs for Yes/No confidence
        sampling_params = SamplingParams(**sampling_kwargs)

        outputs = model.generate(
            request_list,
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
