import json


def serialize_logprobs(logprobs):
    """
    Serialize vLLM SampleLogprobs (list[dict[int, Logprob]]) to JSON string.
    Each position contains token_id -> {logprob, rank, decoded_token}.
    """
    if logprobs is None:
        return None
    try:
        result = []
        for pos_data in logprobs:
            if pos_data is None:
                result.append(None)
                continue
            pos_list = []
            for token_id, lp_obj in pos_data.items():
                entry = {"token_id": token_id}
                if hasattr(lp_obj, "logprob"):
                    entry["logprob"] = float(lp_obj.logprob)
                if hasattr(lp_obj, "rank") and lp_obj.rank is not None:
                    entry["rank"] = int(lp_obj.rank)
                if hasattr(lp_obj, "decoded_token") and lp_obj.decoded_token is not None:
                    entry["decoded_token"] = str(lp_obj.decoded_token)
                pos_list.append(entry)
            result.append(pos_list)
        return json.dumps(result)
    except Exception:
        return None


class BaseVLMAdapter:
    def __init__(self, model_name, device="auto", cache_dir=None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

    # def create_template(self, item):
    #     conversation = {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": item["image"]},
    #                 {"type": "text", "text": item["question"]},
    #             ],
    #         }
    #     return conversation

    def create_template(self, item):
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["question"]},
                ],
            }
        ]

    def prepare_inputs(self, messages, processor, model):
        raise NotImplementedError

    def infer(self, model, processor, inputs, max_new_tokens):
        raise NotImplementedError