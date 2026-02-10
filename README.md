# MMBU Inference Evaluation Pipeline (Autoregressive)

# Setup

llava will contain environment for LLaVA-Med model

.venv will contain environment for all models

- Can run `bash scripts/setup.sh`

OR (if issue)

- Install `requirements-default.txt` with `uv pip install -r requirements-default.txt` to create .venv
- Clone LLaVA-Med repo in src directory `git clone https://github.com/microsoft/LLaVA-Med.git`
- Install `requirements-llava.txt` in a new uv environment called llava

# Running Code (GCP)

To run code you need to edit `all_tasks.yaml` config file in configs/ and either bq_gcp.sh, gcp.sh, or weill.sh (depending on VM location) eval file in eval/ and then run `./eval/{eval_file}.sh`

## Configs

In a .yaml file:

- Edit all paths to be your local directory paths
- Edit runtime to select model cache path and inference settings
- Edit models for all/any models used
- In "tasks", set the "name" for all eval tasks (can do multiple at once)
    - An example is shown in `configs/all_tasks.yaml`
    - All current tasks are defined in the vista_bench valid_tasks.json section (which is found on GCP)
- In "experiment", set the specific experiment(s) you want to run
- Edit subsample (boolean) if want to use subsampled data
- If on weill cluster, set GPU nodes (not used for GCP)

## Eval

In a .sh file:

- Change .venv paths to current directory and environment directory
- EDIT HUGGINGFACE TOKEN (it will not work as the token you put on github expires)

**NOTE**: All models can be run with `run_vlm_eval.py`

## Models

List of all models that can be implement along with MODEL_TYPE and MODEL_NAME

**NOTE**: Currently InternVL3.5, Qwen3, OctoMed, and Gemma are setup for vLLM inference

- Qwen2-VL-2B-Instruct
    - MODEL_TYPE="qwen2vl"
    - MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
- Qwen2.5-VL-3B-Instruct
    - MODEL_TYPE="qwen2_5vl"
    - MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
- Qwen2.5-VL-7B-Instruct
    - MODEL_TYPE="qwen2_5vl"
    - MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
- Qwen2.5-VL-32B-Instruct
    - MODEL_TYPE="qwen2_5vl"
    - MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
- Qwen3-VL-4B-Instruct
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
- Qwen3-VL-4B-Thinking
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-4B-Thinking"
- Qwen3-VL-8B-Instruct
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
- Qwen3-VL-8B-Thinking
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-8B-Thinking"
- Qwen3-VL-32B-Instruct
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"
- Qwen3-VL-32B-Thinking
    - MODEL_TYPE="qwen3vl"
    - MODEL_NAME="Qwen/Qwen3-VL-32B-Thinking"
- InternVL3_5-8B
    - MODEL_TYPE="intern"
    - MODEL_NAME="OpenGVLab/InternVL3_5-8B"	   
- gemma-3-4b-it
    - MODEL_TYPE="gemma3"
    - MODEL_NAME="google/gemma-3-4b-it"		   
- medgemma-4b-it
    - MODEL_TYPE="gemma3"
    - MODEL_NAME="google/medgemma-4b-it"	   
- Lingshu-7B
    - MODEL_TYPE="lingshu"
    - MODEL_NAME="lingshu-medical-mllm/Lingshu-7B"	   
- Lingshu-32B
    - MODEL_TYPE="lingshu"
    - MODEL_NAME="lingshu-medical-mllm/Lingshu-32B"
- llava-1.5-7b-hf
    - MODEL_TYPE="llava"
    - MODEL_NAME="llava-hf/llava-1.5-7b-hf"	   
- llava-med-v1.5-mistral-7b
    - MODEL_TYPE="llavamed"
    - MODEL_NAME="microsoft/llava-med-v1.5-mistral-7b"
- MedVLM-R1
    - MODEL_TYPE="medvlm"
    - MODEL_NAME="JZPeterPan/MedVLM-R1"
