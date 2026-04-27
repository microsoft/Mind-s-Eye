from ollama import chat
import re
import pandas as pd
from tqdm import tqdm
import os
import json
from transformers import MllamaForConditionalGeneration, AutoProcessor,AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor,LlavaNextForConditionalGeneration,AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric
from PIL import Image
import torch
import gc
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import OllamaModel
# Use a pipeline as a high-level helper
# from transformers import pipeline
from internvl35 import get_internvl_results
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_ROOT = os.getenv("MINDS_EYE_DATASET_ROOT", os.path.join(PROJECT_ROOT, "data"))
PARENT_DIR = os.getenv(
    "MINDS_EYE_EVAL_OUTPUT_DIR",
    os.path.join(PROJECT_ROOT, "model_evaluation"),
)
HF_TOKEN = os.getenv("HF_TOKEN", "")

MODELS = {
    "llama":"meta-llama/Llama-3.2-11B-Vision",
    "internvl3_5":"OpenGVLab/InternVL3_5-8B",
    "qwen7b":"Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3b":"Qwen/Qwen2.5-VL-3B-Instruct",
    "llava" : "llava-hf/llava-v1.6-mistral-7b-hf",
    "Idefics3":"HuggingFaceM4/Idefics3-8B-Llama3"
}
DEVICE = os.getenv("MINDS_EYE_DEVICE", "cuda:0")
def load_model_and_processor(model_name: str, device: str):
    """
    Loads the specified model and its corresponding processor.
    """
    model_path = MODELS[model_name]
    if model_name == "llama":
        model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device,token=HF_TOKEN)
        processor = AutoProcessor.from_pretrained(model_path,token=HF_TOKEN)
    elif model_name == "phi":
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.float16,_attn_implementation='eager')
        model.config.use_cache = False
        processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)#,num_crops=4)
    elif model_name == "internvl3_5":
        model, processor = None, None  # Handled separately
    elif model_name == "qwen7b" or model_name == "qwen3b":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_name == "llava":
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device)
        processor = AutoProcessor.from_pretrained(model_path)
    elif model_name == "Idefics3":
        model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
        processor = AutoProcessor.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model, processor

def unload_model_and_clear_cache(model, processor):
    """
    Deletes model and processor objects and clears GPU cache.
    """
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()


class OptionExtractionMetric(BaseMetric):
    """
    A custom metric to extract a final option (e.g., A, B, C, D) from an LLM's
    verbose response and compare it to the expected option.
    """
    def __init__(
        self,
        model: DeepEvalBaseLLM, # Model is now passed during instantiation
        threshold: float = 1.0
    ):
        super().__init__()
        self.model = model

    def measure(self, test_case: LLMTestCase) -> float:
        llm_response_text = test_case.actual_output
        if llm_response_text is None:
            self.success = False
            self.reason = "Actual output was None."
            return 0.0, None

        expected_option = test_case.expected_output.strip().upper()

        extraction_prompt = f"""
        Review the following text which contains a detailed explanation and a final answer choice for a multiple-choice question. Your task is to extract ONLY the final letter of the selected option (e.g., A, B, C, D). Do not provide any explanation or extra text. If there is not clear final answer choice give option X

        Text:
        "{llm_response_text}"

        Extracted Option:
        """
        
        try:
            extracted_option_raw, _ = self.model.generate(prompt=extraction_prompt)
            extracted_option = extracted_option_raw.strip().upper()
        except Exception as e:
            self.success = False
            self.reason = f"Failed to connect to the Ollama model. Ensure Ollama is running and the model is pulled. Error: {e}"
            return 0.0,self.reason

        if len(extracted_option) > 1 or extracted_option not in "ABCDX":
             self.success = False
             self.reason = f"The model extracted an invalid option: '{extracted_option}'. It should be a single letter like A, B, C, or D."
             self.score = 0.0
             return self.score,self.reason

        self.success = (extracted_option.lower() == expected_option.strip('()').lower())
        
        if self.success:
            self.reason = extracted_option
            self.score = 1.0
        else:
            self.reason = extracted_option
            self.score = 0.0
            
        return self.score, self.reason

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Option Extraction"


ollama_deepseek_model = OllamaModel(
    model=os.getenv("MINDS_EYE_OPTION_EXTRACTOR_MODEL", "gemma3:4b"),
    base_url=os.getenv("MINDS_EYE_OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
)
extraction_metric = OptionExtractionMetric(model=ollama_deepseek_model)


JSON_LIST = {
    "mrt": os.path.join(DATASET_ROOT, "mental_rotation", "annotations.json"),
    "paper_folding": os.path.join(DATASET_ROOT, "paper_folding", "annotations.json"),
    "dynamic_isomorphism": os.path.join(DATASET_ROOT, "dynamic_isomorph", "annotations.json"),
    "hierarchical_isomorphism": os.path.join(DATASET_ROOT, "hierarchial_isomorph", "annotations.json"),
    "analogies": os.path.join(DATASET_ROOT, "abstract", "annotations.json"),
    "symmetric_structures": os.path.join(DATASET_ROOT, "symmetric_isomorph", "annotations.json"),
    "mental_composition": os.path.join(DATASET_ROOT, "mental_composition", "annotations.json"),
    "slippage": os.path.join(DATASET_ROOT, "slippage", "annotations.json"),
}

ANSWER_KEY = {
    "analogies":"answer",
    "dynamic_isomorphism":"fifth_label",
    "hierarchical_isomorphism": "answer",
    "mental_composition": "answer",
    "mrt" : "answer",
    "paper_folding":"correct_option",
    "slippage":"violation",
    "symmetric_structures":"asymmetric_label"
}

def run_llama(model, processor, image_path, question):
    # Implement the function to run inference with the LLaMA model
    image = Image.open(image_path)
    prompt = f"<|image|><|begin_of_text|>{question}"
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=1000,temperature=0.1)
    return processor.decode(output[0])

def run_phi(model,processor,image_path,question):
    image = Image.open(image_path)
    messages = [{"role": "user", "content": f"<|image_1|>{question}"},]
    prompt = processor.tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to(model.device) 

    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 0.1, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs,eos_token_id=processor.tokenizer.eos_token_id,**generation_args)

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0] 

    return response


def run_qwen25vl(model, processor,image_path,question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": f"{question}."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs,max_new_tokens=1000,temperature=0.1)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    match = re.search(r'```json\s*({.*?})\s*```', output_text[0], re.DOTALL)
    if not match:
        return output_text[0]
    json_str = match.group(1)
    json_str = json_str.replace('\n', ' ').replace('\t', '')
    return json_str

def run_llava(model, processor,image_path,question):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": f"{question}."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path)
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1000,temperature=0.1)
    output_text = processor.decode(
        generated_ids[0], skip_special_tokens=True
    )
    # Extract JSON block using regex
    # print(output_text)
    match = re.search(r'```json\s*({.*?})\s*```', output_text, re.DOTALL)
    if not match:
        return output_text
    json_str = match.group(1)
    json_str = json_str.replace('\n', ' ').replace('\t', '')
    return json_str
    
def run_idefics(model, processor,image_path, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]
        }
    ]
    image = Image.open(image_path)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # The fix is in the 'images' argument below
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    
    inputs = inputs.to(DEVICE)
    # The 'del' line is optional but harmless if you keep it
    if "image_grid_thw" in inputs:
        del inputs["image_grid_thw"]
    
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=1000,temperature=0.1)
    output_text_list = processor.batch_decode(generated_ids, skip_special_tokens=True)
    output_text = output_text_list[0] # batch_decode returns a list

    match = re.search(r'```json\s*({.*?})\s*```', output_text, re.DOTALL)
    if not match:
        # Also clean up the prompt from the output if it exists
        assistant_response = re.search(r'Assistant:(.*)', output_text, re.DOTALL)
        if assistant_response:
            return assistant_response.group(1).strip()
        return output_text
        
    json_str = match.group(1)
    json_str = json_str.replace('\n', ' ').replace('\t', '')
    return json_str

# --- Dispatcher and Evaluation Loop ---


def get_model_output(model_name: str, model, processor, image_path: str, question: str) -> str:
    """Dispatches to the correct model inference function."""
    if model_name == "llama":
        return run_llama(model, processor, image_path, question)
    elif model_name == "internvl3_5":
        return get_internvl_results(image_path, question)
    elif model_name == "phi":
        return run_phi(model, processor, image_path, question)
    elif model_name == "qwen7b" or model_name == "qwen3b":
        return run_qwen25vl(model, processor, image_path, question)
    elif model_name == "llava":
        return run_llava(model, processor, image_path, question)
    elif model_name == "Idefics3":
        return run_idefics(model, processor, image_path, question)
    # Add other models here
    else:
        raise ValueError(f"Inference function not implemented for model: {model_name}")

def evaluate_model(model_name, model, processor, step, save_results = True):
    output_dir = os.path.join(PARENT_DIR, model_name,f"step_{step+1}")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results_csv_data = []
    results = {}

    print(f"Starting evaluation for {model_name}")

    for dataset_name, json_path in JSON_LIST.items():
        print(f"  Processing dataset: {dataset_name}")
        parent_dir = os.path.dirname(json_path)
        with open(json_path, "r") as f:
            data = json.load(f)
        
        dataset_results = {}
        dataset_csv_data = []

        # Use tqdm for a progress bar over the items in the dataset
        for filename, obj in tqdm(data.items(), desc=f"  -> {dataset_name}"):
            full_path = os.path.join(parent_dir, filename)
            question = obj.get("question")
            if not question:
                continue

            response = get_model_output(model_name, model, processor, full_path, question[0])
            
            obj_with_response = obj.copy()
            obj_with_response["response"] = response
            answer = obj_with_response.get(ANSWER_KEY[dataset_name])
            
            test_case = LLMTestCase(input="",actual_output=response,expected_output=answer)

            score,reason = extraction_metric.measure(test_case=test_case)
            obj_with_response["judge_score"] = score
            obj_with_response["judge_reason"] = reason
            dataset_results[filename] = obj_with_response
            
            csv_row = {
                'task_name': dataset_name,
                'image_name': filename,
                'difficulty': obj_with_response.get("difficulty"),
                'question': question,
                'answer': answer,
                'response': response,
                'judge_score': score,
                'judge_reason': reason
            }
            dataset_csv_data.append(csv_row)
            all_results_csv_data.append(csv_row)
            break
        results[dataset_name] = dataset_results
        # results[dataset_name] = dataset_results
        if save_results:
            # Save intermediate results for each dataset
            tmpdf = pd.DataFrame(dataset_csv_data)
            tmp_csv_path = os.path.join(output_dir, f'{model_name}_judged_{dataset_name}.csv')
            tmpdf.to_csv(tmp_csv_path, index=False)
            
    print("Processing Complete")
    if save_results:
        json_path = os.path.join(output_dir,f'{model_name}_juded_model_answer.json')
        with open(json_path, "w") as out_f:
            json.dump(results, out_f, indent=2)
        print(f"  Saved aggregated JSON to {json_path}")

        df = pd.DataFrame(all_results_csv_data)
        csv_path = os.path.join(output_dir, f'{model_name}_phi3_judged_all_tasks.csv')
        df.to_csv(csv_path, index=False)
        print(f"  Saved aggregated CSV to {csv_path}")


if __name__ == "__main__":
    for model_name_key in MODELS.keys():
        print(f"\n{'='*20} RUNNING PIPELINE FOR MODEL: {model_name_key.upper()} {'='*20}")
        
        # 1. Load the model and its processor
        print(f"Step 1/3: Loading model '{model_name_key}' onto {DEVICE}...")
        model_obj, processor_obj = load_model_and_processor(model_name_key, DEVICE)
        print("         Load complete.")

        # 2. Run the full evaluation
        print(f"Step 2/3: Starting evaluation for '{model_name_key}'...")
        for i in range(3):
            print(f"Step 2.{i+1}/3: Starting evaluation for '{model_name_key}'...")
            evaluate_model(model_name_key, model_obj, processor_obj, i,save_results=True)
        print("         Evaluation complete.")

        # 3. Unload the model and free up memory
        print(f"Step 3/3: Unloading model '{model_name_key}' and clearing cache...")
        unload_model_and_clear_cache(model_obj, processor_obj)
        print("         Unload complete.")

        print(f"{'='*20} PIPELINE FOR {model_name_key.upper()} FINISHED {'='*20}\n")
    
    print("All models have been evaluated.")
