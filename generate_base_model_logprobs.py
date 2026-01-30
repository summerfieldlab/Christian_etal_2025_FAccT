import yaml
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Paths relative to scripts_repository root
SCRIPT_ROOT = Path(__file__).parent
CONFIG_DIR = SCRIPT_ROOT / 'config'
OUTPUT_DIR = SCRIPT_ROOT / 'data' / 'base_model_logits'

# Load model and prompt configurations
with open(CONFIG_DIR / 'gemma_base_models.yaml', 'r') as f:
    models = yaml.safe_load(f)
with open(CONFIG_DIR / 'prompts.yaml', 'r') as f:
    prompts = yaml.safe_load(f)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# For mapping YAML string dtypes to torch dtypes
dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def template_for_pretrained_model(tok, prompt):
    prompt_string = f"User: {prompt}\nAssistant:"
    model_inputs = tok(prompt_string, return_tensors='pt')
    return model_inputs

def template_for_instruction_tuned_model(tok, prompt):
    messages = [
        {'role': 'user', 'content': prompt},
    ]
    model_inputs = tok.apply_chat_template(
        messages,
        return_tensors='pt',
        return_dict=True,
        add_generation_prompt=True
    )
    return model_inputs

for model_info in models:
    print(f"Processing model: {model_info['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
    model = AutoModelForCausalLM.from_pretrained(
        model_info['name'],
        torch_dtype=dtype_map[model_info['dtype']],
        device_map='auto'
    ).eval()
    
    prompt_log_prob_results = {}

    for prompt_name, prompt_content in tqdm(prompts.items(), desc="Prompts", leave=False):

        if model_info['type'] == 'pretrained':
            model_inputs = template_for_pretrained_model(tokenizer, prompt_content)
        elif model_info['type'] == 'instruction-tuned':
            model_inputs = template_for_instruction_tuned_model(tokenizer, prompt_content)
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
        
        # Move inputs to the same device as the model
        device = model.get_input_embeddings().weight.device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # Perform forward pass through the model
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                use_cache=False
            )

            # Get logits for the last token position
            last_token_logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]

            # Upcast to float32 for precision
            logits = last_token_logits.float()                                                                             
            # log_probs = torch.log_softmax(logits, dim=-1)

        # Save it
        prompt_log_prob_results[prompt_name] = logits.cpu().numpy().tolist()

    # Save to CSV with clean format:
    # Columns: token_id, token_name, prompt1_logit, prompt2_logit, ...
    safe_name = model_info['name'].replace('/', '--')
    print(f"Saving model: {model_info['name']}")

    vocab = tokenizer.get_vocab()  # {token_name: token_id}
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])  # sort by token_id
    token_names, token_ids = zip(*vocab_sorted)
    token_decoded = tokenizer.batch_decode([[tid] for tid in token_ids], skip_special_tokens=False)

    df = pd.DataFrame({
        'token_id': token_ids,
        'token_name': token_names,
        'token_decoded': token_decoded
    })

    for prompt_name, prompt_log_probs in prompt_log_prob_results.items():
        df[prompt_name] = [prompt_log_probs[tid] for tid in token_ids]
    
    df.to_csv(OUTPUT_DIR / f"{safe_name}.csv", index=False)

    # Memory cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
