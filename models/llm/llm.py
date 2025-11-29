
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# device = "cuda" # the device to load the model onto
model_name_or_dir = "meta-llama/Llama-3.2-3B-Instruct"


DEFAULT_SYSTEM_PROMPT = """You are an expert 3D asset director. When the user provides a base caption, expand it into a precise,
engineering-grade description that can guide a multi-view image-to-3D pipeline. Focus exclusively on the object itself
(no environments or backgrounds). Describe geometry, materials, logos or labels, manufacturing details, wear, and any
asymmetries. Explicitly mention what should be visible from the front, left, rear, and right views so that a 2×4 RGB/normal
grid can be rendered. Use complete sentences in a single paragraph, favoring factual language over stylistic flourishes.
Do not invent accessories that were not implied by the caption; instead, clarify existing features.

Return only the enriched description in English."""

def load_llm_model(model_name_or_dir, torch_dtype='auto', device_map='cpu'):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_dir,
        torch_dtype=torch_dtype,
        # torch_dtype=torch.float8_e5m2,
        # torch_dtype=torch.float16,
        device_map=device_map
    )
    print(f'set llm model to {model_name_or_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    print(f'set llm tokenizer to {model_name_or_dir}')
    return model, tokenizer


# print(f"Before load llm model: {torch.cuda.memory_allocated() / 1024**3} GB")
# load_model()
# print(f"After load llm model: {torch.cuda.memory_allocated() / 1024**3} GB")

def get_llm_response(model, tokenizer, user_prompt, seed=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    # global model
    # global tokenizer
    # load_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if seed is not None:
        torch.manual_seed(seed)
        
    # breakpoint()
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# if __name__ == "__main__":
    
    # user_prompt="哈利波特"
    # rsp = get_response(user_prompt, seed=0)
    # print(rsp)
    # breakpoint()