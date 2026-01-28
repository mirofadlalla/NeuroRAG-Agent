## Planner Module

# Planner = مخ الـ Agent
# بياخد Query ويحوّله لخطة structured.

# مثال:

# Input

# "هات لي ملخص عن attention في transformers واكتب مثال كود"

# Output

# 1. Retrieve info about Attention from KB
# 2. Summarize concept
# 3. Generate code example

from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig
import torch
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 4 bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype = torch.float16 ,
    quantization_config=bnb_config,
    device_map="auto")


SYSTEM_PROMPT = """
You are a planning agent.
Break the user request into clear ordered steps.
Return ONLY numbered steps.
Don't include anything else.
Just Return the Steps and nothing else.
When a step involves calculation, output the mathematical expression directly, e.g., 'Calculate 5 + 3' instead of 'Calculate the sum of 5 and 3'.
For Example :
1. Retrieve info about Attention from KB
2. Summarize concept
3. Generate code example
4. Calculate 512 * 512
"""

def generate_plan(query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_only = outputs[0][model_inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_only, skip_special_tokens=True)
