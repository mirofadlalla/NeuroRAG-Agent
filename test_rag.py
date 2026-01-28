from agent.loop import FullAgentSystem
from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig
import torch
from tools.python_tool import PythonTool
from tools.calc_tool import CalcTool
from tools.rag_tool import RAGTool
from agent.memory import AgentMemory

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

# --- System Execution ---


agent_system = FullAgentSystem(
    model=model,
    tokenizer=tokenizer,
    tools={"python": PythonTool, "calculator": CalcTool},
    memory=AgentMemory,
    rag_tool=RAGTool
)


query = "What are the two main pre-training objectives used in BERT and give me  a code for lora?"
agent_system.run(query)