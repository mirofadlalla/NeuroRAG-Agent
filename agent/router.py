import re
import torch

TOOL_KEYWORDS = {
    "python": [
        "code", "implement", "run", "execute", "python", "debug",
        "example code", "write code", "script"
    ],
    "calculator": [
        "calculate", "compute", "percentage", "sum", "mean",
        "variance", "probability"
    ],
    "search": [
        "latest", "news", "current", "search", "find online",
        "browse", "google", "documentation"
    ],
}

RAG_KEYWORDS = [
    "retrieve", "from documents", "from kb", "knowledge base",
    "context", "reference", "cite", "explain based on docs",
     "explain" # Added these for explicit RAG routing
]

# def route_step(step: str):
#     step_lower = step.lower()
#     # 3) Heuristic: if step looks like "summarize/explain", use RAG by default
#     if re.search(r"(explain|give overview|definition)", step_lower):
#         return {"route": "rag", "tool_name": None}

#     if re.search(r"(summarize)", step_lower):
#         return {"route": "direct", "tool_name": None}

#     # 1) Detect tool usage
#     for tool_name, keywords in TOOL_KEYWORDS.items():
#         for kw in keywords:
#             if kw in step_lower:
#                 return {"route": "tool", "tool_name": tool_name}

#     # 2) Detect RAG usage
#     for kw in RAG_KEYWORDS:
#         if kw in step_lower:
#             return {"route": "rag", "tool_name": None}


#     # 4) Otherwise direct
#     return {"route": "direct", "tool_name": None}


# for better
def route_step(step: str, model, tokenizer ):
    """
    تستخدم الـ LLM لتحديد المسار (الأداة) المناسبة لكل خطوة بشكل ذكي.
    """
    # البرومبت اللي هيوجه الموديل لاختيار الأداة
    router_prompt = f"""Analyze the task and pick the best tool. Respond with ONLY the tool name.

Tools:
- 'direct': To summarize, explain results, or generic reasoning or Write python Code.
- 'rag': To get technical info, facts, or context from documents.
- 'python': To excute or run code, programming scripts, or examples.
- 'calculator': For math operations or arithmetic.

Task: "{step}"

note : Just Return Python only when the task reqire code execution or running code .
when user asks for write code example just return direct not python
Tool:"""

    # مناداة الموديل
    inputs = tokenizer(router_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1)

    decision = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()

    # تحويل الرد لرد مهيكل (Dictionary) زي ما الكود بتاعك متعود
    if "run" in decision or "run" in decision:
        return {"route": "tool", "tool_name": "python"}
    elif "calculator" in decision:
        return {"route": "tool", "tool_name": "calculator"}
    elif "rag" in decision:
        return {"route": "rag", "tool_name": None}
    else:
        return {"route": "direct", "tool_name": None}
