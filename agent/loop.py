import re
import torch

from planner import generate_plan
from router import route_step

class FullAgentSystem:
    def __init__(self, model, tokenizer, tools, memory, rag_tool):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools  # {"python": tool, "calculator": calc}
        self.memory = memory
        self.rag_tool = rag_tool

    def _call_llm(self, prompt, max_tokens=512):
        """Responsible for calling the LLM to generate text"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

    def _prepare_tool_input(self, step_text, tool_name, context=""):
        """
        Convert natural language instructions into technical tool inputs,
        optionally using accumulated context from previous steps
        """
        if tool_name == "python":
            prompt = (
                f"Previous Context: {context}\n"
                f"Task: {step_text}\n"
                "Write ONLY the Python code. No backticks, no markdown, no explanation."
            )
        elif tool_name == "calculator":
            prompt = (
                f"Extract only the math expression from: {step_text}\n"
                "Example: 'multiply 5 by 5' -> 5*5\n"
                "Result:"
            )
        else:
            return step_text

        raw_output = self._call_llm(prompt)

        # Clean any Markdown artifacts from generated code
        clean_output = re.sub(r"```python|```", "", raw_output).strip()

        if tool_name == "calculator":
            match = re.search(r"[\d\.\+\-\*\/\(\)\s\*\*]+", clean_output)
            return match.group(0).strip() if match else clean_output

        return clean_output

    def run(self, user_query):
        print("🚀 Starting Agentic Pipeline (with Context Passing)...")

        # 1. Planning phase
        plan_raw = generate_plan(user_query)
        steps = [s.strip() for s in plan_raw.strip().split('\n') if s.strip()]
        print(f"📝 Plan:\n{plan_raw}\n" + "-" * 40)

        final_report = []
        accumulated_context = ""  # Working memory passed across steps

        # 2. Step execution loop
        for i, step in enumerate(steps, 1):
            print(f"\n🔍 Processing Step {i}: {step}")
            route_info = route_step(step, self.model, self.tokenizer)
            route = route_info["route"]
            tool_name = route_info["tool_name"]

            print(f"🧠 LLM Router Decision: {route} | Tool: {tool_name}")

            result = {"status": "error", "output": "No execution"}

            try:
                if route == "rag":
                    print("📡 Action: RAG Retrieval...")
                    result = self.rag_tool.run(step)
                    if result["status"] == "success":
                        # Aggregate retrieved documents into a single output
                        content = " ".join([d["text"] for d in result["results"]])
                        result["output"] = content

                elif route == "tool":
                    print(f"🛠️ Action: Using {tool_name}...")
                    # Pass accumulated context so tools can leverage previous knowledge
                    refined_input = self._prepare_tool_input(
                        step,
                        tool_name,
                        context=accumulated_context
                    )

                    if tool_name == "python":
                        print("💻 Executing Python code...")
                        result = self.tools["python"].run_code(refined_input)
                    elif tool_name == "calculator":
                        result = self.tools["calculator"].run(refined_input)

                elif route == "direct":
                    print("🤖 Action: LLM Reasoning (using previous context)...")
                    # Key idea: LLM reasons using accumulated context
                    prompt_with_context = (
                        f"Context from previous steps: {accumulated_context}\n"
                        f"Task: {step}\n"
                        "Instruction: Use the context above to complete the task."
                    )
                    result = {
                        "status": "success",
                        "output": self._call_llm(prompt_with_context)
                    }

            except Exception as e:
                result = {"status": "error", "output": str(e)}

            # Update accumulated context with new successful information
            if result.get("status") == "success":
                new_info = str(result.get("output", ""))
                # Add a concise summary so future steps can reference it
                accumulated_context += (
                    f"\n[Step {i} Result Summary]: "
                    f"{new_info[:500]}...\n"
                )

            print(f"📥 Result Status: {result.get('status')}")
            self.memory.add_interaction(step, tool_name or route, result)
            final_report.append({
                "step": step,
                "output": result.get("output")
            })

        # 3. Final answer synthesis
        print("\n" + "=" * 50)
        print("🎯 Generating Final Answer...")
        self._generate_final_answer(user_query, final_report)

    def _generate_final_answer(self, query, report):
        # Aggregate all step outputs into a single synthesis prompt
        summary_prompt = f"User Question: {query}\nStep-by-Step Execution Results:\n"
        for item in report:
            summary_prompt += (
                f"- Step: {item['step']}\n"
                f"  Result: {str(item['output'])[:600]}\n"
            )

        summary_prompt += (
            "\nSynthesize a complete, professional, and helpful final response in Same Languge. "
            "Integrate the code and calculations found above:"
        )

        final_answer = self._call_llm(summary_prompt, max_tokens=1024)
        print("\n" + "*" * 20 + " Final Answer " + "*" * 20)
        print(final_answer)


