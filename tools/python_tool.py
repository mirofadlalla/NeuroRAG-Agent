# Pyhton Tool

import io
import contextlib
import traceback

class PythonTool:
    name = "PythonTool"
    description = "Execute and run Python Code"

    def __init__(self , max_output_chars = 4000 ):
        self.max_output_chars = max_output_chars

    def run_code(self , code):

        buffer = io.StringIO()

        try :

            local_vars = {}

            with contextlib.redirect_stdout(buffer):
              exec(code , {} , local_vars)

            output = buffer.getvalue().strip()


            if not output :
                if local_vars:
                    output = "Execution finished. Variables:\n" + "\n".join(
                    [f"{k} = {repr(v)[:200]}" for k, v in local_vars.items()]
                )
                else:
                    output = "Execution finished. No output."

            # Cap output
            if len(output) > self.max_output_chars:
                  output = output[: self.max_output_chars] + "\n... [output truncated]"

            return {"status": "success", "output": output}

        except Exception:
            err = traceback.format_exc()
            if len(err) > self.max_output_chars:
                err = err[: self.max_output_chars] + "\n... [error truncated]"

            return {"status": "error", "output": err}