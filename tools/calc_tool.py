import ast
import operator as op

class CalcTool:
    name = "calculator"
    description = "Safely evaluates basic math expressions (+, -, *, /, **, parentheses)."

    # Allowed operators
    ALLOWED_OPERATORS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def run(self, expression: str):
        """
        Evaluate a math expression safely using AST parsing.
        """
        try:
            expression = expression.strip()

            if not expression:
                return {"status": "error", "output": "Empty expression."}

            node = ast.parse(expression, mode="eval").body
            result = self._eval(node)

            return {"status": "success", "output": str(result)}

        except ZeroDivisionError:
            return {"status": "error", "output": "Division by zero."}

        except Exception as e:
            return {"status": "error", "output": f"Invalid expression: {e}"}

    def _eval(self, node):
        # Numbers
        if isinstance(node, ast.Num):  # means instance must be number int or floar
            return node.n

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left)
            right = self._eval(node.right)
            operator_type = type(node.op)

            if operator_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Operator {operator_type} not allowed.")

            return self.ALLOWED_OPERATORS[operator_type](left, right)

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand)
            operator_type = type(node.op)

            if operator_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"Unary operator {operator_type} not allowed.")

            return self.ALLOWED_OPERATORS[operator_type](operand)

        raise ValueError(f"Unsupported expression type: {type(node)}")


# calc = CalcTool()

# print(calc.run("2 + 3 * 10"))          # 32
# print(calc.run("(100 - 25) / 5"))      # 15.0
# print(calc.run("2 ** 8"))              # 256
# print(calc.run("10 / 0"))              # error
# print(calc.run("__import__('os')"))    # error

# expression = "2 + 3 * 4"
'''
expression will be
        +
       / \
      2   *
         / \
        3   4

and so on every expresson will be a tree
'''
# node = ast.parse(expression, mode="eval").body # .body main node
# node.left.value # left vlaue of main noed + which is 2