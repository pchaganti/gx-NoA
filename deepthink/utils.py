"""
Core utilities and shared functions for DeepThink.
"""
import re
import json
import io
import asyncio
from contextlib import redirect_stdout, redirect_stderr


def clean_and_parse_json(llm_output_string):
    """
    Finds and parses the first valid JSON object within a string.

    Args:
        llm_output_string: The raw string output from the language model.

    Returns:
        A Python dictionary representing the JSON data, or None if parsing fails.
    """
    match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output_string)
    if match:
        json_string = match.group(1)
    else:
        try:
            start_index = llm_output_string.index('{')
            end_index = llm_output_string.rindex('}') + 1
            json_string = llm_output_string[start_index:end_index]
        except ValueError:
            print("Error: No JSON object found in the string.")
            return None

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic string: {json_string}")
        return None


def execute_code_in_sandbox(code: str) -> tuple:
    """
    Executes a string of Python code and captures its stdout/stderr.
    Returns a tuple of (success: bool, output: str).
    """
    if not code:
        return True, "No code to execute."
        
    # Extract code from markdown block if present
    code_match = re.search(r"```(?:python\n)?([\s\S]*?)```", code)
    if code_match:
        code = code_match.group(1).strip()

    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            # Using a restricted globals dict for a little more safety
            exec(code, {'__builtins__': {
                'print': print, 'range': range, 'len': len, 'str': str, 'int': int, 'float': float, 
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'True': True, 'False': False, 'None': None
            }})
        return True, output_buffer.getvalue()
    except Exception as e:
        return False, f"{output_buffer.getvalue()}\n\nERROR: {type(e).__name__}: {e}"
