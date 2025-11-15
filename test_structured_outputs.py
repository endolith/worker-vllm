#!/usr/bin/env python3
"""
Test script for vLLM worker structured outputs functionality.
Tests basic responses, structured outputs, and field ordering.
"""

import requests
import json
import time
import os
import datetime
import sys
from typing import Dict, Any, Tuple, Callable
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configuration
ENDPOINT_ID = "4vo3wqk62hyekr"
API_KEY = os.getenv("RUNPOD_API_KEY")
if not API_KEY:
    raise ValueError("RUNPOD_API_KEY not found in environment variables. Please set it in .env file.")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


class Tee:
    """Write to both file and console."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def wait_for_job(job_id: str, timeout: int = 300) -> Dict[str, Any]:
    """Poll for job completion and return the result."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers)
        if response.status_code != 200:
            return {"error": f"Failed to get job status: {response.status_code}"}

        result = response.json()
        status = result.get("status")

        if status == "COMPLETED":
            return result
        elif status == "FAILED":
            return result
        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(2)
            continue
        else:
            return result

    return {"error": f"Job {job_id} timed out after {timeout} seconds"}


def run_test(name: str, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Run a test and return the response."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        print(f"Request: {json.dumps(data, indent=2)}")

    # Submit the job
    response = requests.post(f"{BASE_URL}/run", headers=headers, json=data)

    if verbose:
        print(f"\nStatus Code: {response.status_code}")
    if response.status_code != 200:
        return {"error": f"Failed to submit job: {response.status_code}", "response": response.text}

    job_result = response.json()
    job_id = job_result.get("id")

    if not job_id:
        return {"error": "No job ID returned", "response": job_result}

    if verbose:
        print(f"Job ID: {job_id}")
        print(f"Initial Status: {job_result.get('status')}")
        print("Waiting for job to complete...")

    # Wait for job completion
    final_result = wait_for_job(job_id)

    if verbose:
        print(f"\nFinal Status: {final_result.get('status')}")
        print(f"Response: {json.dumps(final_result, indent=2)}")

    return final_result


def run_test_wrapper(test_func: Callable[[], Dict[str, Any]], test_name: str) -> Tuple[str, Dict[str, Any]]:
    """Wrapper function to run a test and return (test_name, result) tuple."""
    try:
        result = test_func()
        return (test_name, result)
    except Exception as e:
        print(f"[ERROR] {test_name}: {str(e)}", flush=True)
        return (test_name, {"error": f"Exception during test execution: {str(e)}"})


def test_basic_response():
    """Test 1: Basic response without structured outputs."""
    data = {
        "input": {
            "messages": [
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1
            }
        }
    }
    return run_test("Basic Response", data, verbose=False)


def test_structured_outputs_new_api():
    """Test 2: New structured_outputs API (should work)."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML ONLY: reasoning: |\n  your thoughts\nquestion: your question"
                },
                {
                    "role": "user",
                    "content": "Analyze: The Eiffel Tower was completed in 1889 for the World's Fair."
                }
            ],
            "sampling_params": {
                "max_tokens": 500,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string"},
                                "question": {"type": "string"}
                            },
                            "required": ["reasoning", "question"]
                        }
                    }
                }
            }
        }
    }
    return run_test("Structured Outputs (New API)", data, verbose=False)


def test_guided_json_deprecated():
    """Test 3: Old guided_json API (should be rejected)."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML ONLY: reasoning: |\n  your thoughts\nquestion: your question"
                },
                {
                    "role": "user",
                    "content": "Analyze: The Eiffel Tower was completed in 1889 for the World's Fair."
                }
            ],
            "sampling_params": {
                "max_tokens": 500,
                "temperature": 0.1
            },
            "guided_json": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["reasoning", "question"]
            },
            "guided_decoding_backend": "outlines"
        }
    }
    return run_test("Guided JSON (Deprecated - Should Reject)", data, verbose=False)


def test_structured_outputs_with_yaml_conflict():
    """Test 4: Structured outputs with conflicting YAML system message."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "You MUST output YAML format only, not JSON. Use this format:\nreasoning: |\n  your thoughts\nquestion: your question"
                },
                {
                    "role": "user",
                    "content": "Analyze: The Eiffel Tower was completed in 1889 for the World's Fair."
                }
            ],
            "sampling_params": {
                "max_tokens": 500,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string"},
                                "question": {"type": "string"}
                            },
                            "required": ["reasoning", "question"]
                        }
                    }
                }
            }
        }
    }
    return run_test("Structured Outputs vs YAML System Message (Schema should win)", data, verbose=False)


def test_json_field_order():
    """Test 5: JSON field order enforcement."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "Output the question field FIRST, then the reasoning field SECOND. Do not follow the schema order."
                },
                {
                    "role": "user",
                    "content": "What is the capital of France? Think step by step."
                }
            ],
            "sampling_params": {
                "max_tokens": 500,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string"},
                                "question": {"type": "string"}
                            },
                            "required": ["reasoning", "question"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Field Order (reasoning should come before question)", data, verbose=False)


def test_structured_outputs_no_system_message():
    """Test 6: Structured outputs without system message."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze: The Eiffel Tower was completed in 1889 for the World's Fair."
                }
            ],
            "sampling_params": {
                "max_tokens": 500,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string"},
                                "question": {"type": "string"}
                            },
                            "required": ["reasoning", "question"]
                        }
                    }
                }
            }
        }
    }
    return run_test("Structured Outputs (No System Message)", data, verbose=False)


def test_basic_no_structured_outputs():
    """Test 7: Basic response without structured outputs."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'hello world'"
                }
            ],
            "sampling_params": {
                "max_tokens": 10,
                "temperature": 0.1
            }
        }
    }
    return run_test("Basic Response (No Structured Outputs)", data, verbose=False)


def test_json_array_with_number():
    """Test 8: JSON schema with array and number fields."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML ONLY: step1: first step\nstep2: second step\nstep3: third step"
                },
                {
                    "role": "user",
                    "content": "Break down the process of making coffee into steps"
                }
            ],
            "sampling_params": {
                "max_tokens": 300,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "total_time": {"type": "number"}
                            },
                            "required": ["steps", "total_time"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Array with Number Field", data, verbose=False)


def test_json_nested_object():
    """Test 9: JSON schema with nested object."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML ONLY:\nperson:\n  name: John\n  age: 30\n  city: New York"
                },
                {
                    "role": "user",
                    "content": "Describe a fictional character"
                }
            ],
            "sampling_params": {
                "max_tokens": 200,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "character": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "age": {"type": "integer"},
                                        "traits": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["name", "age", "traits"]
                                }
                            },
                            "required": ["character"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Nested Object with Array", data, verbose=False)


def test_json_enum():
    """Test 10: JSON schema with enum and number constraints."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML ONLY:\nstatus: success\nerror: null\nmessage: done"
                },
                {
                    "role": "user",
                    "content": "Process this request and report status"
                }
            ],
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "processing", "completed", "failed"]
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1
                                }
                            },
                            "required": ["status", "confidence"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Enum and Number Constraints", data, verbose=False)


def test_regex_pattern():
    """Test 11: Regex pattern structured output."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: email: user@domain\nphone: 123-456-7890\nNever use the format: name-1234"
                },
                {
                    "role": "user",
                    "content": "Generate a user identifier"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^[a-z]+-[0-9]{4}$"
                    }
                }
            }
        }
    }
    return run_test("Regex Pattern (name-1234)", data, verbose=False)


def test_choice_pattern():
    """Test 12: Choice pattern structured output."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: maybe or sometimes\nNever use: yes/no"
                },
                {
                    "role": "user",
                    "content": "Answer yes or no: Is the sky blue?"
                }
            ],
            "sampling_params": {
                "max_tokens": 20,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "choice": ["yes", "no"]
                    }
                }
            }
        }
    }
    return run_test("Choice Pattern (yes/no)", data, verbose=False)


def test_grammar_pattern():
    """Test 13: Grammar pattern structured output."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: SELECT * FROM users\nNever use WHERE clauses"
                },
                {
                    "role": "user",
                    "content": "Create a SQL query to find users named John"
                }
            ],
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "grammar": "root ::= \"SELECT\" \" \" \"name\" \" \" \"FROM\" \" \" \"users\" \" \" \"WHERE\" \" \" \"name\" \" \" \"=\" \" \" \"'John'\""
                    }
                }
            }
        }
    }
    return run_test("Grammar Pattern (SQL Query)", data, verbose=False)


def test_json_object_pattern():
    """Test 14: JSON object pattern (no schema)."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT PLAIN TEXT: The answer is true\nNever use JSON format"
                },
                {
                    "role": "user",
                    "content": "Output a JSON object"
                }
            ],
            "sampling_params": {
                "max_tokens": 30,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json_object": True
                    }
                }
            }
        }
    }
    return run_test("JSON Object Pattern (No Schema)", data, verbose=False)


def test_guided_json_outlines():
    """Test 15: Guided JSON with outlines backend (deprecated)."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML ONLY: reasoning: |\n  your thoughts\nquestion: your question"
                },
                {
                    "role": "user",
                    "content": "Analyze: The Eiffel Tower was completed in 1889 for the World's Fair."
                }
            ],
            "sampling_params": {
                "max_tokens": 500,
                "temperature": 0.1
            },
            "guided_json": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["reasoning", "question"]
            },
            "guided_decoding_backend": "outlines"
        }
    }
    return run_test("Guided JSON with Outlines (Deprecated)", data, verbose=False)


def test_guided_json_lm_format_enforcer():
    """Test 16: Guided JSON with lm-format-enforcer backend (deprecated)."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML ONLY: reasoning: |\n  your thoughts\nquestion: your question"
                },
                {
                    "role": "user",
                    "content": "Analyze: The Eiffel Tower was completed in 1889 for the World's Fair."
                }
            ],
            "sampling_params": {
                "max_tokens": 500,
                "temperature": 0.1
            },
            "guided_json": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["reasoning", "question"]
            },
            "guided_decoding_backend": "lm-format-enforcer"
        }
    }
    return run_test("Guided JSON with LM Format Enforcer (Deprecated)", data, verbose=False)


def test_json_optional_fields():
    """Test 17: JSON schema with optional fields."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a response with optional fields"
                }
            ],
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "required_field": {"type": "string"},
                                "optional_field": {"type": "string"}
                            },
                            "required": ["required_field"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON with Optional Fields", data, verbose=False)


def test_json_nested_array():
    """Test 18: JSON schema with nested array of objects."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Break down a complex task into steps"
                }
            ],
            "sampling_params": {
                "max_tokens": 200,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "explanation": {"type": "string"},
                                            "substeps": {"type": "array", "items": {"type": "string"}}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Nested Array of Objects", data, verbose=False)


def test_json_simple_fields():
    """Test 19: JSON schema with simple required fields."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Output fields A and B"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "string"},
                                "b": {"type": "string"}
                            },
                            "required": ["a", "b"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Simple Required Fields", data, verbose=False)


def test_yaml_system_message():
    """Test 20: YAML system message with JSON structured output."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML: reasoning: thoughts\nquestion: query"
                },
                {
                    "role": "user",
                    "content": "Analyze: The Eiffel Tower was completed in 1889 for the World's Fair."
                }
            ],
            "sampling_params": {
                "max_tokens": 200,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "reasoning": {"type": "string"},
                                "question": {"type": "string"}
                            },
                            "required": ["reasoning", "question"]
                        }
                    }
                }
            }
        }
    }
    return run_test("YAML System Message with JSON Output", data, verbose=False)


def test_regex_email_conflict():
    """Test 21: Regex pattern with conflicting system message."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: email: user@domain.com"
                },
                {
                    "role": "user",
                    "content": "Generate a user identifier"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^[a-z]+-[0-9]{4}$"
                    }
                }
            }
        }
    }
    return run_test("Regex Pattern with Email Conflict", data, verbose=False)


def test_grammar_sql_conflict():
    """Test 22: Grammar pattern with conflicting system message."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: SELECT * FROM users"
                },
                {
                    "role": "user",
                    "content": "Create a SQL query to find users named John"
                }
            ],
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "grammar": "root ::= \"SELECT\" \" \" \"name\" \" \" \"FROM\" \" \" \"users\" \" \" \"WHERE\" \" \" \"name\" \" \" \"=\" \" \" \"'John'\""
                    }
                }
            }
        }
    }
    return run_test("Grammar Pattern with SQL Conflict", data, verbose=False)


def test_json_object_conflict():
    """Test 23: JSON object pattern with conflicting system message."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT PLAIN TEXT: The answer is true\nNever use JSON format"
                },
                {
                    "role": "user",
                    "content": "Output a JSON object"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json_object": True
                    }
                }
            }
        }
    }
    return run_test("JSON Object Pattern with Text Conflict", data, verbose=False)


def test_json_array_fixed_length():
    """Test 24: JSON schema with fixed-length array."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT COMMA-SEPARATED: item1, item2, item3"
                },
                {
                    "role": "user",
                    "content": "List three colors"
                }
            ],
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Fixed-Length Array", data, verbose=False)


def test_json_number_constraints():
    """Test 25: JSON schema with number constraints."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT TEXT: The number is ten"
                },
                {
                    "role": "user",
                    "content": "Provide a number between 1 and 100"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "number": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100
                                }
                            },
                            "required": ["number"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Number Constraints", data, verbose=False)


def test_json_boolean_field():
    """Test 26: JSON schema with boolean field."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "OUTPUT YAML: success: yes\nerror: no"
                },
                {
                    "role": "user",
                    "content": "Evaluate if this worked"
                }
            ],
            "sampling_params": {
                "max_tokens": 100,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "message": {"type": "string"}
                            },
                            "required": ["success", "message"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Boolean Field", data, verbose=False)


def test_regex_date_format():
    """Test 27: Regex pattern for date format."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: 2024/11/15"
                },
                {
                    "role": "user",
                    "content": "Provide a date in YYYY-MM-DD format"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^\\d{4}-\\d{2}-\\d{2}$"
                    }
                }
            }
        }
    }
    return run_test("Regex Date Format (YYYY-MM-DD)", data, verbose=False)


def test_regex_email_format():
    """Test 28: Regex pattern for email format."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: username: john_doe123"
                },
                {
                    "role": "user",
                    "content": "Generate an email address"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                    }
                }
            }
        }
    }
    return run_test("Regex Email Format", data, verbose=False)


# From https://docs.vllm.ai/en/v0.8.5/getting_started/examples/openai_chat_completion_structured_outputs_structural_tag.html
def test_structural_tag():
    """Test 29: Structural tag pattern - JSON schema within specified tags."""
    data = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "messages": [
                    {
                        "role": "user",
                        "content": """You have access to the following function to retrieve the weather in a city:

    {
        "name": "get_weather",
        "parameters": {
            "city": {
                "param_type": "string",
                "description": "The city to get the weather for",
                "required": True
            }
        }
    }

If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function
              argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query

You are a helpful assistant.

Given the previous instructions, what is the weather in New York City, Boston,
and San Francisco?"""
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1,
                "response_format": {
                    "type": "structural_tag",
                    "structures": [
                        {
                            "begin": "<function=get_weather>",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string"
                                    }
                                }
                            },
                            "end": "</function>"
                        }
                    ],
                    "triggers": ["<function="]
                }
            }
        }
    }
    return run_test("Structural Tag (JSON within tags)", data, verbose=False)


def test_regex_single_word():
    """Test 30: Regex pattern for single word."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: multiple words here"
                },
                {
                    "role": "user",
                    "content": "Output a single word"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^[a-z]+$"
                    }
                }
            }
        }
    }
    return run_test("Regex Single Word Pattern", data, verbose=False)


def test_regex_phone_format():
    """Test 31: Regex pattern for phone format."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: 1234567890"
                },
                {
                    "role": "user",
                    "content": "Generate a phone number"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^\\+1-\\d{3}-\\d{3}-\\d{4}$"
                    }
                }
            }
        }
    }
    return run_test("Regex Phone Format (+1-XXX-XXX-XXXX)", data, verbose=False)


def test_regex_uuid_format():
    """Test 32: Regex pattern for UUID format."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "ALWAYS OUTPUT: id: 12345"
                },
                {
                    "role": "user",
                    "content": "Generate a UUID"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
                    }
                }
            }
        }
    }
    return run_test("Regex UUID Format", data, verbose=False)


# Tests with contradictory prompts to verify constraints are actually enforced
def test_json_constraint_contradictory_prompt():
    """Test: JSON constraint with prompt asking for plain text - constraint should win."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "IMPORTANT: You MUST respond with plain text only. Do NOT use JSON format. Write your answer as a simple sentence."
                },
                {
                    "role": "user",
                    "content": "Tell me about a person with a name and age"
                }
            ],
            "sampling_params": {
                "max_tokens": 200,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            },
                            "required": ["name", "age"]
                        }
                    }
                }
            }
        }
    }
    return run_test("JSON Constraint vs Contradictory Prompt", data, verbose=False)


def test_regex_constraint_contradictory_prompt():
    """Test: Regex constraint with prompt asking for different format - constraint should win."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "CRITICAL: You MUST output exactly 'hello world' (two words with a space). Do NOT output anything else."
                },
                {
                    "role": "user",
                    "content": "Output a single word"
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "regex": "^[a-z]+$"
                    }
                }
            }
        }
    }
    return run_test("Regex Constraint vs Contradictory Prompt", data, verbose=False)


def test_choice_constraint_contradictory_prompt():
    """Test: Choice constraint with prompt asking for different answer - constraint should win."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "You MUST answer with 'maybe'. Do NOT say yes or no."
                },
                {
                    "role": "user",
                    "content": "Is the sky blue? Answer yes or no."
                }
            ],
            "sampling_params": {
                "max_tokens": 50,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "choice": ["yes", "no"]
                    }
                }
            }
        }
    }
    return run_test("Choice Constraint vs Contradictory Prompt", data, verbose=False)


def test_grammar_constraint_contradictory_prompt():
    """Test: Grammar constraint with prompt asking for different format - constraint should win."""
    data = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": "You MUST respond in natural language. Write a paragraph explaining SQL, do NOT write actual SQL code."
                },
                {
                    "role": "user",
                    "content": "Write a SQL query to get names from users table"
                }
            ],
            "sampling_params": {
                "max_tokens": 200,
                "temperature": 0.1,
                "extra_body": {
                    "structured_outputs": {
                        "grammar": "root ::= sql_query\nsql_query ::= \"SELECT \" column \" FROM \" table\ncolumn ::= \"name\" | \"id\" | \"email\"\ntable ::= \"users\" | \"products\""
                    }
                }
            }
        }
    }
    return run_test("Grammar Constraint vs Contradictory Prompt", data, verbose=False)


def test_structural_tag_contradictory_prompt():
    """Test: Structural tag constraint with prompt asking for different content within tags - constraint should win."""
    data = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "messages": [
                    {
                        "role": "system",
                        "content": """When responding, use the <function=get_weather> tag.

CRITICAL INSTRUCTION: Inside the <function=get_weather> tag, you MUST put a plain text description like this:
<function=get_weather>The city is New York</function>

Do NOT put JSON inside the tags. Put natural language text only."""
                    },
                    {
                        "role": "user",
                        "content": "What is the weather in New York City?"
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1,
                "response_format": {
                    "type": "structural_tag",
                    "structures": [
                        {
                            "begin": "<function=get_weather>",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string"
                                    }
                                }
                            },
                            "end": "</function>"
                        }
                    ],
                    "triggers": ["<function="]
                }
            }
        }
    }
    return run_test("Structural Tag Constraint vs Contradictory Prompt", data, verbose=False)


def check_result(test_name: str, result: Dict[str, Any], expected_success: bool = True):
    """Check if test result matches expectations. Returns (passed: bool, message: str)."""
    status = result.get("status")
    is_contradictory_test = "contradictory" in test_name.lower()

    # Check for errors in the response
    if status == "FAILED" or "error" in result:
        error_msg = result.get("error", "Unknown error")
        if expected_success:
            return (False, f"❌ {test_name}: FAILED - Got error: {error_msg}")
        else:
            # Check if it's the expected rejection message
            if "deprecated" in str(error_msg).lower() or "guided_json" in str(error_msg).lower():
                return (True, f"✅ {test_name}: PASSED - Correctly rejected deprecated API")
            else:
                return (True, f"⚠️  {test_name}: Got error but not expected type: {error_msg}")

    if status != "COMPLETED":
        return (False, f"⚠️  {test_name}: Unexpected status: {status}")

    if expected_success:
        # Check if output is valid JSON with correct structure
        output = result.get("output", {})

        # Handle RunPod response format - output is a list of batches
        if isinstance(output, list) and len(output) > 0:
            # Get the first batch (or combine all batches)
            batch = output[0] if isinstance(output[0], dict) else {}

            # Check for choices array (native vLLM format or OpenAI format)
            if "choices" in batch:
                choices = batch.get("choices", [])
                if choices and len(choices) > 0:
                    # Check for native vLLM format (tokens)
                    tokens = choices[0].get("tokens", [])
                    if tokens:
                        text = "".join(tokens) if isinstance(tokens, list) else str(tokens)
                        # For structural_tag, check if content is within tags
                        if "structural" in test_name.lower() and "tag" in test_name.lower():
                            if "<function=get_weather>" in text and "</function>" in text:
                                # Extract JSON from between tags
                                start = text.find("<function=get_weather>") + len("<function=get_weather>")
                                end = text.find("</function>")
                                if start < end:
                                    json_text = text[start:end].strip()
                                    try:
                                        parsed = json.loads(json_text)
                                        return (True, f"✅ {test_name}: PASSED - Response contains valid JSON in tags\n   JSON keys: {list(parsed.keys())}")
                                    except json.JSONDecodeError as e:
                                        return (False, f"❌ {test_name}: FAILED - JSON within tags is invalid: {e}\n   Extracted text: {json_text[:200]}...")
                            # Tags missing - constraint not enforced
                            return (False, f"❌ {test_name}: FAILED - Response does not contain <function=get_weather> tags\n   Expected format: <function=get_weather>{{...}}</function>\n   Got: {text[:200]}...")
                        # Validate constraint-specific outputs for native vLLM format
                        if "regex" in test_name.lower() and not is_contradictory_test:
                            import re
                            if "single word" in test_name.lower():
                                if re.match(r"^[a-z]+$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches regex ^[a-z]+$\n   Got: {text.strip()[:50]}...")
                                return (False, f"❌ {test_name}: FAILED - Does not match regex\n   Got: {text[:200]}...")
                            elif "date" in test_name.lower():
                                if re.match(r"^\d{4}-\d{2}-\d{2}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches date format YYYY-MM-DD\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match date format\n   Got: {text[:200]}...")
                            elif "email format" in test_name.lower():
                                if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches email format\n   Got: {text.strip()[:50]}...")
                                return (False, f"❌ {test_name}: FAILED - Does not match email format\n   Got: {text[:200]}...")
                            elif "email" in test_name.lower() and "conflict" in test_name.lower():
                                # Email conflict test uses word-1234 pattern despite email prompt
                                if re.match(r"^[a-z]+-\d{4}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches pattern word-1234 (constraint won over email prompt)\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match pattern word-1234\n   Got: {text[:200]}...")
                            elif "phone" in test_name.lower():
                                if re.match(r"^\+1-\d{3}-\d{3}-\d{4}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches phone format +1-XXX-XXX-XXXX\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match phone format\n   Got: {text[:200]}...")
                            elif "uuid" in test_name.lower():
                                if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches UUID format\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match UUID format\n   Got: {text[:200]}...")
                            elif "1234" in test_name.lower():
                                if re.match(r"^[a-z]+-\d{4}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches pattern word-1234\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match pattern\n   Got: {text[:200]}...")
                        elif "choice" in test_name.lower() and not is_contradictory_test:
                            text_lower = text.strip().lower()
                            if text_lower in ["yes", "no"]:
                                return (True, f"✅ {test_name}: PASSED - Valid choice (yes/no)\n   Got: {text_lower}")
                            return (False, f"❌ {test_name}: FAILED - Not a valid choice\n   Got: {text.strip()}\n   Expected: yes or no")
                        elif "grammar" in test_name.lower() and not is_contradictory_test:
                            if "SELECT" in text.upper() and "FROM" in text.upper():
                                return (True, f"✅ {test_name}: PASSED - Valid SQL query\n   Got: {text[:200]}...")
                            return (False, f"❌ {test_name}: FAILED - Not a valid SQL query\n   Got: {text[:200]}...")
                        # Try to parse as JSON to verify structure
                        try:
                            # Extract JSON from response if it's wrapped
                            if text.strip().startswith("{"):
                                parsed = json.loads(text)
                                return (True, f"✅ {test_name}: PASSED - Got response\n   Response is valid JSON: {list(parsed.keys())}")
                        except:
                            pass
                        return (True, f"✅ {test_name}: PASSED - Got response\n   Response text: {text[:200]}...")
                    # Check for OpenAI format (message.content)
                    message = choices[0].get("message", {})
                    if message and "content" in message:
                        text = message.get("content", "")
                        if text:
                            # For structural_tag, check if content is within tags
                            if "structural" in test_name.lower() and "tag" in test_name.lower():
                                if "<function=get_weather>" in text and "</function>" in text:
                                    # Extract JSON from between tags
                                    start = text.find("<function=get_weather>") + len("<function=get_weather>")
                                    end = text.find("</function>")
                                    if start < end:
                                        json_text = text[start:end].strip()
                                        try:
                                            parsed = json.loads(json_text)
                                            # For contradictory tests, verify constraint overrode prompt
                                            if is_contradictory_test:
                                                return (True, f"✅ {test_name}: PASSED - Constraint enforced (valid JSON within tags despite prompt asking for plain text)\n   JSON keys: {list(parsed.keys())}")
                                            return (True, f"✅ {test_name}: PASSED - Response contains valid JSON in tags\n   JSON keys: {list(parsed.keys())}")
                                        except json.JSONDecodeError as e:
                                            return (False, f"❌ {test_name}: FAILED - JSON within tags is invalid: {e}\n   Extracted text: {json_text[:200]}...")
                                # Tags missing
                                if is_contradictory_test:
                                    # For contradictory test, tags should be present (prompt asks for them), but plain text content would be the failure
                                    return (False, f"❌ {test_name}: FAILED - Model did not output tags (prompt should have encouraged them)\n   Got: {text[:200]}...")
                                return (False, f"❌ {test_name}: FAILED - Response does not contain <function=get_weather> tags\n   Expected format: <function=get_weather>{{...}}</function>\n   Got: {text[:200]}...")
                            # For contradictory tests, verify constraint was enforced
                            if is_contradictory_test:
                                if "json" in test_name.lower():
                                    # Should be JSON despite prompt asking for plain text
                                    if text.strip().startswith("{") or text.strip().startswith("["):
                                        try:
                                            parsed = json.loads(text)
                                            return (True, f"✅ {test_name}: PASSED - Constraint enforced (JSON output despite plain text prompt)\n   JSON keys: {list(parsed.keys())}")
                                        except:
                                            pass
                                    return (False, f"❌ {test_name}: FAILED - Constraint NOT enforced (got plain text instead of JSON)\n   Got: {text[:200]}...")
                                elif "regex" in test_name.lower():
                                    # Should match regex pattern (single word) despite prompt asking for "hello world"
                                    import re
                                    if re.match(r"^[a-z]+$", text.strip()):
                                        return (True, f"✅ {test_name}: PASSED - Constraint enforced (single word despite 'hello world' prompt)\n   Got: {text.strip()}")
                                    return (False, f"❌ {test_name}: FAILED - Constraint NOT enforced (got '{text.strip()}' instead of single word)\n   Expected: single lowercase word matching ^[a-z]+$")
                                elif "choice" in test_name.lower():
                                    # Should be "yes" or "no" despite prompt asking for "maybe"
                                    text_lower = text.strip().lower()
                                    if text_lower in ["yes", "no"]:
                                        return (True, f"✅ {test_name}: PASSED - Constraint enforced ({text_lower} despite 'maybe' prompt)")
                                    return (False, f"❌ {test_name}: FAILED - Constraint NOT enforced (got '{text.strip()}' instead of yes/no)")
                                elif "grammar" in test_name.lower():
                                    # Should be SQL despite prompt asking for natural language
                                    if "SELECT" in text.upper() and "FROM" in text.upper():
                                        return (True, f"✅ {test_name}: PASSED - Constraint enforced (SQL output despite natural language prompt)\n   Got: {text[:200]}...")
                                    return (False, f"❌ {test_name}: FAILED - Constraint NOT enforced (got natural language instead of SQL)\n   Got: {text[:200]}...")
                            # Validate non-contradictory constraint-specific outputs
                            if "regex" in test_name.lower() and not is_contradictory_test:
                                # Validate regex constraint
                                import re
                                if "single word" in test_name.lower():
                                    if re.match(r"^[a-z]+$", text.strip()):
                                        return (True, f"✅ {test_name}: PASSED - Matches regex ^[a-z]+$\n   Got: {text.strip()[:50]}...")
                                    return (False, f"❌ {test_name}: FAILED - Does not match regex ^[a-z]+$\n   Got: {text[:200]}...")
                                elif "date" in test_name.lower():
                                    if re.match(r"^\d{4}-\d{2}-\d{2}$", text.strip()):
                                        return (True, f"✅ {test_name}: PASSED - Matches date format YYYY-MM-DD\n   Got: {text.strip()}")
                                    return (False, f"❌ {test_name}: FAILED - Does not match date format\n   Got: {text[:200]}...")
                            elif "email format" in test_name.lower():
                                if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches email format\n   Got: {text.strip()[:50]}...")
                                return (False, f"❌ {test_name}: FAILED - Does not match email format\n   Got: {text[:200]}...")
                            elif "email" in test_name.lower() and "conflict" in test_name.lower():
                                # Email conflict test uses word-1234 pattern despite email prompt
                                if re.match(r"^[a-z]+-\d{4}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches pattern word-1234 (constraint won over email prompt)\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match pattern word-1234\n   Got: {text[:200]}...")
                            elif "phone" in test_name.lower():
                                if re.match(r"^\+1-\d{3}-\d{3}-\d{4}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches phone format +1-XXX-XXX-XXXX\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match phone format\n   Got: {text[:200]}...")
                            elif "uuid" in test_name.lower():
                                if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches UUID format\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match UUID format\n   Got: {text[:200]}...")
                            elif "1234" in test_name.lower():
                                if re.match(r"^[a-z]+-\d{4}$", text.strip()):
                                    return (True, f"✅ {test_name}: PASSED - Matches pattern word-1234\n   Got: {text.strip()}")
                                return (False, f"❌ {test_name}: FAILED - Does not match pattern word-1234\n   Got: {text[:200]}...")
                            elif "choice" in test_name.lower() and not is_contradictory_test:
                                # Validate choice constraint
                                text_lower = text.strip().lower()
                                if text_lower in ["yes", "no"]:
                                    return (True, f"✅ {test_name}: PASSED - Valid choice (yes/no)\n   Got: {text_lower}")
                                return (False, f"❌ {test_name}: FAILED - Not a valid choice\n   Got: {text.strip()}\n   Expected: yes or no")
                            elif "grammar" in test_name.lower() and not is_contradictory_test:
                                # Validate grammar constraint (SQL)
                                if "SELECT" in text.upper() and "FROM" in text.upper():
                                    return (True, f"✅ {test_name}: PASSED - Valid SQL query\n   Got: {text[:200]}...")
                                return (False, f"❌ {test_name}: FAILED - Not a valid SQL query\n   Got: {text[:200]}...")
                            # Try to parse as JSON
                            try:
                                if text.strip().startswith("{") or text.strip().startswith("["):
                                    parsed = json.loads(text)
                                    return (True, f"✅ {test_name}: PASSED - Got response\n   Response is valid JSON: {list(parsed.keys()) if isinstance(parsed, dict) else 'array'}")
                            except:
                                pass
                            return (True, f"✅ {test_name}: PASSED - Got response\n   Response text: {text[:200]}...")
        # Check if output itself is a dict (single batch)
        elif isinstance(output, dict):
            if "choices" in output:
                choices = output.get("choices", [])
                if choices and len(choices) > 0:
                    # Check for native vLLM format (tokens)
                    tokens = choices[0].get("tokens", [])
                    if tokens:
                        text = "".join(tokens) if isinstance(tokens, list) else str(tokens)
                        # For structural_tag, check if content is within tags
                        if "structural" in test_name.lower() and "tag" in test_name.lower():
                            if "<function=get_weather>" in text and "</function>" in text:
                                # Extract JSON from between tags
                                start = text.find("<function=get_weather>") + len("<function=get_weather>")
                                end = text.find("</function>")
                                if start < end:
                                    json_text = text[start:end].strip()
                                    try:
                                        parsed = json.loads(json_text)
                                        return (True, f"✅ {test_name}: PASSED - Response contains valid JSON in tags\n   JSON keys: {list(parsed.keys())}")
                                    except json.JSONDecodeError as e:
                                        return (False, f"❌ {test_name}: FAILED - JSON within tags is invalid: {e}\n   Extracted text: {json_text[:200]}...")
                            # Tags missing - constraint not enforced
                            return (False, f"❌ {test_name}: FAILED - Response does not contain <function=get_weather> tags\n   Expected format: <function=get_weather>{{...}}</function>\n   Got: {text[:200]}...")
                        try:
                            if text.strip().startswith("{"):
                                parsed = json.loads(text)
                                return (True, f"✅ {test_name}: PASSED - Got response\n   Response is valid JSON: {list(parsed.keys())}")
                        except:
                            pass
                        return (True, f"✅ {test_name}: PASSED - Got response\n   Response text: {text[:200]}...")
                    # Check for OpenAI format (message.content)
                    message = choices[0].get("message", {})
                    if message and "content" in message:
                        text = message.get("content", "")
                        if text:
                            # For structural_tag, check if content is within tags
                            if "structural" in test_name.lower() and "tag" in test_name.lower():
                                if "<function=get_weather>" in text and "</function>" in text:
                                    # Extract JSON from between tags
                                    start = text.find("<function=get_weather>") + len("<function=get_weather>")
                                    end = text.find("</function>")
                                    if start < end:
                                        json_text = text[start:end].strip()
                                        try:
                                            parsed = json.loads(json_text)
                                            return (True, f"✅ {test_name}: PASSED - Response contains valid JSON in tags\n   JSON keys: {list(parsed.keys())}")
                                        except json.JSONDecodeError as e:
                                            return (False, f"❌ {test_name}: FAILED - JSON within tags is invalid: {e}\n   Extracted text: {json_text[:200]}...")
                                # Tags missing - constraint not enforced
                                return (False, f"❌ {test_name}: FAILED - Response does not contain <function=get_weather> tags\n   Expected format: <function=get_weather>{{...}}</function>\n   Got: {text[:200]}...")
                            # Try to parse as JSON
                            try:
                                if text.strip().startswith("{"):
                                    parsed = json.loads(text)
                                    return (True, f"✅ {test_name}: PASSED - Got response\n   Response is valid JSON: {list(parsed.keys())}")
                            except:
                                pass
                            return (True, f"✅ {test_name}: PASSED - Got response\n   Response text: {text[:200]}...")
            # Check if output itself is a string (could be direct response)
            elif isinstance(output, str):
                # For structural_tag, check if content is within tags
                if "structural" in test_name.lower() and "tag" in test_name.lower():
                    if "<function=get_weather>" in output and "</function>" in output:
                        # Extract JSON from between tags
                        start = output.find("<function=get_weather>") + len("<function=get_weather>")
                        end = output.find("</function>")
                        if start < end:
                            json_text = output[start:end].strip()
                            try:
                                parsed = json.loads(json_text)
                                return (True, f"✅ {test_name}: PASSED - Response contains valid JSON in tags\n   JSON keys: {list(parsed.keys())}")
                            except json.JSONDecodeError as e:
                                return (False, f"❌ {test_name}: FAILED - JSON within tags is invalid: {e}\n   Extracted text: {json_text[:200]}...")
                    # Tags missing - constraint not enforced
                    return (False, f"❌ {test_name}: FAILED - Response does not contain <function=get_weather> tags\n   Expected format: <function=get_weather>{{...}}</function>\n   Got: {output[:200]}...")
                return (True, f"✅ {test_name}: PASSED - Got string response\n   Response: {output[:200]}...")

        return (False, f"⚠️  {test_name}: UNEXPECTED FORMAT - {output}")
    else:
        return (False, f"❌ {test_name}: FAILED - Expected error but got success")


def main():
    """Run all tests in parallel batches."""
    # Create output file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.txt"

    # Redirect output to both file and console
    tee = Tee(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("="*60)
        print("vLLM Worker Structured Outputs Test Suite")
        print("="*60)
        print(f"Output file: {output_file}")
        print("="*60)

        # Define all tests with their expected success values
        BATCH_SIZE = 10
        tests = [
            (test_basic_response, "basic", "Basic Response", True),
            (test_structured_outputs_new_api, "structured_outputs", "Structured Outputs (New API)", True),
            (test_guided_json_deprecated, "guided_json_rejected", "Guided JSON Rejection", False),
            (test_structured_outputs_with_yaml_conflict, "yaml_conflict", "YAML Conflict Test", True),
            (test_json_field_order, "field_order", "Field Order Enforcement", True),
            (test_structured_outputs_no_system_message, "no_system_message", "Structured Outputs (No System Message)", True),
            (test_basic_no_structured_outputs, "basic_no_structured", "Basic Response (No Structured Outputs)", True),
            (test_json_array_with_number, "json_array_number", "JSON Array with Number Field", True),
            (test_json_nested_object, "json_nested_object", "JSON Nested Object with Array", True),
            (test_json_enum, "json_enum", "JSON Enum and Number Constraints", True),
            (test_regex_pattern, "regex_pattern", "Regex Pattern (name-1234)", True),
            (test_choice_pattern, "choice_pattern", "Choice Pattern (yes/no)", True),
            (test_grammar_pattern, "grammar_pattern", "Grammar Pattern (SQL Query)", True),
            (test_json_object_pattern, "json_object_pattern", "JSON Object Pattern (No Schema)", True),
            (test_guided_json_outlines, "guided_json_outlines", "Guided JSON with Outlines (Deprecated)", False),
            (test_guided_json_lm_format_enforcer, "guided_json_lmfe", "Guided JSON with LM Format Enforcer (Deprecated)", False),
            (test_json_optional_fields, "json_optional_fields", "JSON with Optional Fields", True),
            (test_json_nested_array, "json_nested_array", "JSON Nested Array of Objects", True),
            (test_json_simple_fields, "json_simple_fields", "JSON Simple Required Fields", True),
            (test_yaml_system_message, "yaml_system_message", "YAML System Message with JSON Output", True),
            (test_regex_email_conflict, "regex_email_conflict", "Regex Pattern with Email Conflict", True),
            (test_grammar_sql_conflict, "grammar_sql_conflict", "Grammar Pattern with SQL Conflict", True),
            (test_json_object_conflict, "json_object_conflict", "JSON Object Pattern with Text Conflict", True),
            (test_json_array_fixed_length, "json_array_fixed_length", "JSON Fixed-Length Array", True),
            (test_json_number_constraints, "json_number_constraints", "JSON Number Constraints", True),
            (test_json_boolean_field, "json_boolean_field", "JSON Boolean Field", True),
            (test_regex_date_format, "regex_date_format", "Regex Date Format (YYYY-MM-DD)", True),
            (test_regex_email_format, "regex_email_format", "Regex Email Format", True),
            (test_structural_tag, "structural_tag", "Structural Tag (JSON within tags)", True),
            (test_regex_single_word, "regex_single_word", "Regex Single Word Pattern", True),
            (test_regex_phone_format, "regex_phone_format", "Regex Phone Format (+1-XXX-XXX-XXXX)", True),
            (test_regex_uuid_format, "regex_uuid_format", "Regex UUID Format", True),
            # Contradictory prompt tests - verify constraints actually override prompts
            (test_json_constraint_contradictory_prompt, "json_constraint_contradictory", "JSON Constraint vs Contradictory Prompt", True),
            (test_regex_constraint_contradictory_prompt, "regex_constraint_contradictory", "Regex Constraint vs Contradictory Prompt", True),
            (test_choice_constraint_contradictory_prompt, "choice_constraint_contradictory", "Choice Constraint vs Contradictory Prompt", True),
            (test_grammar_constraint_contradictory_prompt, "grammar_constraint_contradictory", "Grammar Constraint vs Contradictory Prompt", True),
            (test_structural_tag_contradictory_prompt, "structural_tag_contradictory", "Structural Tag Constraint vs Contradictory Prompt", True),
        ]

        results = {}
        test_results = {}

        # Run tests in parallel batches
        print(f"\nRunning {len(tests)} tests in parallel batches of {BATCH_SIZE}...")
        with tqdm(total=len(tests), desc="Running tests") as pbar:
            for batch_start in range(0, len(tests), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(tests))
                batch_tests = tests[batch_start:batch_end]

                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                    future_to_test = {
                        executor.submit(run_test_wrapper, test_func, test_key): (test_key, test_name, expected_success)
                        for test_func, test_key, test_name, expected_success in batch_tests
                    }

                    for future in as_completed(future_to_test):
                        test_key, test_name, expected_success = future_to_test[future]
                        try:
                            result_key, result = future.result()
                            test_results[result_key] = (result, test_name, expected_success)
                        except Exception as e:
                            test_results[test_key] = ({"error": f"Exception: {str(e)}"}, test_name, expected_success)
                        pbar.update(1)

        # Check all results and collect messages
        result_messages = {}
        for test_key, (result, test_name, expected_success) in test_results.items():
            passed, message = check_result(test_name, result, expected_success)
            results[test_key] = passed
            result_messages[test_key] = message

        # Print all results at once
        print(f"\n{'='*60}")
        print("Checking Results")
        print(f"{'='*60}")
        for test_key, message in result_messages.items():
            print(message)

        # Summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        for test_key, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            test_name = test_results[test_key][1]
            print(f"{status}: {test_name}")

        total = len(results)
        passed = sum(results.values())
        print(f"\nTotal: {passed}/{total} tests passed")
        print(f"\nResults saved to: {output_file}")

    finally:
        # Restore stdout and close file
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
