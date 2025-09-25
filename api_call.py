import os
import requests
import time
from typing import Dict, Any
import tiktoken

# Azure OpenAI settings
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEPLOYMENT_NAME = "gpt-4o-mini"
API_VERSION = "2023-05-15"
print(f"OPENAI_API_BASE: {OPENAI_API_BASE}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

def format_evaluation(evaluation):
    """Format the evaluation data in a robust manner and extract scores."""
    score = evaluation['evaluation']['score']
    
    if score is None:
        # Attempt to find score in the raw evaluation
        for key, value in evaluation.items():
            if 'score' in key.lower():
                score = extract_score(value)
                break
    
    return {"score": score}

def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def completion_with_backoff_anthropic(**kwargs) -> Dict[str, Any]:
    retry_count = 0
    while True:
        retry_count += 1
        try:
            url = #TODO
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            data = kwargs["messages"] 
            response = requests.post(url, headers=headers, json=data)
            try:
                response.raise_for_status()
            except:
                print(f"Error: {response.text}")
                if retry_count > 3:
                    return {}
                time.sleep(10)
                continue
            return response.json()
        except requests.exceptions.RequestException as error:
            print(f"Error: {error}")
            if hasattr(error, 'response') and error.response is not None:
                print("Response text:", error.response.text)
                if getattr(error.response, 'status_code', None) == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
            if retry_count > 3:
                return {}
            time.sleep(10)

def completion_with_backoff(**kwargs) -> Dict[str, Any]:
    retry_count = 0
    while True:
        retry_count += 1
        try:
            url = f"{OPENAI_API_BASE}/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            data = {
                "messages": kwargs['messages'],
                "max_tokens": kwargs.get('max_tokens', 1000),
                "temperature": kwargs.get('temperature', 0.7)
            }
            response = requests.post(url, headers=headers, json=data)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred: {http_err}")
                print(f"Response text: {response.text}")
                # If it's a 400 error, likely a content policy violation or malformed request, skip this prompt
                if response.status_code == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
                # If it's a 429 error, rate limit, retry
                if response.status_code == 429:
                    print("429 Rate limit hit. Retrying after delay.")
                    time.sleep(10)
                    continue
                # For other errors, retry up to 3 times
                if retry_count > 3:
                    return {}
                time.sleep(10)
                continue
            return response.json()
        except requests.exceptions.RequestException as error:
            print(f"Error: {error}")
            if hasattr(error, 'response') and error.response is not None:
                print("Response text:", error.response.text)
                if getattr(error.response, 'status_code', None) == 400:
                    print("400 error encountered. Skipping this prompt.")
                    return None
            if retry_count > 3:
                return {}
            time.sleep(10)


def extract_score(value):
    """Extract a numeric score from a value, handling various formats."""
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None
