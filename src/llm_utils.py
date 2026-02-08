import os
import logging
import requests
from typing import Optional, List, Dict, Any

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenRouter configuration (used when model is NOT azure-...)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default small model (kept for backward compatibility with filter.py)
MODEL_NAME = "google/gemini-2.0-flash-001"


def _build_openrouter_messages(prompt: Optional[str] = None,
                               messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if messages is not None:
        return messages
    assert prompt is not None, "Either prompt or messages must be provided"
    return [{"role": "user", "content": prompt}]


def call_openrouter_api(prompt: Optional[str] = None,
                        model: str = MODEL_NAME,
                        max_tokens: int = 256,
                        messages: Optional[List[Dict[str, Any]]] = None,
                        attachments: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Call OpenRouter Chat Completions API.

    Notes:
    - If `attachments` is provided, it is passed through as `attachments` on the
      first user message. OpenRouter supports URL-based attachments for some models.
    - Returns the assistant content string or None on error.
    """
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set. Cannot call OpenRouter API.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    msgs = _build_openrouter_messages(prompt, messages)
    if attachments and msgs:
        try:
            # Attach to the first user message if possible
            if msgs[0].get("role") == "user":
                msgs[0]["attachments"] = attachments
        except Exception:
            # Non-fatal; continue without attachments
            logging.warning("Unable to attach files to OpenRouter message; proceeding without attachments.")

    data = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        ai_response = result['choices'][0]['message']['content'].strip()
        return ai_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling OpenRouter API: {e}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"Error parsing OpenRouter response: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error calling OpenRouter: {e}", exc_info=True)
        return None


def call_llm_by_model(prompt: Optional[str],
                      model: str,
                      max_tokens: int,
                      messages: Optional[List[Dict[str, Any]]] = None,
                      attachments: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Dispatch LLM call by model name.

    - If model starts with "azure-", use Azure OpenAI via azure_openai.call_llm.
    - Otherwise, use OpenRouter with the given model id.

    Accepts either `prompt` or full `messages`.
    """
    if model.startswith("azure-"):
        try:
            # Relative import inside function to avoid circular imports
            from src.azure_openai import call_llm as azure_call_llm
        except Exception as e:
            logging.error(f"Failed to import Azure helper: {e}")
            return None
        deployment = model[len("azure-"):]
        try:
            if messages is None:
                messages = [{"role": "user", "content": prompt or ""}]
            # Azure API currently does not natively support arbitrary PDF attachments via chat/completions.
            # We pass messages as-is. If attachments are provided, callers should embed relevant text/links
            # into the messages (handled by the caller).
            return azure_call_llm(messages, azure_deployment_model=deployment, max_tokens=max_tokens)
        except Exception as e:
            logging.error(f"Azure OpenAI call failed: {e}", exc_info=True)
            return None
    else:
        return call_openrouter_api(prompt=prompt, model=model, max_tokens=max_tokens, messages=messages, attachments=attachments)

