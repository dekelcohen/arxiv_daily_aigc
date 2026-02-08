import os
import logging
import tempfile
from typing import Dict, Any, Optional

import requests

from .llm_utils import call_llm_by_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _derive_pdf_url(arxiv_entry_url: str) -> Optional[str]:
    try:
        if not arxiv_entry_url:
            return None
        # Typical entry: https://arxiv.org/abs/2401.12345 -> https://arxiv.org/pdf/2401.12345.pdf
        if "/pdf/" in arxiv_entry_url and arxiv_entry_url.endswith(".pdf"):
            return arxiv_entry_url
        pdf_url = arxiv_entry_url.replace("/abs/", "/pdf/")
        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"
        return pdf_url
    except Exception as e:
        logging.error(f"Failed to derive PDF url from '{arxiv_entry_url}': {e}")
        return None


def _download_pdf(pdf_url: str, project_root: str) -> Optional[str]:
    try:
        cache_dir = os.path.join(project_root, "tmp_pdfs")
        os.makedirs(cache_dir, exist_ok=True)
        local_name = os.path.basename(pdf_url)
        local_path = os.path.join(cache_dir, local_name)
        # Simple caching: skip if exists
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        resp = requests.get(pdf_url, timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    except Exception as e:
        logging.error(f"Failed to download PDF '{pdf_url}': {e}")
        return None


def extract_and_summarize(paper: Dict[str, Any],
                          project_root: str,
                          summarization_prompt: Optional[str],
                          model: str) -> Dict[str, Any]:
    """Download the paper PDF and ask the LLM to summarize it.

    If the target model does not support direct file attachments, we include
    the PDF URL and core metadata within the prompt as a fallback.
    """
    title = paper.get("title", "N/A")
    abstract = paper.get("summary", "")
    url = paper.get("url", "")

    pdf_url = _derive_pdf_url(url)
    if not pdf_url:
        logging.warning(f"Could not derive PDF url for '{title}'. Skipping summarization.")
        return paper

    pdf_path = _download_pdf(pdf_url, project_root)

    # Build a robust user message; prefer file attachments for non-Azure models via OpenRouter
    system_text = "You are a helpful research assistant that accurately summarizes scientific papers."
    base_prompt = summarization_prompt or "Summarize the attached paper in 5-7 bullet points. Highlight key contributions and methods."

    # Always include minimal context for models that cannot read attachments
    context_text = (
        f"Paper title: {title}\n"
        f"Abstract (from arXiv): {abstract[:2000]}\n"
        f"PDF URL: {pdf_url}\n"
        "If you cannot read the PDF attachment or link, base the summary on the abstract."
    )

    # For Azure models (azure-*), attachments are not supported via chat/completions in this codebase.
    # We therefore embed the context in the message content.
    if model.startswith("azure-"):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [
                {"type": "text", "text": base_prompt},
                {"type": "text", "text": context_text},
            ]},
        ]
        response = call_llm_by_model(None, model=model, max_tokens=1500, messages=messages, attachments=None)
    else:
        # OpenRouter path: attempt URL-based attachment if supported by the target
        attachments = None
        if pdf_url:
            attachments = [{"type": "file", "url": pdf_url, "mime": "application/pdf"}]
        prompt = base_prompt + "\n\n" + context_text
        response = call_llm_by_model(prompt, model=model, max_tokens=1500, messages=None, attachments=attachments)

    if response:
        paper["llm_summary"] = response.strip()
    else:
        logging.warning(f"Failed to obtain LLM summary for '{title}'.")
    return paper
