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


def _derive_html_url(arxiv_entry_url: str) -> Optional[str]:
    try:
        if not arxiv_entry_url:
            return None
        # Example: https://arxiv.org/abs/2405.12213v2 -> https://arxiv.org/html/2405.12213v2
        html_url = arxiv_entry_url.replace("/abs/", "/html/")
        return html_url
    except Exception as e:
        logging.error(f"Failed to derive HTML url from '{arxiv_entry_url}': {e}")
        return None


def _fetch_arxiv_html_text(arxiv_entry_url: str) -> Optional[str]:
    """Try fetching the arXiv HTML page and extract main text via trafilatura."""
    html_url = _derive_html_url(arxiv_entry_url)
    if not html_url:
        return None
    try:
        resp = requests.get(html_url, timeout=20)
        if resp.status_code != 200:
            logging.info(f"arXiv HTML not available ({resp.status_code}) at {html_url}")
            return None
        html = resp.text
        try:
            import trafilatura  # lazy import; optional dependency
        except Exception as e:
            logging.warning(f"trafilatura not available: {e}. Falling back to PDF summarization.")
            return None
        text = trafilatura.extract(html)
        if text and len(text.strip()) > 200:
            return text.strip()
        logging.info("trafilatura returned empty/short text; falling back to PDF.")
        return None
    except Exception as e:
        logging.warning(f"Failed fetching/extracting arXiv HTML: {e}")
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
    """Summarize the paper using arXiv HTML text when available, otherwise PDF.

    Preferred path:
    - Fetch arXiv HTML (e.g., https://arxiv.org/html/<id>vN), extract main text via trafilatura,
      and send that text to the summarization prompt.
    Fallback:
    - Download the PDF and attach the URL (OpenRouter) or inline context (Azure).
    """        
    title = paper.get("title", "N/A")
    abstract = paper.get("summary", "")
    url = paper.get("url", "")

    article_text = _fetch_arxiv_html_text(url)

    system_text = "You are a helpful research assistant that accurately summarizes scientific papers."
    base_prompt = (summarization_prompt or "Summarize the article in 5-7 bullet points. Highlight key contributions and methods.")

    # Limit text to avoid excessive token usage
    MAX_INPUT_CHARS = 100000
    MAX_ARTICLE_TOKENS_TEXT_IMAGE = 20000
    MAX_ARTICLE_TOKENS_TEXT = 10000

    if article_text:
        logging.info(f"call llm to summarize paper: '{title}'")
        excerpt = article_text[:MAX_INPUT_CHARS]
        if model.startswith("azure-"):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "text", "text": base_prompt},
                    {"type": "text", "text": f"Paper title: {title}"},
                    {"type": "text", "text": f"Article text:\n{excerpt}"},
                ]},
            ]            
            response = call_llm_by_model(None, model=model, max_tokens=MAX_ARTICLE_TOKENS_TEXT, messages=messages, attachments=None)
        else:
            prompt = (f"{base_prompt}\n\nPaper title: {title}\n\nArticle text:\n{excerpt}")
            response = call_llm_by_model(prompt, model=model, max_tokens=MAX_ARTICLE_TOKENS_TEXT, messages=None, attachments=None)
    else:
        pdf_url = _derive_pdf_url(url)
        if not pdf_url:
            logging.warning(f"Could not derive PDF url for '{title}'. Skipping summarization.")
            return paper
        _download_pdf(pdf_url, project_root)  # best-effort cache; OpenRouter reads via URL
        context_text = (
            f"Paper title: {title}\n"
            f"Abstract (from arXiv): {abstract[:2000]}\n"
            f"PDF URL: {pdf_url}\n"
            "If you cannot read the PDF attachment or link, base the summary on the abstract."
        )
        if model.startswith("azure-"):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "text", "text": base_prompt},
                    {"type": "text", "text": context_text},
                ]},
            ]
            response = call_llm_by_model(None, model=model, max_tokens=MAX_ARTICLE_TOKENS_TEXT_IMAGE, messages=messages, attachments=None)
        else:
            attachments = [{"type": "file", "url": pdf_url, "mime": "application/pdf"}]
            prompt = base_prompt + "\n\n" + context_text
            response = call_llm_by_model(prompt, model=model, max_tokens=MAX_ARTICLE_TOKENS_TEXT_IMAGE, messages=None, attachments=attachments)

    if response:
        paper["llm_summary"] = response.strip()
    else:
        logging.warning(f"Failed to obtain LLM summary for '{title}'.")
    return paper
