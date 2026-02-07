import os
import requests
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenRouter configuration (only used when model is NOT azure-...)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default language model (OpenRouter model id). Adjust via CLI.
MODEL_NAME = "google/gemini-2.0-flash-001"

def call_openrouter_api(prompt: str, model: str = MODEL_NAME, max_tokens: int = 5) -> str | None:
    """Call OpenRouter API and return the model's response.

    Args:
        prompt: Text prompt to send.
        model: OpenRouter model id.
        max_tokens: Max tokens in the response.

    Returns:
        The response text, or None on error.
    """
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY is not set. Cannot call OpenRouter API.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
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


def call_llm_by_model(prompt: str, model: str, max_tokens: int) -> str | None:
    """Dispatch LLM call by model name.

    - If model starts with "azure-", use Azure OpenAI via azure_openai.call_llm.
    - Otherwise, use OpenRouter with the given model id.
    """
    if model.startswith("azure-"):
        try:
            from azure_openai import call_llm
        except Exception as e:
            logging.error(f"Failed to import Azure helper: {e}")
            return None
        deployment = model[len("azure-"):]
        # Build Azure messages and call non-streaming endpoint
        messages = [{"role": "user", "content": prompt}]
        try:
            # Use provided max_tokens (can be large for gpt-5 thinking if desired)            
            max_tokens+=10000
            new_output = call_llm(messages, azure_deployment_model=deployment, max_tokens=max_tokens)
            return new_output
        except Exception as e:
            logging.error(f"Azure OpenAI call failed: {e}", exc_info=True)
            return None
    else:
        return call_openrouter_api(prompt, model=model, max_tokens=max_tokens)


def filter_papers_by_topic(papers: list, topic: str = "image or video or multimodal generation", model: str | None = None) -> list:
    """Filter papers by topic using an LLM.

    Args:
        papers: List of paper dicts with 'title' and 'summary'.
        topic: Topic keyword(s) to filter on.
        model: Language model name. If None, uses module-level MODEL_NAME.

    Returns:
        List of papers judged relevant to the topic.
    """
    if model is None:
        model = MODEL_NAME

    filtered_papers = []
    logging.info(f"Begin filtering {len(papers)} papers by topic: '{topic}' using model '{model}'.")

    for i, paper in enumerate(papers):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')

        prompt = (
            f"Is the following paper primarily about '{topic}'? "
            f"Answer with only 'yes' or 'no'.\n\nTitle: {title}\nAbstract: {summary}"
        )

        ai_response = call_llm_by_model(prompt, model=model, max_tokens=5)

        if ai_response is not None:
            logging.info(f"Paper {i+1}/{len(papers)}: '{title[:50]}...' - LLM reply: {ai_response}")
            if 'yes' in ai_response.lower():
                filtered_papers.append(paper)
        else:
            logging.warning(f"No LLM reply for paper '{title[:50]}...'; skipping.")
            continue

    logging.info(f"Filtering complete. Found {len(filtered_papers)} papers relevant to '{topic}'.")
    return filtered_papers


rating_prompt_template = """
# Role Setting
You are an experienced researcher in the field of Artificial Intelligence, skilled at quickly evaluating the potential value of research papers.

# Task
Based on the following paper's title and abstract, please summarize it and score it across multiple dimensions (1-10 points, 1 being the lowest, 10 being the highest). Finally, provide an overall preliminary priority score.

# Input
Paper Title: %s
Paper Abstract: %s

# My Research Interests
image generation, video generation, multimodal generation

# Output Requirements
Output should always be in JSON format, strictly compliant with RFC8259.
Please output the evaluation and explanations in the following JSON format:
{
  "tldr": "<summary>", // Too Long; Didn't Read. Summarize the paper in one or two brief sentences.
  "tldr_zh": "<summary>", // Too Long; Didn't Read. Summarize the paper in one or two brief sentences, in Chinese.
  "relevance_score": <score>, // Relevance to my research interests
  "novelty_claim_score": <score>, // Degree of novelty claimed in the abstract
  "clarity_score": <score>, // Clarity and completeness of the abstract writing
  "potential_impact_score": <score>, // Estimated potential impact based on abstract claims
  "overall_priority_score": <score> // Preliminary reading priority score combining all factors above
}

# Scoring Guidelines
- Relevance: Focus on whether it is directly related to the research interests I provided.
- Novelty: Evaluate the degree of innovation claimed in the abstract regarding the method or viewpoint compared to known work.
- Clarity: Evaluate whether the abstract itself is easy to understand and complete with essential elements.
- Potential Impact: Evaluate the importance of the problem it claims to solve and the potential application value of the results.
- Overall Priority: Provide an overall score combining all the above factors. A high score indicates suggested priority for reading.
"""


def rate_papers(papers: list, model: str | None = None) -> list:
    """Score papers using an LLM and return updated dicts.

    Args:
        papers: List of paper dicts.
        model: Language model name. If None, uses module-level MODEL_NAME.

    Returns:
        List of papers augmented with rating fields.
    """
    if model is None:
        model = MODEL_NAME

    logging.info(f"Begin scoring {len(papers)} papers using model '{model}'...")
    for i, paper in enumerate(papers):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')
        prompt = rating_prompt_template % (title, summary)

        success = False
        for attempt in range(2):
            ai_response = call_llm_by_model(prompt, model=model, max_tokens=1000)

            if ai_response is not None:
                try:
                    if '```json' in ai_response:
                        ai_response = ai_response.split('```json')[1].split('```')[0]
                    rating_data = json.loads(ai_response)
                    logging.info(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): '{title[:50]}...' - LLM rating parsed.")
                    papers[i].update(rating_data)
                    success = True
                    break
                except json.JSONDecodeError:
                    logging.warning(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): '{title[:50]}...' - LLM reply is not valid JSON: {ai_response[:100]}...")
                except Exception as e:
                    logging.error(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): '{title[:50]}...' - Error parsing reply: {e}", exc_info=True)
            else:
                logging.warning(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): Failed to get LLM rating (None).")

            if attempt < 1:
                logging.info(f"Paper {i+1}/{len(papers)}: retrying...")

        if not success:
            logging.error(f"Paper {i+1}/{len(papers)}: failed twice to obtain rating for '{title[:50]}...'; skipping.")
            continue

    logging.info("Scoring complete.")
    return papers


# Optional local test (requires keys when using OpenRouter/Azure)
if __name__ == '__main__':
    test_papers = [
        {
            'title': 'Generative Adversarial Networks for Image Synthesis ',
            'summary': 'This paper introduces GANs, a framework for estimating generative models via an adversarial process... focusing on image creation.'
        },
        {
            'title': 'Deep Learning for Natural Language Processing',
            'summary': 'We explore various deep learning architectures like RNNs and Transformers for tasks such as machine translation and sentiment analysis.'
        },
        {
            'title': 'Video Generation using Diffusion Models',
            'summary': 'A novel approach to generating high-fidelity video sequences using latent diffusion models.'
        }
    ]
    filtered = filter_papers_by_topic(test_papers, model=MODEL_NAME)
    rated = rate_papers(filtered, model=MODEL_NAME)
