import os
import json
import logging
import argparse
from datetime import date, datetime, timedelta

# Ensure 'src' is in the Python path to import other modules
from .scraper import fetch_papers
from .filter import filter_papers_by_topic, rate_papers, MODEL_NAME as DEFAULT_MODEL_NAME
from .config import Config
from .html_generator import generate_html_from_json
from src.extract_summarize import extract_and_summarize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define default directories
DEFAULT_JSON_DIR = os.path.join(PROJECT_ROOT, 'daily_json')
DEFAULT_HTML_DIR = os.path.join(PROJECT_ROOT, 'daily_html')
DEFAULT_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
DEFAULT_TEMPLATE_NAME = 'paper_template.html'  # Ensure this template exists

def main(target_date: date, provider_feed: str = "cs.CV", model: str = DEFAULT_MODEL_NAME, large_model: str | None = None, config: Config | None = None, filter_prompt_override: str | None = None):
    """Main pipeline: fetch, filter, summarize, save, generate HTML."""
    logging.info(f"Starting processing for date: {target_date.isoformat()} (model={model}, provider_feed={provider_feed})")

    # --- Determine JSON file path (per category) ---
    json_filename = f"{target_date.isoformat()}.json"
    json_dir_for_provider = os.path.join(DEFAULT_JSON_DIR, provider_feed)
    json_filepath = os.path.join(json_dir_for_provider, json_filename)
    logging.info(f"Target JSON file path: {json_filepath}")

    # --- Check if the JSON file exists ---
    if os.path.exists(json_filepath):
        logging.info(f"Found existing JSON file: {json_filepath}. Skipping fetch and filter steps.")
        # No need to load data; generate_html_from_json reads the file directly
    else:
        logging.info(f"JSON file not found: {json_filepath}. Performing fetch and filter.")
        # --- 1. Fetch papers --- #
        logging.info(f"Step 1: Fetch arXiv {provider_feed} papers...")
        # Note: fetch_cv_papers uses UTC dates by default
        raw_papers = fetch_papers(provider_feed=provider_feed, specified_date=target_date)
        if not raw_papers:
            logging.warning(f"No papers found or fetch failed on {target_date.isoformat()}.")
            return
        logging.info(f"Fetched {len(raw_papers)} raw papers.")
        topic_for_filter = (filter_prompt_override if filter_prompt_override else (config.filter_prompt if config and config.filter_prompt else "general image/video/multimodal generation or image/video editing"))

        # --- 2. Filter and score papers --- #
        logging.info("Step 2: Use AI to filter and score papers (topic: image/video/multimodal generation)...")
        filtered_papers = filter_papers_by_topic(
            raw_papers,
            topic=topic_for_filter,
            model=model,
        )
        filtered_papers = rate_papers(filtered_papers, model=model)
        # Sort filtered_papers by overall_priority_score (descending)
        filtered_papers.sort(key=lambda x: x.get('overall_priority_score', 0), reverse=True)
        if not filtered_papers:
            logging.warning("No papers passed the filter. Creating an empty JSON file.")
            filtered_papers = []
            logging.info(f"After filtering, {len(filtered_papers)} papers remain.")

        # --- 2.5 Summarize with large model (PDF attachment) --- #
        summarization_model = (large_model if large_model else model)
        sum_prompt = (config.summarization_prompt if config and config.summarization_prompt else "Summarize the attached paper succinctly.")
        logging.info(f"Step 2.5: Summarize {len(filtered_papers)} papers using model '{summarization_model}'.")
        for idx, p in enumerate(filtered_papers):
            try:
                filtered_papers[idx] = extract_and_summarize(p, PROJECT_ROOT, sum_prompt, summarization_model)
            except Exception as e:
                logging.error(f"Summarization failed for paper {idx+1}: {e}", exc_info=True)

        # --- 3. Save as JSON --- #
        logging.info("Step 3: Save filtered papers as a JSON file...")

        # --- 3.1 Convert dates to strings --- #
        logging.info("Step 3.1: Convert datetime objects to ISO strings for JSON serialization...")
        for paper in filtered_papers:
            if isinstance(paper.get('published_date'), datetime):
                paper['published_date'] = paper['published_date'].isoformat()
            if isinstance(paper.get('updated_date'), datetime):
                paper['updated_date'] = paper['updated_date'].isoformat()

        os.makedirs(json_dir_for_provider, exist_ok=True)  # Ensure category directory exists
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(filtered_papers, f, indent=4, ensure_ascii=False)
            logging.info(f"Filtered papers saved to: {json_filepath}")
        except IOError as e:
            logging.error(f"Failed to save JSON file: {e}")
            return # Cannot continue if saving failed
        except Exception as e:
            logging.error(f"Unexpected error while saving JSON: {e}", exc_info=True)
            return

    # --- 4. Generate HTML (whether JSON is new or existing) --- #
    logging.info("Step 4: Generate HTML report from JSON file...")
    # Double-check the JSON file actually exists (just in case)
    if not os.path.exists(json_filepath):
        logging.error(f"Cannot find JSON file '{json_filepath}' to generate HTML.")
        return

    html_dir_for_feed = os.path.join(DEFAULT_HTML_DIR, provider_feed)
    os.makedirs(html_dir_for_feed, exist_ok=True)
    try:
        generate_html_from_json(
            json_file_path=json_filepath,
            template_dir=DEFAULT_TEMPLATE_DIR,
            template_name=DEFAULT_TEMPLATE_NAME,
            output_dir=html_dir_for_feed
        )
        logging.info(f"HTML report generated in: {html_dir_for_feed}")

        # --- 5. Update reports.json --- #
        logging.info("Step 5: Update reports.json in the project root...")
        reports_json_path = os.path.join(PROJECT_ROOT, 'reports.json')
        try:
            if os.path.exists(html_dir_for_feed) and os.path.isdir(html_dir_for_feed):
                html_files = [os.path.join(provider_feed, f) for f in os.listdir(html_dir_for_feed) if f.endswith('.html')]
                # Sort by filename (date) descending
                html_files.sort(reverse=True)
                with open(reports_json_path, 'w', encoding='utf-8') as f:
                    json.dump(html_files, f, indent=4, ensure_ascii=False)
                logging.info(f"reports.json updated, contains {len(html_files)} reports.")
            else:
                logging.warning(f"HTML directory '{html_dir_for_feed}' does not exist; cannot generate reports.json.")
                # If the directory does not exist, create an empty reports.json
                with open(reports_json_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=4, ensure_ascii=False)
                logging.info("Created empty reports.json.")
        except Exception as e:
            logging.error(f"Error updating reports.json: {e}", exc_info=True)

    except FileNotFoundError:
        logging.error(f"Template file '{DEFAULT_TEMPLATE_NAME}' not found in '{DEFAULT_TEMPLATE_DIR}'.")
    except Exception as e:
        logging.error(f"Unexpected error while generating HTML: {e}", exc_info=True)

    logging.info(f"Processing complete for date {target_date.isoformat()}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch, filter, and generate a daily report for arXiv papers.')
    parser.add_argument(
        '--date',
        type=str,
        help="Specify the date to fetch (YYYY-MM-DD). If omitted, uses today's UTC date."
    )
    parser.add_argument(
        '--provider-feed',
        type=str,
        default='cs.CV',
        help='Provider feed/category to fetch (e.g., cs.CV). Default: cs.CV.'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Name of config folder (e.g., robotics_vlm_vla). Reads prompts and provider feed from config/<name>.'
    )
    parser.add_argument(
        '--filter-prompt',
        type=str,
        help='Override filter prompt used to select relevant papers.'
    )

    parser.add_argument(
        '--small-language-model', '-small-lm',
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=("Language model to use for filtering/scoring. For Azure, prefix with 'azure-' followed by the deployment name; "
              "for OpenRouter, provide the model id (e.g., 'google/gemini-2.0-flash-001').")
    )
    parser.add_argument(
        '--large-language-model', '-large-lm',
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=("Language model to use for PDF summarization. Defaults to the small model if not provided.")
    )

    args = parser.parse_args()

    run_date = None

    # Initialize config if provided
    config = None
    if args.config:
        try:
            config = Config(PROJECT_ROOT, args.config, category_default=args.provider_feed)
            logging.info(f"Loaded config from config/{args.config} (provider_feed={config.provider_feed})")
        except Exception as e:
            logging.error(f"Failed to initialize config '{args.config}': {e}", exc_info=True)
            config = None

    run_date = None
    if args.date:
        try:
            run_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            logging.info(f"Using user-specified date: {run_date.isoformat()}")
        except ValueError:
            logging.error("Invalid date format; use YYYY-MM-DD. Exiting.")
            exit(1)
    else:
        run_date = date.today()
        logging.info(f"No date specified; using default date: {run_date.isoformat()}")

    if not os.path.exists(DEFAULT_TEMPLATE_DIR) or not os.path.exists(os.path.join(DEFAULT_TEMPLATE_DIR, DEFAULT_TEMPLATE_NAME)):
        logging.warning(f"Template directory '{DEFAULT_TEMPLATE_DIR}' or template file '{DEFAULT_TEMPLATE_NAME}' does not exist. HTML generation may fail.")

    # Generate reports for the past two days and today
    main(target_date=run_date - timedelta(days=2), provider_feed=args.provider_feed, model=args.small_language_model, large_model=args.large_language_model, config=config, filter_prompt_override=args.filter_prompt)
    main(target_date=run_date - timedelta(days=1), provider_feed=args.provider_feed, model=args.small_language_model, large_model=args.large_language_model, config=config, filter_prompt_override=args.filter_prompt)
    main(target_date=run_date, provider_feed=args.provider_feed, model=args.small_language_model, large_model=args.large_language_model, config=config, filter_prompt_override=args.filter_prompt)

