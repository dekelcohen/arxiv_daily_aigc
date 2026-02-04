import os
import json
import logging
import argparse
from datetime import date, datetime, timedelta

# Ensure 'src' is in the Python path to import other modules
# Usually handled when running scripts, or via PYTHONPATH
# Prefer relative imports (if structure allows) or install the project as a package
from scraper import fetch_cv_papers
from filter import filter_papers_by_topic, rate_papers
from html_generator import generate_html_from_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define default directories
DEFAULT_JSON_DIR = os.path.join(PROJECT_ROOT, 'daily_json')
DEFAULT_HTML_DIR = os.path.join(PROJECT_ROOT, 'daily_html')
DEFAULT_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
DEFAULT_TEMPLATE_NAME = 'paper_template.html' # Ensure this template exists

def main(target_date: date, category: str = "cs.CV"):
    """Main pipeline: fetch, filter, save, generate HTML."""
    logging.info(f"Starting processing for date: {target_date.isoformat()}")

    # --- Determine JSON file path (per category) ---
    json_filename = f"{target_date.isoformat()}.json"
    json_dir_for_category = os.path.join(DEFAULT_JSON_DIR, category)
    json_filepath = os.path.join(json_dir_for_category, json_filename)
    logging.info(f"Target JSON file path: {json_filepath}")

    # --- Check if the JSON file exists ---
    if os.path.exists(json_filepath):
        logging.info(f"Found existing JSON file: {json_filepath}. Skipping fetch and filter steps.")
        # No need to load data; generate_html_from_json reads the file directly
    else:
        logging.info(f"JSON file not found: {json_filepath}. Performing fetch and filter.")
        # --- 1. Fetch papers --- #
        logging.info(f"Step 1: Fetch arXiv {category} papers...")
        # Note: fetch_cv_papers uses UTC dates by default
        raw_papers = fetch_cv_papers(category=category, specified_date=target_date)
        if not raw_papers:
            logging.warning(f"No papers found or fetch failed on {target_date.isoformat()}.")
            # If fetching fails and no JSON exists, cannot continue
            return
        logging.info(f"Fetched {len(raw_papers)} raw papers.")

        # --- 2. Filter and score papers --- #
        logging.info("Step 2: Use AI to filter and score papers (topic: image/video/multimodal generation)...")
        # Note: filter_papers_by_topic depends on the OPENROUTER_API_KEY environment variable
        filtered_papers = filter_papers_by_topic(raw_papers, topic="general image/video/multimodal generation or image/video editing")
        filtered_papers = rate_papers(filtered_papers)
        # Sort filtered_papers by overall_priority_score (descending)
        filtered_papers.sort(key=lambda x: x.get('overall_priority_score', 0), reverse=True)
        if not filtered_papers:
            logging.warning("No papers passed the filter. Creating an empty JSON file.")
            # Create an empty list so we can save an empty JSON
            filtered_papers = []
            # Even with no filtered papers, we may generate an empty report or stop here
            # We choose to continue and generate a possibly empty report
            logging.info(f"After filtering, {len(filtered_papers)} papers remain.")

        # --- 3. Save as JSON --- #
        logging.info("Step 3: Save filtered papers as a JSON file...")

        # --- 3.1 Convert dates to strings --- #
        logging.info("Step 3.1: Convert datetime objects to ISO strings for JSON serialization...")
        for paper in filtered_papers:
            if isinstance(paper.get('published_date'), datetime):
                paper['published_date'] = paper['published_date'].isoformat()
            if isinstance(paper.get('updated_date'), datetime):
                paper['updated_date'] = paper['updated_date'].isoformat()

        os.makedirs(json_dir_for_category, exist_ok=True)  # Ensure category directory exists
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

    try:
        generate_html_from_json(
            json_file_path=json_filepath,
            template_dir=DEFAULT_TEMPLATE_DIR,
            template_name=DEFAULT_TEMPLATE_NAME,
            output_dir=DEFAULT_HTML_DIR
        )
        logging.info(f"HTML report generated in: {DEFAULT_HTML_DIR}")

        # --- 5. Update reports.json --- #
        logging.info("Step 5: Update reports.json in the project root...")
        reports_json_path = os.path.join(PROJECT_ROOT, 'reports.json')
        try:
            if os.path.exists(DEFAULT_HTML_DIR) and os.path.isdir(DEFAULT_HTML_DIR):
                html_files = [f for f in os.listdir(DEFAULT_HTML_DIR) if f.endswith('.html')]
                # Sort by filename (date) descending
                html_files.sort(reverse=True)
                with open(reports_json_path, 'w', encoding='utf-8') as f:
                    json.dump(html_files, f, indent=4, ensure_ascii=False)
                logging.info(f"reports.json updated, contains {len(html_files)} reports.")
            else:
                logging.warning(f"HTML directory '{DEFAULT_HTML_DIR}' does not exist; cannot generate reports.json.")
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
    parser = argparse.ArgumentParser(description='Fetch, filter, and generate a daily report for arXiv cs.CV papers.')
    parser.add_argument(
        '--date',
        type=str,
        help='Specify the date to fetch (YYYY-MM-DD). If omitted, uses today\'s UTC date.'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='cs.CV',
        help='arXiv category to fetch (e.g., cs.CV). Default: cs.CV.'
    )


    args = parser.parse_args()

    run_date = None
    if args.date:
        try:
            run_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            logging.info(f"Using user-specified date: {run_date.isoformat()}")
        except ValueError:
            logging.error("Invalid date format; use YYYY-MM-DD. Exiting.")
            exit(1)
    else:
        # If no date is provided, use the scraper's default logic (today's UTC)
        # For consistency, compute the default date here and pass it to main
        run_date = date.today()
        logging.info(f"No date specified; using default date: {run_date.isoformat()}")
        # Alternatively, let fetch_cv_papers handle None to use its default UTC date
        # run_date = None  # Uncomment to use fetch_cv_papers' default date logic

    # Ensure the template directory and file exist; otherwise HTML generation may fail
    if not os.path.exists(DEFAULT_TEMPLATE_DIR) or not os.path.exists(os.path.join(DEFAULT_TEMPLATE_DIR, DEFAULT_TEMPLATE_NAME)):
        logging.warning(f"Template directory '{DEFAULT_TEMPLATE_DIR}' or template file '{DEFAULT_TEMPLATE_NAME}' does not exist. HTML generation may fail.")
        # Consider creating a default template here or exit

    # Check the past two days to avoid gaps, and generate today's report
    main(target_date=run_date - timedelta(days=2), category=args.category)
    main(target_date=run_date - timedelta(days=1), category=args.category)
    main(target_date=run_date, category=args.category)
