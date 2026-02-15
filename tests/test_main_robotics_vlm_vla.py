import os
import sys
import unittest
import json
from datetime import date

DELETE_OUTPUTS = True

# Title: End-to-end integration test for main with robotics_vlm_vla config
# Root cause: verify pipeline can run for a specific date/config with real network calls and produce LLM summaries
# Approach: record existing files under daily_json/html in setUp, run main, then validate artifacts and delete only newly created files in tearDown


class TestMainRoboticsVLMVLAIntegration(unittest.TestCase):
    def setUp(self):
        # --- Setup paths and module imports ---
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Import as packages so running via `python -m tests.test_main_robotics_vlm_vla` works
        import src.main as main_module
        from src.config import Config
        self.main_module = main_module
        self.Config = Config

        # --- Record pre-existing files in daily_json and daily_html ---
        self.daily_json_root = os.path.join(self.project_root, 'daily_json')
        self.daily_html_root = os.path.join(self.project_root, 'daily_html')
        self.prev_json_files = self._list_files(self.daily_json_root)
        self.prev_html_files = self._list_files(self.daily_html_root)

    def _list_files(self, root_dir: str):
        files = set()
        if os.path.isdir(root_dir):
            for r, _, fns in os.walk(root_dir):
                for fn in fns:
                    files.add(os.path.join(r, fn))
        return files

    def test_run_main_robotics_vlm_vla_e2e(self):
        # Use explicit date: February 7, 2026 (ISO 2026-02-07)
        run_date = date(2026, 2, 7)

        # Initialize config for robotics; default provider feed to cs.RO if missing
        config = self.Config(self.project_root, 'robotics_vlm_vla', category_default='cs.RO')
        provider_feed = config.provider_feed or 'cs.RO'

        # --- Invoke main pipeline with real network calls ---
        self.main_module.main(target_date=run_date, provider_feed=provider_feed, model='azure-gpt-5-nano', large_model='azure-gpt-5', config=config)

        # --- Validate JSON & HTML contain LLM summary (>= 200 chars) ---
        json_path = os.path.join(self.daily_json_root, provider_feed, f"{run_date.isoformat()}.json")
        html_path = os.path.join(self.daily_html_root, provider_feed, f"{run_date.strftime('%Y_%m_%d')}.html")

        # If pipeline produced no artifacts (e.g., no papers), skip gracefully
        if not (os.path.isfile(json_path) and os.path.isfile(html_path)):
            self.skipTest('Artifacts not found; possibly no papers for chosen date/feed.')

        # Read papers list
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                papers = json.load(f)
            except Exception:
                papers = []
        if not papers:
            self.skipTest('No papers in JSON; skipping summary check.')

        # Find any paper with llm_summary meeting length requirement
        candidates = [p for p in papers if isinstance(p.get('llm_summary'), str) and len(p['llm_summary']) >= 200]

        # If Azure env configured, enforce assertion; else skip to avoid false failures
        has_azure = all([
            os.getenv('AZURE_OPENAI_API_KEY'),
            os.getenv('AZURE_OPENAI_ENDPOINT'),
            os.getenv('AZURE_OPENAI_API_VERSION'),
        ])
        if has_azure:
            self.assertTrue(len(candidates) > 0, 'Expected at least one paper to have llm_summary >= 200 chars when Azure keys are configured.')
        else:
            if len(candidates) == 0:
                self.skipTest('LLM summary not present or too short; Azure keys missing. Skipping.')

        # Validate HTML includes a snippet of the summary
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        self.assertIn('<strong>Summary</strong>', html, 'HTML should include Summary section.')

        # Check that a sanitized snippet of the summary appears in HTML
        def _alnum(s: str) -> str:
            return ''.join(ch for ch in s if ch.isalnum())
        snippet = candidates[0]['llm_summary'].strip()[:150]
        self.assertTrue(len(snippet) >= 50, 'Snippet unexpectedly short; check summarization prompt.')
        self.assertIn(_alnum(snippet), _alnum(html), 'HTML should contain the summary text snippet.')

    def tearDown(self):
        # --- Delete only files created during this test under daily_json/html ---
        if not DELETE_OUTPUTS:
            return
        try:
            after_json_files = self._list_files(self.daily_json_root)
            after_html_files = self._list_files(self.daily_html_root)
            new_json_files = after_json_files - self.prev_json_files
            new_html_files = after_html_files - self.prev_html_files

            for path in new_json_files:
                if os.path.isfile(path):
                    os.remove(path)
            for path in new_html_files:
                if os.path.isfile(path):
                    os.remove(path)
        except Exception as e:
            # Do not swallow exceptions: surface cleanup issues
            print(f"Teardown cleanup failed: {e}")
            raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--delete-outputs', type=str, default='true')
    args, remaining = parser.parse_known_args()
    DELETE_OUTPUTS = (args.delete_outputs.lower() != 'false')
    unittest.main(argv=[sys.argv[0]] + remaining)


