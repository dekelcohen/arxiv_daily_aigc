import os
import sys
import unittest
from datetime import date

# Title: End-to-end integration test for main with robotics_vlm_vla config
# Root cause: verify pipeline can run for a specific date/config with real network calls
# Approach: record existing files under daily_json/html in setUp, run main, then delete only newly created files in tearDown


class TestMainRoboticsVLMVLAIntegration(unittest.TestCase):
    def setUp(self):
        # --- Setup paths and module imports ---
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        #src_path = os.path.join(self.project_root, 'src')
        #if src_path not in sys.path:
        #    sys.path.insert(0, src_path)
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
        # Use explicit date: February 8, 2026 (ISO 2026-02-08), per request 08/02/2026
        run_date = date(2026, 2, 7)

        # Initialize config for robotics; default provider feed to cs.RO if missing
        config = self.Config(self.project_root, 'robotics_vlm_vla', category_default='cs.RO')
        provider_feed = config.provider_feed or 'cs.RO'

        # --- Invoke main pipeline with real network calls ---
        # main() will fetch, filter/score (LLM may be skipped if no keys), save JSON, generate HTML, update reports.json
        self.main_module.main(target_date=run_date, provider_feed=provider_feed, config=config)

        # No strict assertions on outputs; focus on cleanup of newly created artifacts

    def tearDown(self):
        # --- Delete only files created during this test under daily_json/html ---
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
    unittest.main()

