import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Lightweight config loader bound to a folder under project root.

    Reads text prompts and a simple provider feed YAML. Creates sane defaults
    if files are missing to keep the pipeline usable out-of-the-box.
    """

    def __init__(self, project_root: str, name: str, category_default: str = "cs.CV") -> None:
        if not name:
            raise ValueError("Config name must be a non-empty string")
        self.project_root = project_root
        self.folder = os.path.join(project_root, 'config', name)
        self.filter_prompt: Optional[str] = None
        self.summarization_prompt: Optional[str] = None
        self.provider_feed: Optional[str] = None  # e.g., 'cs.RO'

        self._ensure_defaults(category_default)
        self._load_all()

    # --- Defaults creation ---
    def _ensure_defaults(self, category_default: str) -> None:
        os.makedirs(self.folder, exist_ok=True)

        filter_prompt_path = os.path.join(self.folder, 'filter_prompt.txt')
        if not os.path.exists(filter_prompt_path):
            default_filter = (
                "general image/video/multimodal generation or image/video editing"
            )
            with open(filter_prompt_path, 'w', encoding='utf-8') as f:
                f.write(default_filter)
            logging.warning(f"No filter prompt. Created a sample CV (Computer Vision) filter prompt at: {filter_prompt_path}. Edit it to your required document filter")

        summarization_prompt_path = os.path.join(self.folder, 'summarization_prompt.txt')
        if not os.path.exists(summarization_prompt_path):
            default_summarization = (
                "Summarize the paper succinctly (1-2 sentences) and highlight key contributions."
            )
            with open(summarization_prompt_path, 'w', encoding='utf-8') as f:
                f.write(default_summarization)
            logging.info(f"Created default summarization prompt at: {summarization_prompt_path}")

        provider_yaml_path = os.path.join(self.folder, 'config.yaml')
        if not os.path.exists(provider_yaml_path):
            yaml_text = (
                "providers:\n"
                "  - provider: arxiv\n"
                f"    provider_feed: {category_default}\n"
            )
            with open(provider_yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_text)
            logging.warning(f"Created default provider feed YAML at: {provider_yaml_path}. Edit it to your required provider and feeds (Ex: provider=arxiv, provider_feed=cs.RO)")

    # --- Loaders ---
    def _load_all(self) -> None:
        self.filter_prompt = self._read_text(os.path.join(self.folder, 'filter_prompt.txt'))
        self.summarization_prompt = self._read_text(os.path.join(self.folder, 'summarization_prompt.txt'))
        self.provider_feed = self._read_config_yaml(os.path.join(self.folder, 'config.yaml'))

    def _read_text(self, path: str) -> Optional[str]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                return text or None
        except Exception as e:
            logging.error(f"Failed reading text file '{path}': {e}", exc_info=True)
            return None

    def _read_config_yaml(self, path: str) -> Optional[str]:
        """Read a very small YAML structure without extra deps.

        Expected format:
        providers:
          - provider: arxiv
            provider_feed: cs.RO

        Returns the first 'provider_feed' value, or None if not found.
        """
        try:
            # Prefer PyYAML if available
            try:
                import yaml  # type: ignore
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                providers = data.get('providers') if isinstance(data, dict) else None
                if isinstance(providers, list) and providers:
                    entry = providers[0]
                    if isinstance(entry, dict):
                        return entry.get('provider_feed')
                return None
            except Exception:
                # Minimal fallback: line scan
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('provider_feed:'):
                            return line.split(':', 1)[1].strip()
                return None
        except Exception as e:
            logging.error(f"Failed reading YAML file '{path}': {e}", exc_info=True)
            return None

