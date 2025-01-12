from .settings import Settings
import os

# A singleton instance
settings = Settings.load_config(
    yaml_file=os.getenv("CONFIG_FILE_LOCATION", "config.yaml")
)

__all__ = ["settings"]
