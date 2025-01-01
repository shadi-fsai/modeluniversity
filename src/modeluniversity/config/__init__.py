from .settings import Settings

# A singleton instance
settings = Settings.load_config(yaml_file="config.yaml", env_file=".env")

__all__ = ["settings"]
