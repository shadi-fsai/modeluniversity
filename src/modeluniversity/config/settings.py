from datetime import datetime
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator
from pathlib import Path
import yaml
import os


class QuestionSettings(BaseSettings):
    num_total: int = Field(..., gt=0, description="Total number of questions")
    num_easy: int = Field(..., ge=0, description="Number of easy questions")
    num_medium: int = Field(..., ge=0, description="Number of medium questions")
    num_hard: int = Field(..., ge=0, description="Number of hard questions")
    allow_expansion: bool = True

    @field_validator("num_easy", "num_medium", "num_hard")
    @classmethod  # This is required in Pydantic v2
    def validate_numbers(cls, v: int, values: Dict[str, Any]) -> int:
        if "num_total" in values.data:
            total = values.data["num_total"]
            if v > total:
                raise ValueError(f"Subset count ({v}) cannot exceed total ({total})")
        return v

    @field_validator(
        "num_hard"
    )  # We can do the sum validation after all numbers are set
    @classmethod
    def validate_sum(cls, v: int, values: Dict[str, Any]) -> int:
        data = values.data
        if all(k in data for k in ["num_easy", "num_medium"]):
            sum_parts = data["num_easy"] + data["num_medium"] + v
            if "num_total" in data and sum_parts != data["num_total"]:
                raise ValueError(
                    f'Sum of question types ({sum_parts}) must equal total ({data["num_total"]})'
                )
        return v


class APISettings(BaseSettings):
    groq_api_key: str = Field(..., description="Groq API key for LLM access")
    opik_api_key: str = Field(..., description="Opik API key for evaluations")
    opik_workspace: str = Field(..., description="Opik workspace identifier")


class Settings(BaseSettings):
    # Metadata - for future reference, right now they're useless
    config_version: str = "1.0.0"
    last_modified: datetime = Field(
        default=datetime(2024, 12, 31, 16, 40, 54),
        description="Last modification timestamp",
    )
    modified_by: str = Field(
        default="voulkon",  # Your username
        description="User who last modified the config",
    )

    # API Settings
    api: APISettings = Field(default_factory=APISettings)

    # LLM Configuration
    datagen_model: str
    student_role: str
    teacher_role: str
    llm_evals_list: List[str]

    # Question Settings
    practice: QuestionSettings
    test: QuestionSettings

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow"
    )

    @classmethod
    def load_config(
        cls,
        yaml_file: Optional[str] = None,
        env_file: Optional[str] = None,
        overrides: Dict[str, Any] = None,
    ) -> "Settings":

        yaml_path = Path(yaml_file) if yaml_file else Path("config.yaml")

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found at {yaml_path}")

        yaml_settings = {}
        if yaml_file and Path(yaml_file).exists():
            with open(yaml_file, "r") as f:
                yaml_settings = yaml.safe_load(f)

        if env_file:
            os.environ["ENVIRONMENT_FILE"] = env_file

        yaml_settings.update(
            {
                "last_modified": datetime(
                    2024, 12, 31, 16, 40, 54
                ),  # Your current UTC time
                "modified_by": "voulkon",  # Your username
            }
        )

        settings = cls(**yaml_settings)

        if overrides:
            for key, value in overrides.items():
                setattr(settings, key, value)

        return settings

    def save_config(self, yaml_file: str) -> None:
        """Save current configuration to YAML file."""
        with open(yaml_file, "w") as f:
            yaml.dump(self.model_dump(), f)
