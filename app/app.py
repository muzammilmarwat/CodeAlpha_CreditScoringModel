"""Minimal application entrypoint for the credit scoring deployment layer."""

from __future__ import annotations

from app import config
from app.services.model_loader import get_available_models


def main() -> None:
    """Print lightweight app metadata until the Streamlit UI is implemented."""
    print(f"{config.APP_NAME} v{config.APP_VERSION}")
    print("Available models:")
    for model_name in get_available_models():
        print(f"- {model_name}")
    print("Streamlit UI is intentionally not implemented in Phase 8A/8B.")


if __name__ == "__main__":
    main()
