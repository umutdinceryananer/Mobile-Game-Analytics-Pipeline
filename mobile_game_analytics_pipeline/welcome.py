from __future__ import annotations

from pathlib import Path

from mobile_game_analytics_pipeline.config import PROJ_ROOT


ASCII_PATH = PROJ_ROOT / "docs" / "docs" / "greetings.txt"


NEXT_STEPS = [
    ("Rebuild synthetic dataset", "python -m mobile_game_analytics_pipeline.dataset"),
    ("Generate analytics tables & figures", "python -m mobile_game_analytics_pipeline.analytics"),
    ("Create feature set", "python -m mobile_game_analytics_pipeline.features"),
    ("Train churn model", "python -m mobile_game_analytics_pipeline.modeling.train"),
    ("(Optional) Predict churn probabilities", "python -m mobile_game_analytics_pipeline.modeling.predict"),
]


def _load_ascii() -> str:
    try:
        return ASCII_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return """Mobile Game Analytics Pipeline\n(greeting banner not found; run generators to recreate docs/docs/greetings.txt)\n"""


def _format_next_steps() -> str:
    lines = ["Next steps to explore:"]
    for idx, (description, command) in enumerate(NEXT_STEPS, start=1):
        lines.append(f"  {idx}. {description}\n     $ {command}")
    lines.append("\nTip: run 'pre-commit install' to enable Git hooks once dependencies are in place.")
    return "\n".join(lines)


def main() -> None:
    banner = _load_ascii()
    print(banner)
    print(_format_next_steps())


if __name__ == "__main__":
    main()
