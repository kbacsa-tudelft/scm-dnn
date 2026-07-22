"""Minimal markdown builder so the three EDA scripts assemble reports the
same way instead of hand-formatting strings three times."""
from __future__ import annotations

import pandas as pd


class MarkdownReport:
    def __init__(self, title: str):
        self._lines: list[str] = [f"# {title}", ""]

    def h2(self, text: str) -> "MarkdownReport":
        self._lines += [f"## {text}", ""]
        return self

    def h3(self, text: str) -> "MarkdownReport":
        self._lines += [f"### {text}", ""]
        return self

    def para(self, text: str) -> "MarkdownReport":
        self._lines += [text, ""]
        return self

    def bullets(self, items: list[str]) -> "MarkdownReport":
        self._lines += [f"- {item}" for item in items] + [""]
        return self

    def table(self, df: pd.DataFrame, float_format: str = "{:.3f}") -> "MarkdownReport":
        formatted = df.copy()
        for col in formatted.columns:
            if pd.api.types.is_float_dtype(formatted[col]):
                formatted[col] = formatted[col].map(lambda v: float_format.format(v) if pd.notna(v) else "")
        self._lines += [formatted.to_markdown(), ""]
        return self

    def image(self, path: str, alt: str, caption: str | None = None) -> "MarkdownReport":
        self._lines += [f"![{alt}]({path})", ""]
        if caption:
            self._lines += [f"*{caption}*", ""]
        return self

    def raw(self, text: str) -> "MarkdownReport":
        self._lines += [text, ""]
        return self

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self._lines))
