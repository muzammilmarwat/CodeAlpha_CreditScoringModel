"""CSS styling helpers for the Streamlit application."""

from __future__ import annotations


CUSTOM_CSS = """
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.subtitle {
    color: #4b5563;
    font-size: 1.02rem;
    margin-bottom: 1rem;
}
.info-note {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 0.8rem 1rem;
    border-radius: 0.25rem;
    color: #1f2937;
}
.risk-card {
    padding: 1rem;
    border-radius: 0.4rem;
    margin: 0.6rem 0 1rem 0;
    border: 1px solid rgba(0, 0, 0, 0.08);
}
.risk-low {
    background: #ecfdf5;
    border-left: 6px solid #059669;
}
.risk-medium {
    background: #fffbeb;
    border-left: 6px solid #d97706;
}
.risk-high {
    background: #fef2f2;
    border-left: 6px solid #dc2626;
}
.metric-caption {
    color: #6b7280;
    font-size: 0.85rem;
}
</style>
"""


def get_custom_css() -> str:
    """Return the custom CSS block used by the Streamlit app."""
    return CUSTOM_CSS
