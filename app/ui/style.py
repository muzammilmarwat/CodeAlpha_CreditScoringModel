"""CSS styling helpers for the Streamlit application."""

from __future__ import annotations


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}
.main-title {
    font-size: 2.45rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    letter-spacing: 0;
    color: #111827;
}
.subtitle {
    color: #4b5563;
    font-size: 1.02rem;
    margin-bottom: 1rem;
}
.hero-panel {
    background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 0.75rem;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.2rem;
}
.info-note {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 0.8rem 1rem;
    border-radius: 0.45rem;
    color: #1f2937;
}
.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 0.65rem;
    padding: 1rem 1.1rem;
    margin: 0.75rem 0;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.risk-card {
    padding: 1.1rem 1.25rem;
    border-radius: 0.65rem;
    margin: 0.8rem 0 1rem 0;
    border: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
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
.result-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.8rem;
    margin-top: 0.8rem;
}
.result-item {
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 0.55rem;
    padding: 0.8rem;
}
.result-label {
    color: #6b7280;
    font-size: 0.82rem;
    margin-bottom: 0.15rem;
}
.result-value {
    color: #111827;
    font-size: 1.08rem;
    font-weight: 700;
}
.factor-pill {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 0.32rem 0.65rem;
    margin: 0.18rem;
    color: #1e3a8a;
    font-size: 0.9rem;
}
.metric-caption {
    color: #6b7280;
    font-size: 0.85rem;
}
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
    font-size: 0.86rem;
}
@media (max-width: 900px) {
    .result-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 560px) {
    .result-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def get_custom_css() -> str:
    """Return the custom CSS block used by the Streamlit app."""
    return CUSTOM_CSS
