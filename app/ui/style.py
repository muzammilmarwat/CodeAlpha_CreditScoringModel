"""CSS styling helpers for the Streamlit application."""

from __future__ import annotations


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.brand-logo {
    width: 3rem;
    height: 3rem;
    border-radius: 0.65rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #1d4ed8;
    color: #ffffff;
    font-size: 1.35rem;
    font-weight: 700;
    box-shadow: 0 6px 16px rgba(29, 78, 216, 0.18);
}
.main-title {
    font-size: 2.15rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
    letter-spacing: 0;
    color: #111827;
}
.subtitle {
    color: #4b5563;
    font-size: 0.98rem;
    margin-bottom: 0;
}
.hero-panel {
    background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 0.65rem;
    padding: 0.78rem 1rem;
    margin-bottom: 0.85rem;
}
.info-note {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 0.55rem 0.75rem;
    border-radius: 0.45rem;
    color: #1f2937;
    font-size: 0.9rem;
    margin-top: 0.65rem;
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
    padding: 1.2rem 1.35rem;
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
    padding: 0.85rem;
    min-height: 5.2rem;
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
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    padding: 0.26rem 0.65rem;
    font-size: 0.88rem;
    font-weight: 700;
    border: 1px solid rgba(15, 23, 42, 0.08);
}
.badge-low {
    background: #d1fae5;
    color: #065f46;
}
.badge-medium {
    background: #fef3c7;
    color: #92400e;
}
.badge-high {
    background: #fee2e2;
    color: #991b1b;
}
.probability-panel {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 0.65rem;
    padding: 1rem 1.1rem;
    margin: 0.8rem 0 1rem 0;
}
.probability-row {
    margin: 0.65rem 0;
}
.probability-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.28rem;
}
.bar-track {
    height: 0.72rem;
    background: #f3f4f6;
    border-radius: 999px;
    overflow: hidden;
}
.bar-fill-good {
    height: 100%;
    background: #059669;
    border-radius: 999px;
}
.bar-fill-bad {
    height: 100%;
    background: #dc2626;
    border-radius: 999px;
}
.explanation-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 1rem 0;
}
.explanation-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 0.65rem;
    padding: 0.95rem;
    min-height: 9rem;
}
.explanation-card h4 {
    margin: 0 0 0.45rem 0;
    color: #111827;
}
.explanation-card p {
    margin: 0;
    color: #4b5563;
    font-size: 0.92rem;
}
.status-list {
    line-height: 1.75;
    font-size: 0.92rem;
}
.status-check {
    color: #047857;
    font-weight: 700;
}
.pipeline {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.6rem;
    margin: 1rem 0;
}
.pipeline-step {
    background: #f8fafc;
    border: 1px solid #dbeafe;
    border-radius: 0.55rem;
    padding: 0.75rem;
    text-align: center;
    font-weight: 700;
    color: #1f2937;
    min-height: 4.1rem;
}
.pipeline-arrow {
    color: #2563eb;
    font-size: 1rem;
    margin-top: 0.2rem;
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
    .explanation-grid,
    .pipeline {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 560px) {
    .result-grid {
        grid-template-columns: 1fr;
    }
    .brand-row {
        align-items: flex-start;
    }
    .explanation-grid,
    .pipeline {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def get_custom_css() -> str:
    """Return the custom CSS block used by the Streamlit app."""
    return CUSTOM_CSS
