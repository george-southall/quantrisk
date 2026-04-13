"""Reusable export helpers — CSV and interactive HTML chart downloads."""

from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def csv_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    key: str | None = None,
) -> None:
    """Render a Streamlit download button that exports *df* as CSV."""
    csv = df.to_csv(index=True)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def chart_download_button(
    fig: go.Figure,
    filename: str,
    label: str = "Download Chart",
    key: str | None = None,
) -> None:
    """Render a Streamlit download button that exports *fig* as a self-contained HTML file."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    st.download_button(
        label=label,
        data=html,
        file_name=filename,
        mime="text/html",
        key=key,
    )
