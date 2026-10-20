"""
visualizations.py - Unified Histogram Visualization Engine
Provides `plot_dashboard_histograms` with interactive tooltips, exact counts,
gray-level indices, and seamless support for both manual counts and real image histograms.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Distinctive, elegant modern color theme
PALETTE = {
    "Original": "#3B82F6",          # Tailwind Blue
    "Source Original": "#3B82F6",   # Blue
    "Equalized": "#10B981",         # Emerald Green
    "Matched Result": "#10B981",    # Emerald Green
    "Target Specified": "#F59E0B",  # Amber / Gold
    "Specified Target": "#F59E0B",
    "Reference Target": "#F59E0B",
    "Intermediate": "#8B5CF6",     # Purple
    "Default": "#6366F1"           # Indigo
}


def get_color_for_label(label: str) -> str:
    for key, color in PALETTE.items():
        if key.lower() in label.lower():
            return color
    return PALETTE["Default"]


def plot_dashboard_histograms(
    data_dict: Dict[str, Union[np.ndarray, list]],
    title: str = "Histogram Distribution",
    L: int = 8,
    barmode: str = "group",
    height: int = 380,
    show_percentages: bool = False
):
    """
    Unified visualization function that renders clean distributions with rich hover details:
    - Exact pixel count (formatted with commas)
    - Gray-level index (0 to L-1)
    - Normalized frequency / percentage
    
    Returns a Plotly Figure if plotly is available, else a pandas DataFrame for st.bar_chart.
    """
    if not HAS_PLOTLY:
        # Fallback DataFrame for st.bar_chart
        df = pd.DataFrame(index=np.arange(L))
        for label, counts in data_dict.items():
            arr = np.asarray(counts, dtype=np.int64)
            if len(arr) < L:
                arr = np.pad(arr, (0, L - len(arr)))
            elif len(arr) > L:
                arr = arr[:L]
            df[label] = arr
        df.index.name = "Gray Level"
        return df

    fig = go.Figure()

    gray_levels = np.arange(L)

    for label, counts in data_dict.items():
        arr = np.asarray(counts, dtype=np.int64)
        if len(arr) < L:
            arr = np.pad(arr, (0, L - len(arr)))
        elif len(arr) > L:
            arr = arr[:L]

        total_pixels = int(np.sum(arr))
        if total_pixels > 0:
            percentages = (arr / total_pixels) * 100.0
        else:
            percentages = np.zeros(L)

        # Prepare hover custom data: [count, percentage, total_pixels]
        custom_data = np.stack([arr, percentages, np.full(L, total_pixels)], axis=-1)

        y_vals = percentages if show_percentages else arr
        y_axis_label = "Percentage (%)" if show_percentages else "Pixel Count (Frequency)"

        color = get_color_for_label(label)

        fig.add_trace(go.Bar(
            x=gray_levels,
            y=y_vals,
            name=label,
            marker=dict(
                color=color,
                opacity=0.88,
                line=dict(width=1, color="rgba(255, 255, 255, 0.4)")
            ),
            customdata=custom_data,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Gray Level Index (r): <b>%{x}</b><br>"
                "Pixel Count: <b>%{customdata[0]:,}</b><br>"
                "Percentage: <b>%{customdata[1]:.2f}%</b><br>"
                "<extra></extra>"
            )
        ))

    # Determine x-tick step to keep x-axis clean if L is large (e.g. 128 or 256)
    if L <= 16:
        tickvals = list(range(L))
        ticktext = [str(i) for i in range(L)]
    elif L <= 64:
        step = 4 if L <= 32 else 8
        tickvals = list(range(0, L, step))
        if (L - 1) not in tickvals:
            tickvals.append(L - 1)
        ticktext = [str(i) for i in tickvals]
    else:
        step = 16 if L <= 128 else 32
        tickvals = list(range(0, L, step))
        if (L - 1) not in tickvals:
            tickvals.append(L - 1)
        ticktext = [str(i) for i in tickvals]

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b> (Gray Levels L = {L})",
            font=dict(size=16)
        ),
        barmode=barmode,
        xaxis=dict(
            title="Gray Level Value Index",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)"
        ),
        yaxis=dict(
            title=y_axis_label,
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)"
        ),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=height,
        template="plotly_white"
    )

    return fig
