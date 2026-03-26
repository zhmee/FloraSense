"""
Render compact radar charts from SVD latent vectors.

The chart logic follows the standard radar-chart flow:
1. choose a fixed set of axes,
2. compute equally spaced polar angles with NumPy,
3. close the polygon by repeating the first point,
4. render with Matplotlib on a non-interactive backend.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MAX_RADAR_AXES = 6
PROFILE_THEMES = {
    "query": {
        "line": "#677354",
        "fill": "#677354",
        "fill_alpha": 0.18,
        "point_size": 12,
    },
    "flower": {
        "line": "#d78461",
        "fill": "#d78461",
        "fill_alpha": 0.24,
        "point_size": 14,
    },
}


def _closed_loop(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values, [values[0]]))


def _format_axis_label(label: str) -> str:
    label = (label or "").strip().lower()
    if len(label) <= 16:
        return label

    words = label.split()
    if len(words) <= 1:
        return label

    midpoint = (len(words) + 1) // 2
    return " ".join(words[:midpoint]) + "\n" + " ".join(words[midpoint:])


def _rank_axis_indices(vector: np.ndarray) -> np.ndarray:
    """
    Pick the dominant latent dimensions for one profile.
    """
    return np.argsort(np.abs(vector))[::-1]


def select_latent_axes(
    vector: np.ndarray,
    component_labels: list[str],
    axis_limit: int = MAX_RADAR_AXES,
) -> dict | None:
    profile_array = np.asarray(vector, dtype=np.float32).ravel()
    if profile_array.size < 3:
        return None

    ranked_indices = _rank_axis_indices(profile_array)
    selected_indices = []
    selected_labels = []
    seen_labels = set()

    for index in ranked_indices:
        label = _format_axis_label(
            component_labels[int(index)] if int(index) < len(component_labels) else f"Component {int(index) + 1}"
        )
        if label in seen_labels:
            continue
        seen_labels.add(label)
        selected_indices.append(int(index))
        selected_labels.append(label)
        if len(selected_indices) >= min(axis_limit, profile_array.size):
            break

    if len(selected_indices) < 3:
        return None

    return {
        "axis_indices": np.asarray(selected_indices, dtype=int),
        "axis_labels": selected_labels,
    }


def build_latent_radar_chart(
    vector: np.ndarray,
    component_labels: list[str],
    profile_kind: str = "flower",
    axis_indices: np.ndarray | None = None,
    axis_labels: list[str] | None = None,
    axis_limit: int = MAX_RADAR_AXES,
) -> dict | None:
    """
    Return a PNG data URL and metadata for a compact radar chart.
    """
    profile_array = np.asarray(vector, dtype=np.float32).ravel()
    theme = PROFILE_THEMES.get(profile_kind, PROFILE_THEMES["flower"])

    if profile_array.size < 3:
        return None

    if axis_indices is None or axis_labels is None:
        selected_axes = select_latent_axes(profile_array, component_labels, axis_limit)
        if selected_axes is None:
            return None
        axis_indices = selected_axes["axis_indices"]
        axis_labels = selected_axes["axis_labels"]
    else:
        axis_labels = [_format_axis_label(label) for label in axis_labels]

    axis_indices = np.asarray(axis_indices, dtype=int)
    profile_values = np.abs(profile_array[axis_indices])
    scale = float(max(np.max(profile_values), 1e-6))
    normalized_profile = profile_values / scale

    angles = np.linspace(0, 2 * np.pi, len(axis_labels), endpoint=False)
    closed_angles = _closed_loop(angles)
    closed_profile = _closed_loop(normalized_profile)

    fig = plt.figure(figsize=(2.05, 1.92), dpi=170)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(closed_angles, closed_profile, color=theme["line"], linewidth=2.1)
    ax.fill(closed_angles, closed_profile, color=theme["fill"], alpha=theme["fill_alpha"])
    ax.scatter(angles, normalized_profile, color=theme["line"], s=theme["point_size"], zorder=4)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.set_thetagrids(angles * 180 / np.pi, axis_labels, fontsize=6.4, color="#5d4d3b")
    ax.tick_params(pad=2)
    ax.grid(color="#ccb99e", alpha=0.58, linewidth=0.75)
    ax.spines["polar"].set_color((0.51, 0.39, 0.27, 0.18))

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)

    image_data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "image_data_url": image_data_url,
        "axis_labels": list(axis_labels),
    }
