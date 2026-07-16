"""
Electronic structure data viewer for high-throughput calculations.

Data layout:  DATA_DIR/calc_name/  (no csv/ subdirectory)
  - INIT       : first line has key=value fields like A=16.608 C=22.165
  - band_structure.csv      : columns "k,Energy (eV)"
  - DOS-total.csv           : columns "Energy (eV),DOS,Intg. DOS"
  - COHP_*.csv              : columns "Energy (eV),COHP,Int. COHP"

Usage: cd /opt/data/structure_viz && streamlit run app.py --server.headless true
"""

import math
import os
import re
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- local modules ----
from element_data import element_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# Transition metals (same set as plotting.py)
TRANSITION_METALS = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
]

# Distinct colors for COHP pairs — each pair gets its own color from this palette
PAIR_COLOR_PALETTE = [
    "#2255AA",  # blue
    "#E74C3C",  # red
    "#27AE60",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#FF69B4",  # pink
    "#E67E22",  # dark orange
    "#3498DB",  # light blue
    "#8E44AD",  # dark purple
    "#16A085",  # dark teal
    "#D35400",  # pumpkin
    "#2C3E50",  # navy
    "#7F8C8D",  # grey
    "#A93226",  # dark red
    "#117A65",  # dark teal
    "#6C3483",  # deep purple
    "#1F618D",  # steel blue
    "#1E8449",  # forest green
    "#B7950B",  # mustard
]

PLOT_WIDTH = 600
PLOT_HEIGHT = 1200


# ---- helpers ----

def get_mendeleev(element):
    """Return Mendeleev number for an element symbol from element_data."""
    return element_data.get(element, [0, None, 999])[2]


def extract_elements(label):
    """Extract chemical element symbols from a label string.

    Matches uppercase letter optionally followed by lowercase.
    """
    return re.findall(r"[A-Z][a-z]?", label)


def get_element_color(label, sorted_elements, custom_colors=None):
    """Return color matching plotting.py conventions.

    - Transition metals -> grey
    - Total DOS -> black
    - E (energy-only) -> darkgrey, dashed
    - Non-transition elements colored by position in sorted_elements list:
        count 1: blue
        count 2: blue/red
        count >=3: blue/green/red/orange for non-TM positions, grey for TM

    If custom_colors dict maps label -> hex color, use that instead.
    """
    if custom_colors and label in custom_colors:
        return custom_colors[label]

    label_lower = label.lower()
    if label_lower == "total":
        return "#000000"
    if label_lower == "e":
        return "#4A4A4A"

    elements = [e for e in sorted_elements if e.lower() != "e"]
    element_count = len(elements)

    if element_count == 1:
        return "#2255AA"  # blue
    if element_count == 2:
        if label == elements[0]:
            return "#2255AA"  # blue
        return "#E74C3C"  # red
    if element_count >= 3:
        # Transition metals always grey
        if label in TRANSITION_METALS:
            return "#888888"

        idx = elements.index(label) if label in elements else -1

        if element_count == 3:
            if idx == 0:
                return "#2255AA"  # blue
            if idx == 2:
                return "#E74C3C"  # red
            return "#888888"  # fallback grey for TM
        if element_count == 4:
            # Special case: Cu, Cd, Sn, S
            if set(elements) == {"Cu", "Cd", "Sn", "S"}:
                return {"Cu": "#27AE60", "Cd": "#2255AA", "Sn": "#E74C3C", "S": "#F1C40F"}.get(label, "#888888")
            if idx == 0:
                return "#2255AA"  # blue
            if idx == 2:
                return "#27AE60"  # green
            if idx == 3:
                return "#E74C3C"  # red
            return "#F39C12"  # orange fallback
        if element_count >= 5:
            if idx == 0:
                return "#2255AA"  # blue
            if idx == 2:
                return "#27AE60"  # green
            if idx == 3:
                return "#E74C3C"  # red
            if idx == 4:
                return "#F39C12"  # orange
            return "#FF69B4"  # pink fallback

    return "#888888"  # grey fallback


def sort_by_mendeleev(elements):
    """Sort element symbols by their Mendeleev number."""
    return sorted(elements, key=lambda e: get_mendeleev(e))


# ---- data loading ----

def load_init(path):
    """Parse the first line of an INIT file for A and C values.

    Format: SPCGRP=142  IORIGIN=2  ATUNITS=F  A=16.6083  C=22.165
    """
    with open(path) as f:
        line = next(f).strip()
    m_a = re.search(r"A=(\S+)", line)
    m_c = re.search(r"C=(\S+)", line)
    return float(m_a.group(1)), float(m_c.group(1))


def scan_calculations():
    """Scan DATA_DIR/calc_name/INIT and return {name: {A, C, dir}}."""
    calcs = {}
    for entry in sorted(os.listdir(DATA_DIR)):
        calc_dir = os.path.join(DATA_DIR, entry)
        init_path = os.path.join(calc_dir, "INIT")
        if not os.path.isdir(calc_dir) or not os.path.isfile(init_path):
            continue
        A, C = load_init(init_path)
        calcs[entry] = {"A": A, "C": C, "dir": calc_dir}
    return calcs


# ---- data inspection helpers ----

def get_dos_elements(calc_dir):
    """Return sorted list of unique elements found in DOS files."""
    dos_files = sorted(glob.glob(os.path.join(calc_dir, "DOS-*.csv")))
    if not dos_files:
        dos_files = sorted(glob.glob(os.path.join(calc_dir, "DOS*.csv")))
    if not dos_files:
        return []

    all_elements = set()
    for fpath in dos_files:
        name = os.path.basename(fpath).replace("DOS-", "").replace("DOS", "")
        if name.endswith(".csv"):
            name = name[:-4]
        elements = extract_elements(name)
        all_elements.update(elements)

    return sort_by_mendeleev(all_elements)


def get_cohp_pairs(calc_dir):
    """Return sorted list of COHP pair labels."""
    cohp_files = sorted(glob.glob(os.path.join(calc_dir, "COHP_*.csv")))
    if not cohp_files:
        return []

    pairs = []
    for fpath in cohp_files:
        pname = os.path.basename(fpath).replace("COHP_", "").replace(".csv", "")
        elements = extract_elements(pname.replace("-", "/"))
        sorted_pair = sort_by_mendeleev(elements)
        pair_label = "-".join(sorted_pair)
        pairs.append(pair_label)

    return pairs


# ---- plot functions (plotting.py conventions) ----

def plot_band_structure(calc_dir, emin=-5, emax=5, line_width=0.85):
    """Plot band structure: x=k-points (horizontal), y=Energy (vertical).

    Conventions from plotting.py:
      - Adjustable line width for bands (default 0.85)
      - Vertical grid at k-point boundaries (zorder=2)
      - E_F annotation at zero energy on right edge
      - Spine linewidth 1.5, tick width 1.5/length 8
      - Energy range defaults to [-5, 5]
    """
    path = os.path.join(calc_dir, "band_structure.csv")
    if not os.path.isfile(path):
        return None

    df = pd.read_csv(path)
    # Strip whitespace from column names like plotting.py does
    df.columns = df.columns.str.strip()

    # Load k-point labels for axis ticks
    pts_path = os.path.join(calc_dir, "band_structure_points.csv")
    k_positions = []
    k_labels = {}
    if os.path.isfile(pts_path):
        pts = pd.read_csv(pts_path)
        pts.columns = pts.columns.str.strip()
        for _, row in pts.iterrows():
            k_positions.append(row["values"])
            k_labels[row["values"]] = row["point"]

    fig = go.Figure()

    # Band trace — adjustable line width
    fig.add_trace(go.Scatter(
        x=df["k"], y=df["Energy (eV)"], mode="lines",
        line=dict(color="#4A90D9", width=line_width),
        hoverinfo="skip",
    ))

    # Energy range defaults to [-5, 5]
    energy_emin, energy_emax = -5, 5
    if "Energy (eV)" in df.columns and len(df) > 0:
        energy_emin = min(energy_emin, df["Energy (eV)"].min())
        energy_emax = max(energy_emax, df["Energy (eV)"].max())

    # Vertical grid lines at high-symmetry k-points (zorder=2 from plotting.py)
    x_min = None
    x_max = None
    if k_positions:
        x_min = min(k_positions)
        x_max = max(k_positions)
        unique_positions = sorted(set(k_positions))
        for pos in unique_positions:
            fig.add_vline(x=pos, line_width=1.5, line_dash="dot", line_color="black")

        ticktext = []
        tickvals = []
        for pos in unique_positions:
            label = next((k_labels.get(p, "") for p in k_positions if p == pos), "")
            ticktext.append(label)
            tickvals.append(pos)

        fig.update_xaxes(
            title=dict(text=r"<i>k</i>-points", font=dict(size=12)),
            ticktext=ticktext,
            tickvals=tickvals,
            tickfont=dict(size=12),
            range=[x_min, x_max],
        )
    else:
        fig.update_xaxes(
            title=dict(text=r"<i>k</i>-points", font=dict(size=12)),
            tickfont=dict(size=12),
        )

    # Y-axis — Energy (rotated convention) — use user-provided emin/emax
    fig.update_yaxes(
        title=dict(text="energy (eV)", font=dict(size=12)),
        tickfont=dict(size=12),
        range=[emin, emax],
    )

    # Zero-line (dashed black) and E_F annotation
    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="black")

    # E_F annotation at right edge of plot
    fig.add_annotation(
        x=x_max if x_max else (df["k"].max() if len(df) > 0 else 0),
        y=0,
        text="EF",
        showarrow=False,
        font=dict(size=20, color="black"),
        xref="x", yref="y",
        xshift=10, yshift=0,
    )

    # Compact size for side-by-side layout in a column
    fig.update_layout(
        title=dict(text="Band Structure", font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=PLOT_WIDTH, 
        height=PLOT_HEIGHT,
        margin=dict(l=50, r=30, t=40, b=40),
    )

    return fig


def plot_dos(calc_dir, emin=-6, emax=2, custom_colors=None, line_width=2):
    """Plot DOS with rotated axes: x=DOS states (horizontal), y=Energy (vertical).

    Conventions from plotting.py:
      - Rotated: x = DOS values, y = Energy
      - Element colors based on Mendeleev-sorted composition (transition metals grey)
      - Adjustable line width for DOS curves (default 5)
      - Thick zero-line (dashed black), E_F annotation at right edge
      - Legend: frameless, placed optimally
      - Integer ticks on energy axis
      - Data sorted by Mendeleev number ordering

    Integrated DOS plotted on secondary x-axis (right side).
    """
    dos_files = sorted(glob.glob(os.path.join(calc_dir, "DOS-*.csv")))
    if not dos_files:
        dos_files = sorted(glob.glob(os.path.join(calc_dir, "DOS*.csv")))
    if not dos_files:
        return None

    fig = go.Figure()

    # Collect all elements across DOS files
    all_elements = set()
    file_data = []  # (x=energy, y=dos_values, label)

    for fpath in dos_files:
        name = os.path.basename(fpath).replace("DOS-", "").replace("DOS", "")
        if name.endswith(".csv"):
            name = name[:-4]
        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip()

        energy_col = "Energy (eV)"
        dos_col = "DOS"

        x_energy = df[energy_col].values
        y_dos = df[dos_col].values
        elements = extract_elements(name)
        all_elements.update(elements)
        file_data.append((x_energy, y_dos, name))

    # Sort elements by Mendeleev number
    sorted_elements = sort_by_mendeleev(all_elements)

    # Sort file data: total first, then by Mendeleev order
    def sort_key(item):
        label = item[2]
        if label.lower() == "total":
            return (-1, 0)
        elements = extract_elements(label)
        min_mendel = min(get_mendeleev(e) for e in elements)
        return (min_mendel, label)

    file_data.sort(key=sort_key)

    # Compute max DOS for x-axis range
    all_y_values = []
    for x_energy, y_dos, _label in file_data:
        all_y_values.extend(y_dos[y_dos >= 0])

    max_dos = max(all_y_values) if all_y_values else 1
    buffer = 0.05 * max_dos
    x_max = max_dos + buffer

    # Plot each DOS trace — rotated: x=DOS, y=Energy, adjustable line width
    for x_energy, y_dos, label in file_data:
        color = get_element_color(label, sorted_elements, custom_colors)

        # Filter to energy range
        mask = (x_energy >= emin) & (x_energy <= emax)
        xe = x_energy[mask]
        ye = y_dos[mask]

        if label.lower() == "total":
            zorder = 10
        else:
            zorder = 0

        fig.add_trace(go.Scatter(
            x=ye, y=xe,  # rotated: DOS on x, Energy on y
            mode="lines", name=label,
            line=dict(color=color, width=line_width),
            hoverinfo="skip",
        ))

    # Integrated DOS — secondary x-axis on right side
    total = os.path.join(calc_dir, "DOS-total.csv")
    if os.path.isfile(total):
        df = pd.read_csv(total)
        df.columns = df.columns.str.strip()
        energy_col = "Energy (eV)"

        if "Intg. DOS" in df.columns:
            ig_dos = df["Intg. DOS"].values
            e_vals = df[energy_col].values
            mask = (e_vals >= emin) & (e_vals <= emax)
            fig.add_trace(go.Scatter(
                x=ig_dos[mask], y=e_vals[mask],  # rotated
                mode="lines", name="Int. DOS",
                line=dict(color="#4A4A4A", width=line_width, dash="dash"),
                hoverinfo="skip",
            ))

    # X-axis = DOS states (horizontal), Y-axis = Energy (vertical)
    fig.update_xaxes(
        title=dict(text="states", font=dict(size=14)),
        tickfont=dict(size=14),
        range=[0, x_max],
    )

    fig.update_yaxes(
        title=dict(text="energy (eV)", font=dict(size=14)),
        tickfont=dict(size=14),
        range=[emin, emax],
        dtick=1,  # integer ticks on energy axis
    )

    # Zero-line + E_F annotation at right edge
    fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")
    fig.add_annotation(
        x=x_max,
        y=0,
        text="EF",
        showarrow=False,
        font=dict(size=14, color="black"),
        xref="x", yref="y",
        xshift=15, yshift=0,
    )

    # Compact size for side-by-side layout in a column
    fig.update_layout(
        title=dict(text="DOS", font=dict(size=14)),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=12),
            x=0.75, y=1,
        ),
        width=PLOT_WIDTH, 
        height=PLOT_HEIGHT,
        margin=dict(l=50, r=30, t=40, b=40),
    )

    return fig


def plot_cohp(calc_dir, emin=-6, emax=2, custom_colors=None, line_width=3, selected_pairs=None):
    """Plot COHP with rotated axes: x=COHP (horizontal), y=Energy (vertical).

    Conventions from plotting.py:
      - Rotated: x = COHP values, y = Energy
      - No fill-to-zero — solid lines only for COHP
      - iCOHP plotted as dashed lines (derived from line_width)
      - Pair labels sorted by Mendeleev numbers of constituent elements
      - Each pair gets a distinct color from the palette (not element-based)
      - Adjustable line width for COHP (default 3), iCOHP uses max(lw-2, 1)
      - Zero-line cross: both horizontal and vertical dashed black lines
      - E_F annotation at right edge
      - Legend: frameless, zorder=99 equivalent

    Parameters
    ----------
    selected_pairs : list or None
        If provided, only plot these pairs. If None/empty, show all pairs found.
    """
    cohp_files = sorted(glob.glob(os.path.join(calc_dir, "COHP_*.csv")))
    if not cohp_files:
        return None

    # Build normalized pair names (dash-separated, Mendeleev-sorted) for consistent comparison
    pair_names_normalized = []
    pair_name_map = {}  # raw filename -> normalized label
    for f in cohp_files:
        raw = os.path.basename(f).replace("COHP_", "").replace(".csv", "")
        elems = extract_elements(raw.replace("_", "-").replace("-", "/"))
        sorted_pair = sort_by_mendeleev(elems)
        norm = "-".join(sorted_pair)
        pair_names_normalized.append(norm)
        pair_name_map[raw] = norm

    # Use externally provided selection, or default to all normalized names
    selected = selected_pairs if selected_pairs else pair_names_normalized

    fig = go.Figure()

    all_x_values = []

    for i, fpath in enumerate(cohp_files):
        raw_pname = os.path.basename(fpath).replace("COHP_", "").replace(".csv", "")
        pair_label = pair_name_map[raw_pname]
        if not selected or pair_label not in selected:
            continue

        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip()

        # Determine color: use custom override, otherwise palette by position
        if custom_colors and pair_label in custom_colors:
            cohp_color = custom_colors[pair_label]
        else:
            # Find this pair's index in the full pairs list for consistent palette assignment
            pair_idx = pair_names_normalized.index(pair_label) if pair_label in pair_names_normalized else i
            cohp_color = PAIR_COLOR_PALETTE[pair_idx % len(PAIR_COLOR_PALETTE)]

        mask = (df["Energy (eV)"] >= emin) & (df["Energy (eV)"] <= emax)

        # COHP: solid line, adjustable width, rotated x=COHP y=Energy
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "COHP"].values,
            y=df.loc[mask, "Energy (eV)"].values,
            mode="lines", name=pair_label,
            line=dict(color=cohp_color, width=line_width),
            hoverinfo="skip",
        ))

        all_x_values.extend(np.abs(df.loc[mask, "COHP"].values))

        # iCOHP: dashed line, thinner (max(lw-2, 1)), rotated x=Int. COHP y=Energy
        if "Int. COHP" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.loc[mask, "Int. COHP"].values,
                y=df.loc[mask, "Energy (eV)"].values,
                mode="lines", name=f"{pair_label} (iCOHP)",
                line=dict(color=cohp_color, width=max(line_width - 2, 1), dash="dash"),
                hoverinfo="skip",
            ))
            all_x_values.extend(np.abs(df.loc[mask, "Int. COHP"].values))

    # X-axis limits: auto-range with buffer
    max_x = max(all_x_values) if all_x_values else 1
    buffer = 0.05 * max_x
    fig.update_xaxes(
        title=dict(text="-COHP", font=dict(size=14)),
        tickfont=dict(size=14),
        range=[-(max_x + buffer), max_x + buffer],
    )

    # Y-axis: Energy with integer ticks
    fig.update_yaxes(
        title=dict(text="energy (eV)", font=dict(size=14)),
        tickfont=dict(size=14),
        range=[emin, emax],
        dtick=1,
    )

    # Zero-line cross: horizontal + vertical dashed black lines (lw=3)
    fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black")

    # E_F annotation at right edge of plot
    fig.add_annotation(
        x=max_x + buffer,
        y=0,
        text="EF",
        showarrow=False,
        font=dict(size=14, color="black"),
        xref="x", yref="y",
        xshift=15, yshift=0,
    )

    # Compact size for side-by-side layout in a column
    fig.update_layout(
        title=dict(text="COHP", font=dict(size=14)),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=12),
            x=-0.05, y=0.0,
        ),
        width=PLOT_WIDTH, 
        height=PLOT_HEIGHT,
        margin=dict(l=50, r=30, t=40, b=40),
    )

    return fig


# ---- app ----

st.set_page_config(page_title="Structure Viz", layout="wide")
st.title("Electronic Structure Visualizer")

CALCS = scan_calculations()

if not CALCS:
    st.error(f"No calculations found in {DATA_DIR}.")
    st.stop()

st.info(f"Loaded {len(CALCS)} calculations")

def render_tick_ruler(values, label, shared_map=None):
    """Render a wide HTML ruler with '|' tick marks and value labels.

    - All values are always visible (full-width container).
    - If *shared_map[v]* > 1 (i.e. this value maps to multiple counterparts),
      the '|' is colored; otherwise it stays grey.
    - Labels are shown for every available value (not spaced at fixed intervals)
      so nothing is hidden.
    """
    if len(values) < 2:
        return
    vmin = min(values)
    vmax = max(values)
    vrange = vmax - vmin or 1

    # Build tick marks: '|' for every available value
    tick_marks = []
    for v in values:
        pct = (v - vmin) / vrange * 100
        is_shared = False
        if shared_map is not None:
            is_shared = shared_map.get(v, 1) > 1
        color = "#E74C3C" if is_shared else "#999"
        # Show the value label directly under each tick (all values always visible)
        tick_marks.append(
            f'<span style="position:absolute;left:{pct}%;transform:translateX(-50%);'
            f'font-size:14px;color:{color};margin-top:2px">|</span>'
            f'<span style="position:absolute;left:{pct}%;transform:translateX(-50%);'
            f'font-size:9px;color:#666;margin-top:16px">{v:.3f}</span>'
        )

    all_spans = " ".join(tick_marks)
    st.markdown(
        f'<div style="position:relative;height:38px;border-bottom:1px solid #ddd;'
        f'margin-top:-8px;margin-bottom:4px;min-width:100%;overflow-x:auto;">{all_spans}</div>',
        unsafe_allow_html=True,
    )

# -- A slider with tick marks --
unique_A = sorted(set(v["A"] for v in CALCS.values()))

# Build shared_map: how many distinct C values each A maps to (across ALL calcs)
a_shared_map = {}
for a_val in unique_A:
    c_vals = set()
    for v in CALCS.values():
        if abs(v["A"] - a_val) < 1e-6:
            c_vals.add(v["C"])
    a_shared_map[a_val] = len(c_vals)

sel_A = st.select_slider("Select A", options=unique_A, value=unique_A[0], key="a_slider",
                         format_func=lambda x: str(x))
# render_tick_ruler(unique_A, "A", shared_map=a_shared_map)

# -- C slider filtered by selected A --
matching = {k: v for k, v in CALCS.items() if abs(v["A"] - sel_A) < 1e-6}
if not matching:
    st.warning(f"No calculations match A={sel_A:.4f}")
    st.stop()

unique_C = sorted(set(v["C"] for v in matching.values()))

# Build shared_map: how many distinct A values each C maps to (across ALL calcs)
c_shared_map = {}
for c_val in unique_C:
    a_vals = set()
    for v in CALCS.values():
        if abs(v["C"] - c_val) < 1e-6:
            a_vals.add(v["A"])
    c_shared_map[c_val] = len(a_vals)

sel_C = st.select_slider("Select C", options=unique_C, value=unique_C[0], key="c_slider")
render_tick_ruler(unique_C, "C", shared_map=c_shared_map)

# -- pick the matched calculation (first match) --
calc = next((v for v in matching.values() if abs(v["C"] - sel_C) < 1e-6), None)
if calc is None:
    st.warning("No match found.")
    st.stop()

st.markdown(f"**{os.path.basename(calc['dir'])}** | A={calc['A']:.4f}, C={calc['C']:.4f}")

# ---- session state for plot settings (persist across recalculation) ----
if "band_emin" not in st.session_state:
    st.session_state.band_emin = -5.0
if "band_emax" not in st.session_state:
    st.session_state.band_emax = 2.0
if "band_line_width" not in st.session_state:
    st.session_state.band_line_width = 0.85
if "dos_emin" not in st.session_state:
    st.session_state.dos_emin = -6.0
if "dos_emax" not in st.session_state:
    st.session_state.dos_emax = 2.0
if "dos_line_width" not in st.session_state:
    st.session_state.dos_line_width = 2.0
if "cohp_emin" not in st.session_state:
    st.session_state.cohp_emin = -6.0
if "cohp_emax" not in st.session_state:
    st.session_state.cohp_emax = 2.0
if "cohp_line_width" not in st.session_state:
    st.session_state.cohp_line_width = 3.0

# ---- plots in one row (three columns) ----
col_band, col_dos, col_cohp = st.columns(3)

with col_band:
    st.subheader("Band Structure")
    with st.expander("Settings"):
        cols = st.columns(3)
        st.session_state.band_emin = cols[0].number_input(
            "Min Energy (eV)", value=st.session_state.band_emin, step=0.5)
        st.session_state.band_emax = cols[1].number_input(
            "Max Energy (eV)", value=st.session_state.band_emax, step=0.5)
        st.session_state.band_line_width = cols[2].number_input(
            "Line Width", value=st.session_state.band_line_width,
            min_value=0.1, max_value=20.0, step=0.1, format="%.2f")

    fig = plot_band_structure(calc["dir"],
                              emin=st.session_state.band_emin,
                              emax=st.session_state.band_emax,
                              line_width=st.session_state.band_line_width)
    if fig:
        st.plotly_chart(fig, width='content')
    else:
        st.warning("band_structure.csv not found.")

with col_dos:
    st.subheader("DOS")
    with st.expander("Settings"):
        cols = st.columns(3)
        st.session_state.dos_emin = cols[0].number_input(
            "Min Energy (eV)", value=st.session_state.dos_emin, step=0.5,
            key="dos_min")
        st.session_state.dos_emax = cols[1].number_input(
            "Max Energy (eV)", value=st.session_state.dos_emax, step=0.5,
            key="dos_max")
        st.session_state.dos_line_width = cols[2].number_input(
            "Line Width", value=st.session_state.dos_line_width,
            min_value=0.1, max_value=30.0, step=0.5, format="%.1f",
            key="dos_lw")

    # Color customization for elements
    dos_elements = get_dos_elements(calc["dir"])
    custom_colors_dos = {}

    with st.expander("Element Colors"):
        if dos_elements:
            colorable = [e for e in dos_elements if e.lower() not in ("total", "e")]
            if colorable:
                sorted_el = sort_by_mendeleev(colorable)
                defaults = {e: get_element_color(e, sorted_el) for e in colorable}

                n_cols = min(len(colorable), 6)
                cols = st.columns(n_cols if n_cols > 1 else 2)
                col_idx = 0
                for elem in colorable:
                    with cols[col_idx % n_cols]:
                        key = f"dos_color_{elem}"
                        val = st.color_picker(elem, value=defaults[elem], key=key)
                        col_idx += 1

            custom_colors_dos = {}
            for elem in colorable:
                key = f"dos_color_{elem}"
                if st.session_state.get(key):
                    custom_colors_dos[elem] = st.session_state[key]
        else:
            st.info("No DOS data available.")

    fig = plot_dos(calc["dir"],
                   emin=st.session_state.dos_emin,
                   emax=st.session_state.dos_emax,
                   custom_colors=custom_colors_dos or None,
                   line_width=st.session_state.dos_line_width)
    if fig:
        st.plotly_chart(fig, width='content')
    else:
        st.warning("No DOS files found.")

with col_cohp:
    st.subheader("COHP")
    with st.expander("Settings"):
        cols = st.columns(3)
        st.session_state.cohp_emin = cols[0].number_input(
            "Min Energy (eV)", value=st.session_state.cohp_emin, step=0.5,
            key="cohp_min")
        st.session_state.cohp_emax = cols[1].number_input(
            "Max Energy (eV)", value=st.session_state.cohp_emax, step=0.5,
            key="cohp_max")
        st.session_state.cohp_line_width = cols[2].number_input(
            "Line Width", value=st.session_state.cohp_line_width,
            min_value=0.1, max_value=30.0, step=0.5, format="%.1f",
            key="cohp_lw")

    # Color customization for pairs
    cohp_pairs = get_cohp_pairs(calc["dir"])
    custom_colors_cohp = {}

    with st.expander("Pair Colors"):
        if cohp_pairs:
            n_cols = min(len(cohp_pairs), 4)
            cols = st.columns(n_cols if n_cols > 1 else 2)
            col_idx = 0
            for pair in cohp_pairs:
                pair_idx = cohp_pairs.index(pair)
                default_color = PAIR_COLOR_PALETTE[pair_idx % len(PAIR_COLOR_PALETTE)]

                with cols[col_idx % n_cols]:
                    key = f"cohp_color_{pair.replace('/', '_')}"
                    val = st.color_picker(pair, value=default_color, key=key)
                    col_idx += 1

            custom_colors_cohp = {}
            for pair in cohp_pairs:
                key = f"cohp_color_{pair.replace('/', '_')}"
                if st.session_state.get(key):
                    custom_colors_cohp[pair] = st.session_state[key]
        else:
            st.info("No COHP data available.")

    # Pair selection — rendered inline in the column
    with st.expander("Pair Selection"):
        if cohp_pairs:
            st.multiselect(
                "Show pairs:",
                options=cohp_pairs,
                default=cohp_pairs,
                key="cohp_pair_select",
            )

    selected = st.session_state.get("cohp_pair_select") if cohp_pairs else None

    fig = plot_cohp(calc["dir"],
                    emin=st.session_state.cohp_emin,
                    emax=st.session_state.cohp_emax,
                    custom_colors=custom_colors_cohp or None,
                    line_width=st.session_state.cohp_line_width,
                    selected_pairs=selected)
    if fig:
        st.plotly_chart(fig, width='content')
    else:
        st.warning("No COHP files found.")
