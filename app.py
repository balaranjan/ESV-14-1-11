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
# from cascade_selector import cascade_selector

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


PARAM_COLS = [
    {"col": "A",     "label": "A (Å)",   "param": "sel_a"},
    {"col": "C",     "label": "C (Å)",   "param": "sel_c"},
    {"col": "Yb1_x", "label": "Yb1 x",  "param": "sel_yb1x"},
    {"col": "Yb1_y", "label": "Yb1 y",  "param": "sel_yb1y"},
    {"col": "Yb1_z", "label": "Yb1 z",  "param": "sel_yb1z"},
    {"col": "Yb3_x", "label": "Yb3 x",  "param": "sel_yb3x"},
    {"col": "Yb3_y", "label": "Yb3 y",  "param": "sel_yb3y"},
    {"col": "Yb3_z", "label": "Yb3 z",  "param": "sel_yb3z"},
    {"col": "Yb4_x", "label": "Yb4 x",  "param": "sel_yb4x"},

    {"col": "Sb1_x", "label": "Sb1 x",  "param": "sel_sb1x"},
    {"col": "Sb1_y", "label": "Sb1 y",  "param": "sel_sb1y"},
    {"col": "Sb1_z", "label": "Sb1 z",  "param": "sel_sb1z"},

    {"col": "Sb2_x", "label": "Sb2 x",  "param": "sel_sb2x"},
    {"col": "Sb2_y", "label": "Sb2 y",  "param": "sel_sb2y"},
    {"col": "Sb2_z", "label": "Sb2 z",  "param": "sel_sb2z"},

    {"col": "Sb3_x", "label": "Sb3 x",  "param": "sel_sb3x"},
    {"col": "Sb3_y", "label": "Sb3 y",  "param": "sel_sb3y"},

]

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


def order_site_info(site_info):
   
    Yb = np.array([
    [0.02204, 0.37564, 0.00273],  # 'Yb1', 
    [0.04246, 0.07375, 0.17229],  # 'Yb2', 
    [0.3404, 0.07028, 0.09279],   # 'Yb3', 
    [0.35462, 0.0, 0.25]          # 'Yb4', 
    ]
)
    Sb = np.array([
    [0.1306, 0.02704, 0.04634],   #  Sb1
    [0.35899, 0.25562, 0.05999],
    [0.13647, 0.38647, 0.125],
    [0.0, 0.25, 0.125],
    ])

    ordered = []
    for l, x, y, z in site_info:
        lp = "Yb"
        m = Yb
        if 'Sb' in l:
            lp = "Sb"
            m = Sb

        dists = np.linalg.norm(m - np.array([x, y, z]), axis=1)
        ind = np.argmin(dists)
        
        lp = f"{lp}{ind+1}"
        ordered.append([lp, x, y, z])

    return ordered

def load_init(path):
    """Parse the first line of an INIT file for A and C values.
    Format: SPCGRP=142  IORIGIN=2  ATUNITS=F  A=16.6083  C=22.165
    """
    with open(path) as f:
        line = next(f).strip()
    m_a = re.search(r"A=(\S+)", line)
    m_c = re.search(r"C=(\S+)", line)

    site_info = []
    with open(path) as f:
        for line in f.readlines()[1:]:
            line = line.split()
            if not len(line) > 3:
                continue
            x, y, z = line[-3:]
            x = float(x.replace("X=", ""))
            y, z, = float(y), float(z)
            site = line[0].replace("ATOM=", "")

            site_info.append([site, x, y, z])

    site_info = order_site_info(site_info)
    return float(m_a.group(1)), float(m_c.group(1)), site_info

def scan_calculations():
    """Scan DATA_DIR/calc_name/INIT and return {name: {A, C, dir}}."""
    calcs = {}

    for entry in sorted(os.listdir(DATA_DIR)):
        calc_dir = os.path.join(DATA_DIR, entry)
        init_path = os.path.join(calc_dir, "INIT")
        if not os.path.isdir(calc_dir) or not os.path.isfile(init_path):
            continue
        A, C, site_info = load_init(init_path)

        row = {"sample": entry, "A": A, "C": C, "dir": calc_dir}
        for l, x, y, z in site_info:
            row[f"{l}_x"] = x
            row[f"{l}_y"] = y
            row[f"{l}_z"] = z

        calcs[entry] = row

    return calcs

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

def plot_band_structure(calc_dir, emin=-5, emax=5, line_width=0.85):
    """Plot band structure: x=k-points (horizontal), y=Energy (vertical)."""
    path = os.path.join(calc_dir, "band_structure.csv")
    if not os.path.isfile(path):
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

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

    fig.add_trace(go.Scatter(
        x=df["k"], y=df["Energy (eV)"], mode="lines",
        line=dict(color="#4A90D9", width=line_width),
        hoverinfo="skip",
    ))

    energy_emin, energy_emax = -5, 5
    if "Energy (eV)" in df.columns and len(df) > 0:
        energy_emin = min(energy_emin, df["Energy (eV)"].min())
        energy_emax = max(energy_emax, df["Energy (eV)"].max())

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

    fig.update_yaxes(
        title=dict(text="energy (eV)", font=dict(size=12)),
        tickfont=dict(size=12),
        range=[emin, emax],
    )

    fig.add_hline(y=0, line_width=1.5, line_dash="dash", line_color="black")

    fig.add_annotation(
        x=x_max if x_max else (df["k"].max() if len(df) > 0 else 0),
        y=0,
        text="EF",
        showarrow=False,
        font=dict(size=20, color="black"),
        xref="x", yref="y",
        xshift=10, yshift=0,
    )

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
    """Plot DOS with rotated axes: x=DOS states (horizontal), y=Energy (vertical)."""
    dos_files = sorted(glob.glob(os.path.join(calc_dir, "DOS-*.csv")))
    if not dos_files:
        dos_files = sorted(glob.glob(os.path.join(calc_dir, "DOS*.csv")))
    if not dos_files:
        return None

    fig = go.Figure()

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

    sorted_elements = sort_by_mendeleev(all_elements)

    def sort_key(item):
        label = item[2]
        if label.lower() == "total":
            return (-1, 0)
        elements = extract_elements(label)
        min_mendel = min(get_mendeleev(e) for e in elements)
        return (min_mendel, label)

    file_data.sort(key=sort_key)

    all_y_values = []
    for x_energy, y_dos, label in file_data:
        color = get_element_color(label, sorted_elements, custom_colors)
        mask = (x_energy >= emin) & (x_energy <= emax)
        xe = x_energy[mask]
        ye = y_dos[mask]

        fig.add_trace(go.Scatter(
            x=ye, y=xe,  # rotated: DOS on x, Energy on y
            mode="lines", name=label,
            line=dict(color=color, width=line_width),
            hoverinfo="skip",
        ))
        all_y_values.extend(ye)

    max_dos = max(all_y_values) if all_y_values else 1
    buffer = 0.05 * max_dos
    x_max = max_dos + buffer

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
    """Plot COHP with rotated axes: x=COHP (horizontal), y=Energy (vertical)."""
    cohp_files = sorted(glob.glob(os.path.join(calc_dir, "COHP_*.csv")))
    if not cohp_files:
        return None

    pair_names_normalized = []
    pair_name_map = {}  # raw filename -> normalized label
    for f in cohp_files:
        raw = os.path.basename(f).replace("COHP_", "").replace(".csv", "")
        elems = extract_elements(raw.replace("_", "-").replace("-", "/"))
        sorted_pair = sort_by_mendeleev(elems)
        norm = "-".join(sorted_pair)
        pair_names_normalized.append(norm)
        pair_name_map[raw] = norm

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

        if custom_colors and pair_label in custom_colors:
            cohp_color = custom_colors[pair_label]
        else:
            # Find this pair's index in the full pairs list for consistent palette assignment
            pair_idx = pair_names_normalized.index(pair_label) if pair_label in pair_names_normalized else i
            cohp_color = PAIR_COLOR_PALETTE[pair_idx % len(PAIR_COLOR_PALETTE)]

        mask = (df["Energy (eV)"] >= emin) & (df["Energy (eV)"] <= emax)
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "COHP"].values,
            y=df.loc[mask, "Energy (eV)"].values,
            mode="lines", name=pair_label,
            line=dict(color=cohp_color, width=line_width),
            hoverinfo="skip",
        ))

        all_x_values.extend(np.abs(df.loc[mask, "COHP"].values))

        if "Int. COHP" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.loc[mask, "Int. COHP"].values,
                y=df.loc[mask, "Energy (eV)"].values,
                mode="lines", name=f"{pair_label} (iCOHP)",
                line=dict(color=cohp_color, width=max(line_width - 2, 1), dash="dash"),
                hoverinfo="skip",
            ))
            all_x_values.extend(np.abs(df.loc[mask, "Int. COHP"].values))

    max_x = max(all_x_values) if all_x_values else 1
    buffer = 0.05 * max_x
    fig.update_xaxes(
        title=dict(text="-COHP", font=dict(size=14)),
        tickfont=dict(size=14),
        range=[-(max_x + buffer), max_x + buffer],
    )

    fig.update_yaxes(
        title=dict(text="energy (eV)", font=dict(size=14)),
        tickfont=dict(size=14),
        range=[emin, emax],
        dtick=1,
    )

    fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=3, line_dash="dash", line_color="black")

    fig.add_annotation(
        x=max_x + buffer,
        y=0,
        text="EF",
        showarrow=False,
        font=dict(size=14, color="black"),
        xref="x", yref="y",
        xshift=15, yshift=0,
    )

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


# ---- selector helpers ----

def _unique_sorted(df, col):
    """Return sorted unique non-null values for a column."""
    vals = df[col].dropna().unique()
    numeric = _is_numeric(df, col)
    if numeric:
        return sorted(float(v) for v in vals)
    return sorted(str(v) for v in vals)


def _is_numeric(df, col):
    """Check if a column is numeric."""
    return pd.api.types.is_numeric_dtype(df[col])


def _filter(df, col, val):
    """Filter DataFrame rows matching *val* in *col*."""
    if _is_numeric(df, col):
        try:
            v_num = float(val)
            return df[(df[col] - v_num).abs() < 1e-6]
        except (ValueError, TypeError):
            return df[df[col].astype(str) == str(val)]
    return df[df[col] == val]


def _fmt_val(v):
    """Format a numeric value for display (4 decimal places for floats)."""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def render_cascade_selectors(
    df: pd.DataFrame,
    param_cols: list[dict] = PARAM_COLS,
    cols_per_row: int = 2,
):
    """
    Generalized cascade selector for any number of numeric columns.

    param_cols entries: {"col": str, "label": str, "param": str}

    Color scheme:
      - Selected     : blue   #2255AA  bold + underline
      - Compatible   : purple #9B59B6  (exists in rows matching all OTHER selections)
      - Incompatible : grey   #AAAAAA
    """
    SEL_COLOR      = "#2255AA"
    COMPAT_COLOR   = "#9B59B6"
    INCOMPAT_COLOR = "#AAAAAA"
    separator = ' <span style="color:#CCCCCC;">|</span> '

    qp       = st.query_params
    all_vals = {pc["col"]: _unique_sorted(df, pc["col"]) for pc in param_cols}

    # ── Read / resolve selections in column order ──────────────────────────
    # Each selection is resolved against the progressively narrowed df so
    # that defaults remain consistent even when query params are missing.
    sel: dict[str, float] = {}
    narrowed = df
    for pc in param_cols:
        avail   = _unique_sorted(narrowed, pc["col"])
        default = avail[0] if avail else all_vals[pc["col"]][0]
        try:
            v = float(qp.get(pc["param"], default))
            if v not in all_vals[pc["col"]]:
                v = default
        except (ValueError, TypeError):
            v = default
        sel[pc["param"]] = v
        narrowed = _filter(narrowed, pc["col"], v)

    # ── Compatible set for column i = values present when filtering by all
    #    OTHER currently-selected columns ───────────────────────────────────
    def _compatible_for(pc_col: str) -> set:
        tmp = df
        for pc in param_cols:
            if pc["col"] != pc_col:
                tmp = _filter(tmp, pc["col"], sel[pc["param"]])
        return set(_unique_sorted(tmp, pc_col))

    # ── Build href when user clicks value v in column pc_idx ──────────────
    # Keep all other selections; snap any that become incompatible to their
    # first compatible value (resolved left-to-right through remaining cols).
    def _href_for(pc_idx: int, v: float) -> str:
        new_sel = dict(sel)
        new_sel[param_cols[pc_idx]["param"]] = v

        tmp = _filter(df, param_cols[pc_idx]["col"], v)
        for pc in param_cols:
            if pc["col"] == param_cols[pc_idx]["col"]:
                continue
            avail = _unique_sorted(tmp, pc["col"])
            if new_sel[pc["param"]] not in avail:
                new_sel[pc["param"]] = avail[0] if avail else new_sel[pc["param"]]
            tmp = _filter(tmp, pc["col"], new_sel[pc["param"]])

        return "?" + "&".join(
            f"{pc['param']}={new_sel[pc['param']]}" for pc in param_cols
        )

    def _link(val, is_selected: bool, is_compatible: bool, href: str) -> str:
        if is_selected:
            style = (f"color:{SEL_COLOR}; font-weight:bold; "
                     f"text-decoration:underline; font-size:0.95rem;")
        elif is_compatible:
            style = (f"color:{COMPAT_COLOR}; font-weight:normal; "
                     f"text-decoration:none; font-size:0.95rem;")
        else:
            style = (f"color:{INCOMPAT_COLOR}; font-weight:normal; "
                     f"text-decoration:none; font-size:0.95rem;")
        label = _fmt_val(val)
        return f'<a href="{href}" style="{style}" title="{label}">{label}</a>'

    # ── Render in rows of cols_per_row ─────────────────────────────────────
    for row_start in range(0, len(param_cols), cols_per_row):
        row_pcs  = param_cols[row_start : row_start + cols_per_row]
        ui_cols  = st.columns(len(row_pcs))
        for pc, ui_col in zip(row_pcs, ui_cols):
            compatible = _compatible_for(pc["col"])
            idx        = param_cols.index(pc)
            links = [
                _link(
                    v,
                    is_selected  = (v == sel[pc["param"]]),
                    is_compatible= (v in compatible),
                    href         = _href_for(idx, v),
                )
                for v in all_vals[pc["col"]]
            ]
            with ui_col:
                st.markdown(f"**{pc['label']}**")
                st.markdown(separator.join(links), unsafe_allow_html=True)

    # ── Resolve matched calc ───────────────────────────────────────────────
    matched = df
    for pc in param_cols:
        matched = _filter(matched, pc["col"], sel[pc["param"]])

    if len(matched) > 0:
        return CALCS[matched["sample"].iloc[0]]

    # Fallback: relax to first compatible row for the leading column
    fallback = _filter(df, param_cols[0]["col"], sel[param_cols[0]["param"]])
    if len(fallback) > 0:
        return CALCS[fallback["sample"].iloc[0]]

    st.warning("No matching sample found for the current selection.")
    st.stop()


# ---- app logic ----

st.set_page_config(page_title="Structure Viz", layout="wide")
st.title("Electronic Structure Visualizer")

CALCS = scan_calculations()

if not CALCS:
    st.error(f"No calculations found in {DATA_DIR}.")
    st.stop()

st.info(f"Loaded {len(CALCS)} calculations")

# ---- build DataFrame for selectors ----
calc_df = pd.DataFrame([{"sample": v["sample"], "A": v["A"], "C": v["C"], 
                         "Yb1_x": v["Yb1_x"], "Yb1_y": v["Yb1_y"], "Yb1_z": v["Yb1_z"],
                         "Yb2_x": v["Yb2_x"], "Yb2_y": v["Yb2_y"], "Yb2_z": v["Yb2_z"],
                         "Yb3_x": v["Yb3_x"], "Yb3_y": v["Yb3_y"], "Yb3_z": v["Yb3_z"],
                         "Yb4_x": v["Yb4_x"], "Yb4_y": v["Yb4_y"], "Yb4_z": v["Yb4_z"],

                         "Sb1_x": v["Sb1_x"], "Sb1_y": v["Sb1_y"], "Sb1_z": v["Sb1_z"],
                         "Sb2_x": v["Sb2_x"], "Sb2_y": v["Sb2_y"], "Sb2_z": v["Sb2_z"],
                         "Sb3_x": v["Sb3_x"], "Sb3_y": v["Sb3_y"], "Sb3_z": v["Sb3_z"],
                         "Sb4_x": v["Sb4_x"], "Sb4_y": v["Sb4_y"], "Sb4_z": v["Sb4_z"],
                         } for v in CALCS.values()])

ss = st.session_state

# ---- render selectors and get matched calc ----
calc = render_cascade_selectors(calc_df)
st.info(f"Showing {calc}")

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
        
    st.expander("")
    st.expander("")

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
    st.expander("")
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