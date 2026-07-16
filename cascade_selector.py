"""Cascading filter selector for Streamlit.

Renders dropdown selects stacked vertically (one per column).
Changes trigger Streamlit rerun which recomputes availability for all selectors.
When a later column's selection becomes unavailable, the first value of the
preceding column that makes it available is auto-selected (reverse cascade).
"""

import pandas as pd
import streamlit as st


def cascade_selector(df: pd.DataFrame, filter_cols: list, sample_col: str = "sample"):
    """Render cascading dropdown selectors with availability value strips.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *filter_cols* and *sample_col*.
    filter_cols : list[str]
        Columns to create selectors for (e.g. ["A", "C"]).
    sample_col : str
        Column name identifying each row / sample.

    Returns
    -------
    matched_sample or None when no rows match all selections.
    """
    if df.empty:
        st.warning("No data to display.")
        return None

    # Remove empty columns early
    filter_cols = [c for c in filter_cols if c in df.columns and not df[c].dropna().empty]
    if not filter_cols:
        st.warning(f"No valid filter columns found. Expected: {filter_cols}")
        return None

    ss = st.session_state

    # ---- initialise session-state defaults (first value per column) ----
    for col in filter_cols:
        key = f"_csel_{col}"
        if key not in ss:
            vals = _unique_sorted(df, col)
            ss[key] = vals[0] if vals else None

    # ---- current selections ----
    selections = {col: ss[f"_csel_{col}"] for col in filter_cols}

    # ---- compute availability via cascaded filtering (forward) ----
    running_df = df.copy()
    availability = {}
    for col in filter_cols:
        v = selections[col]
        avail = _unique_sorted(running_df, col)
        availability[col] = avail

        # Cascade into next column
        if v is not None and avail:
            running_df = _filter(running_df, col, v)

    # ---- reverse cascade: check each selection's viability ----
    # Process columns in reverse order so that picking a C value can auto-pick a compatible A
    for i in range(len(filter_cols) - 1, -1, -1):
        col = filter_cols[i]
        key = f"_csel_{col}"
        v = selections[col]

        # Compute values of this column that are available when earlier columns' selections are applied
        fwd_df = df.copy()
        for j in range(i):
            prev_v = ss[f"_csel_{filter_cols[j]}"]
            if prev_v is not None:
                fwd_df = _filter(fwd_df, filter_cols[j], prev_v)
        fwd_avail = _unique_sorted(fwd_df, col)

        if v is not None and v not in fwd_avail:
            # Auto-pick first available value of this column given earlier selections
            ss[key] = fwd_avail[0] if fwd_avail else None
            selections[col] = ss[key]

    # ---- forward cascade: reset later columns if their selection became stale ----
    running_df = df.copy()
    for col in filter_cols:
        key = f"_csel_{col}"
        v = ss[f"_csel_{col}"]
        avail = _unique_sorted(running_df, col)

        cur = ss[key]
        if cur is not None and cur not in avail:
            ss[key] = avail[0] if avail else None
            selections[col] = ss[key]

        if ss[f"_csel_{col}"] is not None and avail:
            running_df = _filter(running_df, col, ss[f"_csel_{col}"])

    # ---- render selectors stacked vertically ----
    for col in filter_cols:
        st.markdown(f"**{col}**")

        # Value strip: show all unique values with color coding (before the selector)
        all_vals = _unique_sorted(df, col)
        avail_vals = availability[col]
        sel_val = ss[f"_csel_{col}"]
        _render_value_strip(col, all_vals, avail_vals, sel_val)

        key = f"_csel_{col}"
        fmt_func = lambda x, c=col: f"{x:.4f}" if _is_numeric(df, c) else str(x)
        options = [fmt_func(v) for v in avail_vals]

        chosen = st.segmented_control(
            "Select",
            options=options,
            default=options[0] if (not avail_vals or sel_val not in avail_vals) else options[avail_vals.index(sel_val)],
            key=f"{key}_ui",
        )


        # Map label back to original value
        if chosen is not None:
            idx = options.index(chosen)
            ss[key] = avail_vals[idx] if idx < len(avail_vals) else None

    # ---- compute matched sample ----
    matched_df = df.copy()
    for col in filter_cols:
        v = selections[col]
        if v is not None:
            matched_df = _filter(matched_df, col, v)

    return matched_df[sample_col].iloc[0] if len(matched_df) > 0 else None


# ---- internal helpers ----

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


def _render_value_strip(col, all_vals, avail_vals, selected_val):
    """Render a horizontal strip of value pills showing availability.

    Purple + bold = available (and selected)
    Purple = available but not selected
    Grey = unavailable
    """
    if not all_vals:
        return

    avail_set = set(avail_vals)
    is_numeric = isinstance(all_vals[0], (int, float))

    pills = []
    for v in all_vals:
        label_text = f"{v:.4f}" if is_numeric else str(v)

        if v == selected_val and v in avail_set:
            # Selected: dark background, white text, bold
            pills.append(
                f'<span style="display:inline-block;background:#6D28D9;color:white;'
                f'font-weight:bold;padding:1px 5px;border-radius:3px;font-size:11px;'
                f'margin:1px 2px;white-space:nowrap;">{label_text}</span>'
            )
        elif v in avail_set:
            # Available: light purple background
            pills.append(
                f'<span style="display:inline-block;background:#EDE9FE;color:#6D28D9;'
                f'padding:1px 5px;border-radius:3px;font-size:11px;margin:1px 2px;'
                f'white-space:nowrap;">{label_text}</span>'
            )
        else:
            # Unavailable: grey
            pills.append(
                f'<span style="display:inline-block;background:#E5E7EB;color:#9CA3AF;'
                f'padding:1px 5px;border-radius:3px;font-size:11px;margin:1px 2px;'
                f'white-space:nowrap;">{label_text}</span>'
            )

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;margin:4px 0;padding:2px;">'
        f"{''.join(pills)}</div>",
        unsafe_allow_html=True,
    )


__all__ = ["cascade_selector"]
