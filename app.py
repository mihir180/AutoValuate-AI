import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    format_inr,
    calculate_emi,
    estimate_fuel_economy_kmpl,
    load_data_and_model,
    predict_price,
    get_valuation_range,
    estimate_running_cost,
    create_brand_price_chart,
    create_depreciation_chart,
    create_kms_vs_price_chart,
    create_fuel_distribution_chart,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoValuate — Used Car Valuation",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Theme & CSS
# ═══════════════════════════════════════════════════════════════════════════════

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"


def _css(is_dark: bool) -> str:
    """Return the full custom stylesheet for light or dark mode."""
    if is_dark:
        bg = "#1A1A18"
        surface = "#242422"
        surface_alt = "#2E2E2B"
        text = "#E8E7E3"
        text2 = "#9B9A96"
        text3 = "#6D6D69"
        accent = "#C94F47"
        accent_hover = "#D4645C"
        border = "#3A3A37"
        input_bg = "#2E2E2B"
        tag_bg = "#2E2E2B"
    else:
        bg = "#F7F6F2"
        surface = "#FFFFFF"
        surface_alt = "#F0EFEB"
        text = "#20211F"
        text2 = "#777873"
        text3 = "#A3A29E"
        accent = "#B83A32"
        accent_hover = "#9C302A"
        border = "#E3E1DA"
        input_bg = "#FFFFFF"
        tag_bg = "#F0EFEB"

    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Base ───────────────────────────────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {text};
    }}
    .stApp {{
        background-color: {bg} !important;
    }}

    /* ── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {surface} !important;
        border-right: 1px solid {border};
    }}

    /* ── Top wordmark bar ───────────────────────────────────────────── */
    .av-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 20px 0;
        border-bottom: 1px solid {border};
        margin-bottom: 24px;
    }}
    .av-wordmark {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text};
        letter-spacing: -0.01em;
    }}
    .av-wordmark span {{
        color: {accent};
    }}
    .av-tagline {{
        font-size: 0.8rem;
        color: {text2};
        margin-top: 1px;
    }}

    /* ── Tab navigation ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid {border};
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 10px 20px;
        font-size: 0.85rem;
        font-weight: 500;
        color: {text2};
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {accent} !important;
        border-bottom: 2px solid {accent} !important;
        background: transparent !important;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {text} !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* ── Buttons ─────────────────────────────────────────────────────── */
    div[data-testid="stButton"] > button,
    .stButton > button {{
        background-color: {accent} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.3rem !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: background-color 0.15s ease !important;
    }}
    div[data-testid="stButton"] > button:hover,
    .stButton > button:hover {{
        background-color: {accent_hover} !important;
    }}
    /* secondary / download buttons */
    div[data-testid="stDownloadButton"] > button {{
        background-color: transparent !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: {surface_alt} !important;
    }}

    /* ── Inputs ──────────────────────────────────────────────────────── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background-color: {input_bg} !important;
        border-color: {border} !important;
        border-radius: 6px !important;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {accent} !important;
    }}
    .stSlider [data-testid="stTickBar"] {{
        display: none;
    }}

    /* ── Section label ──────────────────────────────────────────────── */
    .av-section-label {{
        font-size: 0.72rem;
        font-weight: 600;
        color: {text3};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
    }}

    /* ── Valuation result ───────────────────────────────────────────── */
    .av-result {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 28px 24px;
    }}
    .av-result-vehicle {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {text};
        margin-bottom: 2px;
    }}
    .av-result-specs {{
        font-size: 0.82rem;
        color: {text2};
        margin-bottom: 20px;
    }}
    .av-result-price {{
        font-size: 2.4rem;
        font-weight: 700;
        color: {text};
        letter-spacing: -0.02em;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }}
    .av-result-caption {{
        font-size: 0.82rem;
        color: {text2};
        margin-top: 4px;
        margin-bottom: 20px;
    }}

    /* tier row */
    .av-tiers {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1px;
        background: {border};
        border: 1px solid {border};
        border-radius: 6px;
        overflow: hidden;
    }}
    .av-tier {{
        background: {surface};
        padding: 14px 12px;
        text-align: center;
    }}
    .av-tier-label {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {text3};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }}
    .av-tier-value {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {text};
        font-variant-numeric: tabular-nums;
    }}
    .av-tier-sub {{
        font-size: 0.7rem;
        color: {text2};
        margin-top: 2px;
    }}

    /* ── Tags ────────────────────────────────────────────────────────── */
    .av-tags {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }}
    .av-tag {{
        background: {tag_bg};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 0.78rem;
        color: {text2};
    }}
    .av-tag strong {{
        color: {text};
        font-weight: 600;
    }}

    /* ── Finance summary ────────────────────────────────────────────── */
    .av-fin-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: {border};
        border: 1px solid {border};
        border-radius: 6px;
        overflow: hidden;
        margin-top: 12px;
    }}
    @media (max-width: 768px) {{
        .av-fin-grid {{ grid-template-columns: repeat(2, 1fr); }}
        .av-tiers {{ grid-template-columns: 1fr; }}
    }}
    .av-fin-cell {{
        background: {surface};
        padding: 16px 14px;
    }}
    .av-fin-label {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {text3};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }}
    .av-fin-value {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {text};
        font-variant-numeric: tabular-nums;
    }}
    .av-fin-sub {{
        font-size: 0.72rem;
        color: {text2};
        margin-top: 2px;
    }}

    /* ── Compare card ───────────────────────────────────────────────── */
    .av-compare-result {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 20px;
        margin-top: 12px;
    }}
    .av-compare-price {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {accent};
        font-variant-numeric: tabular-nums;
    }}
    .av-compare-detail {{
        font-size: 0.82rem;
        color: {text2};
        margin-top: 4px;
    }}

    /* ── Info note ───────────────────────────────────────────────────── */
    .av-note {{
        background: {surface_alt};
        border-left: 3px solid {border};
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.8rem;
        color: {text2};
        line-height: 1.55;
        margin: 14px 0;
    }}

    /* ── Comparison table ───────────────────────────────────────────── */
    .av-compare-summary {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 16px 20px;
        font-size: 0.88rem;
        color: {text};
    }}
    .av-compare-summary strong {{
        color: {accent};
    }}

    /* ── Footer ──────────────────────────────────────────────────────── */
    .av-footer {{
        border-top: 1px solid {border};
        padding: 20px 0 8px 0;
        margin-top: 48px;
        font-size: 0.75rem;
        color: {text3};
        line-height: 1.7;
    }}

    /* ── Hide default Streamlit footer & header ──────────────────────── */
    header[data-testid="stHeader"] {{
        background: {bg} !important;
    }}
    footer {{
        display: none !important;
    }}

    /* ── Dataframe styling ───────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: 6px;
        overflow: hidden;
    }}
</style>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Load resources
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading dataset and model…")
def get_cached_resources():
    return load_data_and_model()


try:
    df, model = get_cached_resources()
except Exception as e:
    st.error(f"Failed to load data or model: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — minimal: theme toggle only
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("#### Appearance")
    theme_choice = st.radio(
        "Theme",
        ["Light", "Dark"],
        index=0 if st.session_state["theme"] == "light" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    is_dark = theme_choice == "Dark"
    st.session_state["theme"] = "dark" if is_dark else "light"
    chart_template = "plotly_dark" if is_dark else "plotly_white"

    st.divider()
    st.caption(
        "AutoValuate uses a linear regression model trained on "
        f"~{len(df):,} cleaned listings from the Quikr used car dataset. "
        "Predictions are estimates, not guaranteed prices."
    )

# Inject CSS after theme is known
st.markdown(_css(is_dark), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Wordmark header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="av-topbar">
    <div>
        <div class="av-wordmark">Auto<span>Valuate</span></div>
        <div class="av-tagline">Used car valuation & market analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Navigation
# ═══════════════════════════════════════════════════════════════════════════════
tab_est, tab_insights, tab_compare, tab_inventory, tab_method = st.tabs([
    "Estimate",
    "Market Insights",
    "Compare Cars",
    "Inventory",
    "Methodology",
])

# ─── Shared data ────────────────────────────────────────────────────────────
companies = sorted(df["company"].unique().tolist())
fuel_types = sorted(df["fuel_type"].unique().tolist())
min_year = int(df["year"].min())
max_year = int(df["year"].max())
median_year = int(df["year"].median())
median_kms = int(df["kms_driven"].median())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Estimate
# ═══════════════════════════════════════════════════════════════════════════════
with tab_est:
    st.markdown("## Find out what your car is worth")
    st.markdown(
        "Get an estimated resale value based on your car's details "
        "and the available used-car dataset."
    )
    st.markdown("---")

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        # ── Vehicle details ──────────────────────────────────────────────
        st.markdown('<div class="av-section-label">Vehicle details</div>', unsafe_allow_html=True)

        default_company = companies.index("Maruti") if "Maruti" in companies else 0
        selected_company = st.selectbox(
            "Manufacturer",
            companies,
            index=default_company,
            key="est_company",
        )

        available_models = sorted(
            df[df["company"] == selected_company]["name"].unique().tolist()
        )
        selected_model = st.selectbox(
            "Model",
            available_models,
            key="est_model",
        )

        c_yr, c_fuel = st.columns(2)
        with c_yr:
            selected_year = st.slider(
                "Model year",
                min_value=min_year,
                max_value=max_year,
                value=min(median_year, max_year),
                key="est_year",
            )
        with c_fuel:
            default_fuel = fuel_types.index("Petrol") if "Petrol" in fuel_types else 0
            selected_fuel = st.radio(
                "Fuel type",
                fuel_types,
                index=default_fuel,
                horizontal=True,
                key="est_fuel",
            )

        # ── Usage ────────────────────────────────────────────────────────
        st.markdown('<div class="av-section-label" style="margin-top:20px;">Usage</div>', unsafe_allow_html=True)

        selected_kms = st.number_input(
            "Kilometres driven",
            min_value=100,
            max_value=400000,
            value=min(median_kms, 400000),
            step=1000,
            key="est_kms",
            help="Total distance shown on the odometer",
        )

        presets = {"25,000 km": 25000, "55,000 km": 55000, "95,000 km": 95000}
        preset_cols = st.columns(len(presets))
        for col, (label, val) in zip(preset_cols, presets.items()):
            with col:
                if st.button(label, key=f"preset_{val}", use_container_width=True):
                    st.session_state["est_kms"] = val
                    st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        estimate_btn = st.button("Estimate resale value", use_container_width=True, key="btn_estimate")

    # ── Results ──────────────────────────────────────────────────────────
    with col_result:
        should_calculate = estimate_btn or "valuation" not in st.session_state

        if should_calculate:
            with st.spinner("Calculating…"):
                raw_pred = predict_price(
                    model, selected_company, selected_model,
                    selected_year, selected_kms, selected_fuel,
                )
                tiers = get_valuation_range(raw_pred)
                fuel_eff = estimate_fuel_economy_kmpl(selected_model, selected_company, selected_fuel)
                cost_km, cost_1k = estimate_running_cost(selected_fuel, fuel_eff)

                st.session_state["valuation"] = {
                    "price": raw_pred,
                    "tiers": tiers,
                    "company": selected_company,
                    "model": selected_model,
                    "year": selected_year,
                    "kms": selected_kms,
                    "fuel": selected_fuel,
                    "kmpl": fuel_eff,
                    "cost_km": cost_km,
                    "cost_1k": cost_1k,
                }

        val = st.session_state.get("valuation")
        if val:
            t = val["tiers"]
            label_color = '#9B9A96' if is_dark else '#A3A29E'
            st.markdown(f"""<div class="av-result">
<div class="av-result-vehicle">{val['year']} {val['model']}</div>
<div class="av-result-specs">{val['company']}  ·  {val['fuel']}  ·  {val['kms']:,} km</div>
<div style="font-size:0.7rem;font-weight:600;color:{label_color};text-transform:uppercase;letter-spacing:0.05em;">Estimated resale value</div>
<div class="av-result-price">{format_inr(t['fair_market'])}</div>
<div class="av-result-caption">Indicative range: {format_inr(t['fair_range_low'])} – {format_inr(t['fair_range_high'])}</div>
<div class="av-tiers">
<div class="av-tier">
<div class="av-tier-label">Trade-in estimate</div>
<div class="av-tier-value">{format_inr(t['trade_in_low'])} – {format_inr(t['trade_in_high'])}</div>
<div class="av-tier-sub">If selling to a dealer</div>
</div>
<div class="av-tier">
<div class="av-tier-label">Private sale</div>
<div class="av-tier-value">{format_inr(t['fair_market'])}</div>
<div class="av-tier-sub">Direct buyer-to-seller</div>
</div>
<div class="av-tier">
<div class="av-tier-label">Dealer retail</div>
<div class="av-tier-value">{format_inr(t['retail_low'])} – {format_inr(t['retail_high'])}</div>
<div class="av-tier-sub">Dealer showroom price</div>
</div>
</div>
<div class="av-tags">
<div class="av-tag">Est. fuel economy: <strong>{val['kmpl']} km/L</strong></div>
<div class="av-tag">Est. fuel cost: <strong>₹{val['cost_km']}/km</strong></div>
</div>
</div>""", unsafe_allow_html=True)

            st.markdown("""<div class="av-note">
This is a statistical estimate based on a linear regression model trained on the
Quikr used-car dataset. It is not a guaranteed price, formal appraisal, or inspection
report. Actual sale prices depend on vehicle condition, location, demand, and other
factors not captured by this model.
</div>""", unsafe_allow_html=True)

    # ── EMI calculator ───────────────────────────────────────────────────
    if st.session_state.get("valuation"):
        val = st.session_state["valuation"]
        pred_val = val["price"]

        st.markdown("---")
        st.markdown("### Financing estimate")
        st.caption(
            "Illustrative EMI calculation. Does not include insurance, "
            "registration, processing fees, or other charges."
        )

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            down_payment = st.slider(
                "Down payment",
                min_value=0,
                max_value=int(pred_val),
                value=int(pred_val * 0.20),
                step=10000,
                format="₹%d",
                key="emi_down",
            )
        with fc2:
            tenure = st.selectbox(
                "Loan term",
                [1, 2, 3, 4, 5, 7],
                index=2,
                format_func=lambda x: f"{x} {'year' if x == 1 else 'years'}",
                key="emi_tenure",
            )
        with fc3:
            rate = st.slider(
                "Interest rate (% p.a.)",
                min_value=6.0, max_value=16.0,
                value=8.5, step=0.25,
                key="emi_rate",
            )

        principal = max(0, pred_val - down_payment)
        emi, total_interest = calculate_emi(principal, rate, tenure)
        total_repay = principal + total_interest

        st.markdown(f"""
        <div class="av-fin-grid">
            <div class="av-fin-cell">
                <div class="av-fin-label">Loan amount</div>
                <div class="av-fin-value">{format_inr(principal)}</div>
                <div class="av-fin-sub">After {format_inr(down_payment)} down</div>
            </div>
            <div class="av-fin-cell">
                <div class="av-fin-label">Monthly EMI</div>
                <div class="av-fin-value">{format_inr(emi)}</div>
                <div class="av-fin-sub">{tenure * 12} payments</div>
            </div>
            <div class="av-fin-cell">
                <div class="av-fin-label">Total interest</div>
                <div class="av-fin-value">{format_inr(total_interest)}</div>
                <div class="av-fin-sub">At {rate:.2f}% p.a.</div>
            </div>
            <div class="av-fin-cell">
                <div class="av-fin-label">Total repayment</div>
                <div class="av-fin-value">{format_inr(total_repay)}</div>
                <div class="av-fin-sub">Principal + interest</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Market Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("## Market insights")
    st.markdown(
        f"Summary statistics and charts derived from **{len(df):,} listings** "
        f"across **{df['company'].nunique()} manufacturers** in the Quikr dataset."
    )
    st.markdown("---")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Median price", format_inr(df["Price"].median()))
    with k2:
        st.metric("Mean price", format_inr(df["Price"].mean()))
    with k3:
        st.metric("Avg. kilometres", f"{int(df['kms_driven'].mean()):,} km")
    with k4:
        top_brand = df["company"].value_counts().idxmax()
        top_share = round(df["company"].value_counts().max() / len(df) * 100, 1)
        st.metric("Most listed brand", f"{top_brand} ({top_share}%)")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Charts — row 1
    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Average listing price by manufacturer.")
        st.plotly_chart(
            create_brand_price_chart(df, template=chart_template, is_dark=is_dark),
            use_container_width=True,
        )
    with ch2:
        st.caption("How listing prices vary across model years.")
        st.plotly_chart(
            create_depreciation_chart(df, template=chart_template, is_dark=is_dark),
            use_container_width=True,
        )

    # Charts — row 2
    ch3, ch4 = st.columns(2)
    with ch3:
        st.caption("Relationship between odometer reading and listed price.")
        st.plotly_chart(
            create_kms_vs_price_chart(df, template=chart_template, is_dark=is_dark),
            use_container_width=True,
        )
    with ch4:
        st.caption("Share of listings by fuel type.")
        st.plotly_chart(
            create_fuel_distribution_chart(df, template=chart_template, is_dark=is_dark),
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Compare Cars
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## Compare two cars")
    st.markdown("Select two vehicles to compare their estimated values and running costs.")
    st.markdown("---")

    cmp1, cmp2 = st.columns(2, gap="large")

    # ── Vehicle A ────────────────────────────────────────────────────────
    with cmp1:
        st.markdown('<div class="av-section-label">Vehicle A</div>', unsafe_allow_html=True)
        a_company = st.selectbox(
            "Manufacturer",
            companies,
            index=companies.index("Hyundai") if "Hyundai" in companies else 0,
            key="cmp_a_co",
        )
        a_models = sorted(df[df["company"] == a_company]["name"].unique().tolist())
        a_model = st.selectbox("Model", a_models, key="cmp_a_mod")
        a_yr = st.slider("Model year", min_year, max_year, min(2016, max_year), key="cmp_a_yr")
        a_kms = st.number_input("Kilometres driven", 100, 350000, 35000, step=5000, key="cmp_a_km")
        a_fuel = st.selectbox("Fuel type", fuel_types, key="cmp_a_fl")

        price_a = predict_price(model, a_company, a_model, a_yr, a_kms, a_fuel)
        kmpl_a = estimate_fuel_economy_kmpl(a_model, a_company, a_fuel)
        cost_a, _ = estimate_running_cost(a_fuel, kmpl_a)

        st.markdown(f"""
        <div class="av-compare-result">
            <div class="av-compare-price">{format_inr(price_a)}</div>
            <div class="av-compare-detail">
                {a_model} ({a_yr}) · Est. {kmpl_a} km/L · ₹{cost_a}/km
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Vehicle B ────────────────────────────────────────────────────────
    with cmp2:
        st.markdown('<div class="av-section-label">Vehicle B</div>', unsafe_allow_html=True)
        b_company = st.selectbox(
            "Manufacturer",
            companies,
            index=companies.index("Honda") if "Honda" in companies else min(1, len(companies) - 1),
            key="cmp_b_co",
        )
        b_models = sorted(df[df["company"] == b_company]["name"].unique().tolist())
        b_model = st.selectbox("Model", b_models, key="cmp_b_mod")
        b_yr = st.slider("Model year", min_year, max_year, min(2018, max_year), key="cmp_b_yr")
        b_kms = st.number_input("Kilometres driven", 100, 350000, 48000, step=5000, key="cmp_b_km")
        b_fuel = st.selectbox("Fuel type", fuel_types, key="cmp_b_fl")

        price_b = predict_price(model, b_company, b_model, b_yr, b_kms, b_fuel)
        kmpl_b = estimate_fuel_economy_kmpl(b_model, b_company, b_fuel)
        cost_b, _ = estimate_running_cost(b_fuel, kmpl_b)

        st.markdown(f"""
        <div class="av-compare-result">
            <div class="av-compare-price">{format_inr(price_b)}</div>
            <div class="av-compare-detail">
                {b_model} ({b_yr}) · Est. {kmpl_b} km/L · ₹{cost_b}/km
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Comparison table ─────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    comparison_data = {
        "": ["Estimated price", "Model year", "Kilometres driven", "Fuel type",
             "Est. fuel economy", "Est. fuel cost"],
        f"{a_model}": [
            format_inr(price_a), str(a_yr), f"{a_kms:,} km", a_fuel,
            f"{kmpl_a} km/L", f"₹{cost_a}/km",
        ],
        f"{b_model}": [
            format_inr(price_b), str(b_yr), f"{b_kms:,} km", b_fuel,
            f"{kmpl_b} km/L", f"₹{cost_b}/km",
        ],
    }
    st.dataframe(
        pd.DataFrame(comparison_data).set_index(""),
        use_container_width=True,
    )

    # Summary
    price_diff = abs(price_a - price_b)
    cheaper = "Vehicle A" if price_a < price_b else "Vehicle B"
    more_eff = "Vehicle A" if kmpl_a > kmpl_b else "Vehicle B"
    kmpl_diff = abs(kmpl_a - kmpl_b)

    st.markdown(f"""
    <div class="av-compare-summary">
        <strong>{cheaper}</strong> is estimated at {format_inr(price_diff)} less.
        <strong>{more_eff}</strong> has a higher estimated fuel economy by {kmpl_diff:.1f} km/L.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Inventory
# ═══════════════════════════════════════════════════════════════════════════════
with tab_inventory:
    st.markdown("## Vehicle inventory")
    st.markdown(
        f"Browse and filter **{len(df):,} listings** from the dataset. "
        "Prices are listed amounts, not verified sale prices."
    )
    st.markdown("---")

    # Filters
    fi1, fi2, fi3 = st.columns(3)
    with fi1:
        search_text = st.text_input(
            "Search by model or manufacturer",
            key="inv_search",
            placeholder="e.g. Swift, Hyundai…",
        )
    with fi2:
        brand_filter = st.multiselect("Manufacturer", companies, default=[], key="inv_brand")
    with fi3:
        fuel_filter = st.multiselect("Fuel type", fuel_types, default=[], key="inv_fuel")

    max_price = int(df["Price"].max())
    price_range = st.slider(
        "Price range",
        min_value=0,
        max_value=max_price,
        value=(0, max_price),
        step=50000,
        format="₹%d",
        key="inv_price",
    )

    # Apply filters
    filtered = df.copy()
    if search_text:
        q = search_text.lower()
        filtered = filtered[
            filtered["name"].str.lower().str.contains(q, na=False)
            | filtered["company"].str.lower().str.contains(q, na=False)
        ]
    if brand_filter:
        filtered = filtered[filtered["company"].isin(brand_filter)]
    if fuel_filter:
        filtered = filtered[filtered["fuel_type"].isin(fuel_filter)]
    filtered = filtered[
        (filtered["Price"] >= price_range[0]) & (filtered["Price"] <= price_range[1])
    ]

    st.caption(f"{len(filtered):,} listings match your filters")

    # Display table
    display = filtered.copy()
    display["Price (₹)"] = display["Price"].apply(format_inr)
    display["Kilometres"] = display["kms_driven"].apply(lambda x: f"{int(x):,} km")
    display["Est. km/L"] = display["fuel_efficiency_kmpl"].apply(lambda x: f"{x} km/L")
    display = display[["name", "company", "year", "fuel_type", "Kilometres", "Est. km/L", "Price (₹)"]]
    display.columns = ["Model", "Manufacturer", "Year", "Fuel", "Kilometres", "Est. km/L", "Price"]

    st.dataframe(display, use_container_width=True, height=440)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data (CSV)",
        data=csv_bytes,
        file_name="autovaluate_listings.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Methodology
# ═══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown("## Methodology")
    st.markdown(
        "How the prediction model works, its limitations, and what the "
        "numbers mean."
    )
    st.markdown("---")

    st.markdown("### Dataset")
    st.markdown(f"""
    The model is trained on a cleaned version of the **Quikr used car listings
    dataset**, containing **{len(df):,} records** across
    **{df['company'].nunique()} manufacturers**.

    Each record includes the car model name, manufacturer, model year,
    odometer reading (km), fuel type, and listed price (₹).

    This is a historical dataset of listing prices, not verified transaction
    prices. Listings may not reflect actual sale amounts.
    """)

    st.markdown("### Data cleaning")
    st.markdown("""
    - Removed listings with non-numeric or missing prices (e.g. "Ask For Price").
    - Stripped units from odometer values and converted to integers.
    - Trimmed extreme outliers (prices above ₹60,00,000) to stabilise
      regression.
    - Normalised car model names to their primary tokens to reduce
      high-cardinality sparsity.
    """)

    st.markdown("### Model")
    st.markdown("""
    The prediction pipeline uses **scikit-learn** with the following
    components:

    1. **OneHotEncoder** on categorical columns (`name`, `company`,
       `fuel_type`).
    2. **Linear Regression** as the estimator.
    3. Numeric features (`year`, `kms_driven`) are passed through unchanged.
    """)

    st.code("""
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_),
     ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

pipe = make_pipeline(column_trans, LinearRegression())
pipe.fit(X_train, y_train)
    """, language="python")

    st.markdown("### Evaluation")
    st.markdown("""
    - **R² score on test data: 0.84** (as reported during training).
    - R² measures how well the model fits the evaluation data. An R² of 0.84
      means the model explains roughly 84% of the variance in the test set's
      prices.
    - **This does not mean any individual prediction is 84% accurate.**
      Predictions for rare models, extreme odometer readings, or unusual
      market conditions may be significantly less accurate.
    - A minimum price floor of ₹25,000 is applied to prevent negative or
      unrealistically low predictions from linear extrapolation.
    """)

    st.markdown("### Fuel economy estimates")
    st.markdown("""
    The fuel economy figures shown in this application are **rough estimates**
    based on publicly available specification ranges for common Indian car
    segments (hatchbacks, sedans, SUVs, etc.).

    They are **not** certified ARAI test results for any specific vehicle.
    Actual fuel economy varies with driving conditions, maintenance, vehicle
    age, and individual driving habits.

    Running cost estimates use approximate metro fuel prices:
    Petrol ₹96/L, Diesel ₹88/L, LPG ₹58/kg.
    """)

    st.markdown("### Valuation tiers")
    st.markdown("""
    The trade-in and dealer retail figures shown alongside the main estimate
    are derived by applying fixed multipliers to the predicted price:

    | Tier | Multiplier range |
    |------|-----------------|
    | Trade-in estimate | 88–94% of predicted price |
    | Private sale (fair market) | 100% (the prediction itself) |
    | Dealer retail estimate | 106–115% of predicted price |

    These are rough approximations of typical pricing tiers, not independent
    market valuations or verified quotes.
    """)

    st.markdown("### Limitations")
    st.markdown("""
    - The model does not account for vehicle condition, accident history,
      service records, location-specific demand, or current market trends.
    - The training data is a historical snapshot from one platform (Quikr)
      and may not represent the broader Indian used car market.
    - Linear regression assumes a linear relationship between features and
      price, which may not hold for all vehicle segments.
    - Predictions for manufacturers or models with very few training examples
      are likely less reliable.
    - The fuel economy estimates are category-level approximations, not
      vehicle-specific measurements.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="av-footer">
    <strong>AutoValuate</strong> · Used car valuation and market analysis<br>
    Predictions are statistical estimates based on the Quikr dataset.
    Not a certified appraisal, inspection report, or guaranteed price.<br>
    Model: Linear Regression · Dataset: ~816 cleaned Quikr listings
</div>
""", unsafe_allow_html=True)
