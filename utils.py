import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Design tokens – chart colours matched to the app's design system
# ---------------------------------------------------------------------------
PALETTE = {
    "red": "#B83A32",
    "red_light": "#D4645C",
    "charcoal": "#20211F",
    "grey": "#777873",
    "border": "#E3E1DA",
    "surface": "#FFFFFF",
    "petrol": "#B83A32",
    "diesel": "#5C7A6B",
    "lpg": "#C4903D",
}


def format_inr(number):
    """Format a number into Indian Rupee representation (e.g. ₹4,50,000)."""
    if number is None or (isinstance(number, float) and np.isnan(number)):
        return "₹0"
    number = int(round(number))
    if number < 0:
        return f"-₹{abs(number):,}"

    s = str(number)
    if len(s) <= 3:
        return f"₹{s}"

    last_three = s[-3:]
    other_digits = s[:-3]
    parts = []
    while len(other_digits) > 2:
        parts.insert(0, other_digits[-2:])
        other_digits = other_digits[:-2]
    if other_digits:
        parts.insert(0, other_digits)
    parts.append(last_three)
    return f"₹{','.join(parts)}"


def calculate_emi(principal, annual_rate, tenure_years):
    """Calculate monthly EMI for a car loan.

    Returns (monthly_emi, total_interest) rounded to the nearest rupee.
    """
    if principal <= 0:
        return 0, 0
    monthly_rate = annual_rate / (12 * 100)
    months = tenure_years * 12
    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = (principal * monthly_rate * ((1 + monthly_rate) ** months)) / (
            ((1 + monthly_rate) ** months) - 1
        )
    total_payment = emi * months
    total_interest = total_payment - principal
    return round(emi), round(total_interest)


def estimate_fuel_economy_kmpl(car_name, company, fuel_type):
    """Estimate fuel economy (km/L) based on vehicle category.

    These are rough estimates derived from publicly available specification
    ranges for common Indian car segments. They are NOT certified ARAI
    figures for any specific vehicle.
    """
    name_lower = str(car_name).lower()
    comp_lower = str(company).lower()
    fuel = str(fuel_type).capitalize()

    # Hatchbacks (small / city cars)
    if any(k in name_lower for k in ["alto", "kwid", "nano", "eon", "spark"]):
        base = 22.0 if fuel == "Petrol" else (26.0 if fuel == "Diesel" else 18.0)
    elif any(
        k in name_lower
        for k in [
            "swift", "wagon", "celerio", "i10", "santro", "beat",
            "ritz", "indica", "tiago", "polo", "figo",
        ]
    ):
        base = 20.5 if fuel == "Petrol" else (24.5 if fuel == "Diesel" else 17.5)
    elif any(k in name_lower for k in ["baleno", "i20", "glanza", "jazz"]):
        base = 19.5 if fuel == "Petrol" else (23.8 if fuel == "Diesel" else 17.0)
    # Compact sedans & executive sedans
    elif any(
        k in name_lower
        for k in ["dzire", "amaze", "etios", "indigo", "zest", "aspire"]
    ):
        base = 19.0 if fuel == "Petrol" else (24.0 if fuel == "Diesel" else 16.5)
    elif any(
        k in name_lower
        for k in [
            "city", "verna", "ciaz", "vento", "rapid", "corolla",
            "elantra", "civic", "cruze", "optra",
        ]
    ):
        base = 16.5 if fuel == "Petrol" else (21.5 if fuel == "Diesel" else 14.0)
    # Compact SUVs / MUVs
    elif any(
        k in name_lower
        for k in [
            "brezza", "venue", "ecosport", "duster", "creta",
            "ertiga", "triber", "nexon",
        ]
    ):
        base = 16.0 if fuel == "Petrol" else (20.5 if fuel == "Diesel" else 15.0)
    # Heavy SUVs & MPVs
    elif any(
        k in name_lower
        for k in [
            "innova", "scorpio", "safari", "xuv", "bolero", "qualis",
            "tavera", "fortuner", "endeavour", "pajero",
        ]
    ):
        base = 11.5 if fuel == "Petrol" else (14.2 if fuel == "Diesel" else 10.5)
    # Luxury vehicles
    elif any(
        k in comp_lower
        for k in ["audi", "bmw", "mercedes", "jaguar", "volvo", "land rover"]
    ):
        base = 10.5 if fuel == "Petrol" else (14.0 if fuel == "Diesel" else 9.0)
    else:
        # Default fallback
        base = 16.0 if fuel == "Petrol" else (19.0 if fuel == "Diesel" else 13.5)

    return round(base, 1)


def load_data_and_model():
    """Load cleaned dataset and trained sklearn pipeline.

    Raises FileNotFoundError if the CSV or model pickle is missing.
    """
    try:
        df = pd.read_csv("Cleaned_Car_data.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find 'Cleaned_Car_data.csv'. "
            "Make sure the dataset file is in the working directory."
        )

    try:
        with open("LinearRegressionModel.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find 'LinearRegressionModel.pkl'. "
            "Make sure the trained model file is in the working directory."
        )

    # Enrich with estimated fuel efficiency if not already present
    if "fuel_efficiency_kmpl" not in df.columns:
        df["fuel_efficiency_kmpl"] = df.apply(
            lambda r: estimate_fuel_economy_kmpl(r["name"], r["company"], r["fuel_type"]),
            axis=1,
        )
    return df, model


def predict_price(model, company, name, year, kms_driven, fuel_type):
    """Predict vehicle price using the trained scikit-learn pipeline.

    Returns the predicted price floored at ₹25,000 (minimum scrap value).
    """
    input_df = pd.DataFrame(
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
        data=[[name, company, int(year), int(kms_driven), fuel_type]],
    )
    predicted_val = model.predict(input_df)[0]
    # Floor at a realistic minimum
    predicted_val = max(25000, predicted_val)
    return predicted_val


def get_valuation_range(price):
    """Compute indicative valuation tiers from the predicted price.

    The multipliers (88–94% for trade-in, 106–115% for dealer retail)
    are rough industry approximations, not verified market quotes.
    """
    fair_market = int(round(price))
    return {
        "fair_market": fair_market,
        "trade_in_low": int(round(fair_market * 0.88)),
        "trade_in_high": int(round(fair_market * 0.94)),
        "retail_low": int(round(fair_market * 1.06)),
        "retail_high": int(round(fair_market * 1.15)),
        "fair_range_low": int(round(fair_market * 0.94)),
        "fair_range_high": int(round(fair_market * 1.06)),
    }


def estimate_running_cost(fuel_type, kmpl):
    """Estimate fuel cost per km and per 1,000 km.

    Uses approximate metro fuel prices: Petrol ₹96/L, Diesel ₹88/L, LPG ₹58/kg.
    """
    if kmpl <= 0:
        return 0, 0
    fuel = str(fuel_type).capitalize()
    if fuel == "Diesel":
        fuel_price = 88.0
    elif fuel == "LPG":
        fuel_price = 58.0
    else:
        fuel_price = 96.0  # Petrol default

    cost_per_km = fuel_price / kmpl
    cost_per_1000km = cost_per_km * 1000
    return round(cost_per_km, 2), round(cost_per_1000km)


# ---------------------------------------------------------------------------
# Chart helpers — restrained automotive palette, theme-aware
# ---------------------------------------------------------------------------

_FUEL_COLORS = {"Petrol": "#B83A32", "Diesel": "#5C7A6B", "LPG": "#C4903D"}


def _chart_font(is_dark=False):
    color = "#E8E7E3" if is_dark else "#20211F"
    return dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12, color=color)


def _chart_layout_defaults(fig, is_dark=False):
    """Apply consistent layout defaults across all charts."""
    grid_color = "rgba(255,255,255,0.07)" if is_dark else "rgba(0,0,0,0.06)"
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=44, b=16),
        font=_chart_font(is_dark),
        hoverlabel=dict(
            bgcolor="#242422" if is_dark else "#FFFFFF",
            font_color="#E8E7E3" if is_dark else "#20211F",
            bordercolor="#3A3A37" if is_dark else "#E3E1DA",
        ),
    )
    fig.update_xaxes(
        gridcolor=grid_color, zeroline=False,
        tickfont=dict(size=11, color="#9B9A96" if is_dark else "#777873"),
    )
    fig.update_yaxes(
        gridcolor=grid_color, zeroline=False,
        tickfont=dict(size=11, color="#9B9A96" if is_dark else "#777873"),
    )
    return fig


def create_brand_price_chart(df, template="plotly_white", is_dark=False):
    """Horizontal bar chart showing average listing price by manufacturer."""
    avg_price = df.groupby("company")["Price"].mean().reset_index()
    avg_price = avg_price.sort_values(by="Price", ascending=True)

    fig = go.Figure(
        go.Bar(
            y=avg_price["company"],
            x=avg_price["Price"],
            orientation="h",
            marker=dict(color=PALETTE["red"], cornerradius=3),
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Average listing price by manufacturer", font=dict(size=14)),
        xaxis_title="Average price (₹)",
        yaxis_title=None,
        template=template,
    )
    _chart_layout_defaults(fig, is_dark)
    return fig


def create_depreciation_chart(df, template="plotly_white", is_dark=False):
    """Line chart showing how listing prices vary by model year."""
    yearly = df.groupby("year")["Price"].agg(["mean", "median"]).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=yearly["year"], y=yearly["mean"],
            mode="lines+markers", name="Mean price",
            line=dict(color=PALETTE["red"], width=2.5),
            marker=dict(size=6, color=PALETTE["red"]),
            hovertemplate="Year %{x}: ₹%{y:,.0f}<extra>Mean</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yearly["year"], y=yearly["median"],
            mode="lines+markers", name="Median price",
            line=dict(color=PALETTE["diesel"], width=2, dash="dash"),
            marker=dict(size=5, color=PALETTE["diesel"]),
            hovertemplate="Year %{x}: ₹%{y:,.0f}<extra>Median</extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Price trend by model year", font=dict(size=14)),
        xaxis_title="Model year",
        yaxis_title="Price (₹)",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    _chart_layout_defaults(fig, is_dark)
    return fig


def create_kms_vs_price_chart(df, template="plotly_white", is_dark=False):
    """Scatter plot of odometer reading vs listing price, coloured by fuel type."""
    fig = px.scatter(
        df,
        x="kms_driven", y="Price",
        color="fuel_type",
        color_discrete_map=_FUEL_COLORS,
        labels={
            "kms_driven": "Kilometres driven",
            "Price": "Listing price (₹)",
            "fuel_type": "Fuel type",
        },
        hover_data=["name", "year"],
        template=template,
    )
    fig.update_layout(
        title=dict(text="Odometer reading vs listing price", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    _chart_layout_defaults(fig, is_dark)
    return fig


def create_fuel_distribution_chart(df, template="plotly_white", is_dark=False):
    """Donut chart showing the share of listings by fuel type."""
    fuel_counts = df["fuel_type"].value_counts().reset_index()
    fuel_counts.columns = ["fuel_type", "count"]

    fig = px.pie(
        fuel_counts,
        values="count", names="fuel_type",
        hole=0.55,
        color="fuel_type",
        color_discrete_map=_FUEL_COLORS,
        template=template,
    )
    fig.update_layout(
        title=dict(text="Listings by fuel type", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    fig.update_traces(
        textinfo="percent+label",
        textfont_size=11,
        hovertemplate="%{label}: %{value} listings (%{percent})<extra></extra>",
    )
    _chart_layout_defaults(fig, is_dark)
    return fig
