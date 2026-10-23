import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

def format_inr(number):
    """Format a number into Indian Rupee representation (e.g. ₹ 4,50,000)"""
    if number is None or np.isnan(number):
        return "₹ 0"
    number = int(round(number))
    if number < 0:
        return f"-₹ {abs(number):,}"
    
    s = str(number)
    if len(s) <= 3:
        return f"₹ {s}"
    
    last_three = s[-3:]
    other_digits = s[:-3]
    parts = []
    while len(other_digits) > 2:
        parts.insert(0, other_digits[-2:])
        other_digits = other_digits[:-2]
    if other_digits:
        parts.insert(0, other_digits)
    parts.append(last_three)
    return f"₹ {','.join(parts)}"

def calculate_emi(principal, annual_rate, tenure_years):
    """Calculate monthly EMI for car loan"""
    if principal <= 0:
        return 0, 0
    monthly_rate = annual_rate / (12 * 100)
    months = tenure_years * 12
    if monthly_rate == 0:
        emi = principal / months
    else:
        emi = (principal * monthly_rate * ((1 + monthly_rate) ** months)) / (((1 + monthly_rate) ** months) - 1)
    total_payment = emi * months
    total_interest = total_payment - principal
    return round(emi), round(total_interest)

def estimate_fuel_economy_kmpl(car_name, company, fuel_type):
    """
    Standard ARAI Fuel Efficiency (Mileage in km/L or km/kg)
    derived from Indian Automotive specifications (CarDekho / Autocar / ARAI benchmarks).
    """
    name_lower = str(car_name).lower()
    comp_lower = str(company).lower()
    fuel = str(fuel_type).capitalize()

    # Hatchbacks (Ultra high mileage)
    if any(k in name_lower for k in ["alto", "kwid", "nano", "eon", "spark"]):
        base = 22.0 if fuel == "Petrol" else (26.0 if fuel == "Diesel" else 18.0)
    elif any(k in name_lower for k in ["swift", "wagon", "celerio", "i10", "santro", "beat", "ritz", "indica", "tiago", "polo", "figo"]):
        base = 20.5 if fuel == "Petrol" else (24.5 if fuel == "Diesel" else 17.5)
    elif any(k in name_lower for k in ["baleno", "i20", "glanza", "jazz", "al भाषा"]):
        base = 19.5 if fuel == "Petrol" else (23.8 if fuel == "Diesel" else 17.0)
    # Compact Sedans & Executive Sedans
    elif any(k in name_lower for k in ["dzire", "amaze", "etios", "indigo", "zest", "aspire"]):
        base = 19.0 if fuel == "Petrol" else (24.0 if fuel == "Diesel" else 16.5)
    elif any(k in name_lower for k in ["city", "verna", "ciaz", "vento", "rapid", "corolla", "elantra", "civic", "cruze", "optra"]):
        base = 16.5 if fuel == "Petrol" else (21.5 if fuel == "Diesel" else 14.0)
    # Compact SUVs / MUVs
    elif any(k in name_lower for k in ["brezza", "venue", "ecosport", "duster", "creta", "ertiga", "triber", "nexon"]):
        base = 16.0 if fuel == "Petrol" else (20.5 if fuel == "Diesel" else 15.0)
    # Heavy SUVs & MPVs
    elif any(k in name_lower for k in ["innova", "scorpio", "safari", "xuv", "bolero", "qualis", "tavera", "fortuner", "endeavour", "pajero"]):
        base = 11.5 if fuel == "Petrol" else (14.2 if fuel == "Diesel" else 10.5)
    # Luxury vehicles
    elif any(k in comp_lower for k in ["audi", "bmw", "mercedes", "jaguar", "volvo", "land rover"]):
        base = 10.5 if fuel == "Petrol" else (14.0 if fuel == "Diesel" else 9.0)
    else:
        # Default fallback
        base = 16.0 if fuel == "Petrol" else (19.0 if fuel == "Diesel" else 13.5)

    return round(base, 1)

def load_data_and_model():
    """Load cleaned dataset and trained pipeline"""
    df = pd.read_csv("Cleaned_Car_data.csv")
    with open("LinearRegressionModel.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Enrich with ARAI Fuel Efficiency (km/L) if not already present
    if "fuel_efficiency_kmpl" not in df.columns:
        df["fuel_efficiency_kmpl"] = df.apply(
            lambda r: estimate_fuel_economy_kmpl(r["name"], r["company"], r["fuel_type"]),
            axis=1
        )
    return df, model

def predict_price(model, company, name, year, kms_driven, fuel_type):
    """Predict vehicle price using trained scikit-learn pipeline"""
    input_df = pd.DataFrame(
        columns=["name", "company", "year", "kms_driven", "fuel_type"],
        data=[[name, company, int(year), int(kms_driven), fuel_type]]
    )
    predicted_val = model.predict(input_df)[0]
    # Bound lower limit so price is realistic (min scrap value ₹25,000)
    predicted_val = max(25000, predicted_val)
    return predicted_val

def create_brand_price_chart(df, template="plotly_dark"):
    """Bar chart for average car price by brand"""
    avg_price = df.groupby("company")["Price"].mean().reset_index()
    avg_price = avg_price.sort_values(by="Price", ascending=False)
    
    fig = px.bar(
        avg_price,
        x="company",
        y="Price",
        title="Average Resale Price by Manufacturer",
        labels={"company": "Brand", "Price": "Average Price (₹)"},
        color="Price",
        color_continuous_scale="Purples",
        template=template
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_tickangle=-45,
        font=dict(family="sans-serif", size=12),
        coloraxis_showscale=False
    )
    return fig

def create_depreciation_chart(df, template="plotly_dark"):
    """Line/Scatter chart showing price depreciation over manufacturing years"""
    yearly_stats = df.groupby("year")["Price"].agg(["mean", "median"]).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_stats["year"],
        y=yearly_stats["mean"],
        mode="lines+markers",
        name="Average Price",
        line=dict(color="#6366f1", width=3),
        marker=dict(size=8, color="#818cf8")
    ))
    fig.add_trace(go.Scatter(
        x=yearly_stats["year"],
        y=yearly_stats["median"],
        mode="lines+markers",
        name="Median Price",
        line=dict(color="#10b981", width=2, dash="dash"),
        marker=dict(size=6, color="#34d399")
    ))
    
    fig.update_layout(
        title="Market Depreciation Curve Across Model Years",
        xaxis_title="Manufacturing Year",
        yaxis_title="Price (₹)",
        template=template,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_kms_vs_price_chart(df, template="plotly_dark"):
    """Scatter plot: Price vs Kilometers Driven on Odometer"""
    fig = px.scatter(
        df,
        x="kms_driven",
        y="Price",
        color="fuel_type",
        color_discrete_map={"Petrol": "#f59e0b", "Diesel": "#3b82f6", "LPG": "#10b981"},
        title="Odometer Distance (Total Run) vs Selling Price",
        labels={"kms_driven": "Total Kilometers Driven (Odometer)", "Price": "Price (₹)", "fuel_type": "Fuel Type"},
        hover_data=["name", "year", "fuel_efficiency_kmpl"],
        template=template
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="sans-serif", size=12)
    )
    return fig

def create_fuel_distribution_chart(df, template="plotly_dark"):
    """Donut chart for fuel type distribution"""
    fuel_counts = df["fuel_type"].value_counts().reset_index()
    fuel_counts.columns = ["fuel_type", "count"]
    
    fig = px.pie(
        fuel_counts,
        values="count",
        names="fuel_type",
        hole=0.55,
        title="Inventory Breakdown by Fuel Type",
        color="fuel_type",
        color_discrete_map={"Petrol": "#f59e0b", "Diesel": "#3b82f6", "LPG": "#10b981"},
        template=template
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="sans-serif", size=12)
    )
    return fig

