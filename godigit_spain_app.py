import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

# -------------------------------------------------------------------
# 1. Streamlit Page Configuration & Digit Theme Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Go Digit Spain | Motor Premium Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Digit Insurance Premium Dark Background Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Overall Page Dark Background */
    .stApp {
        background: linear-gradient(180deg, #0F0F11 0%, #18181B 50%, #0F0F11 100%) !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #F4F4F5 !important;
    }
    
    /* Ensure All Text & Labels Have High Visibility */
    label, p, span, h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #F4F4F5 !important;
    }
    
    /* Top Global Navigation Bar - Centered Go Digit Spain Title */
    .digit-nav {
        background-color: #18181B;
        padding: 1.2rem 2.2rem;
        border: 1px solid #27272A;
        border-bottom: 4px solid #FFC700;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .digit-logo-text {
        font-family: 'Fredoka', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        text-align: center;
    }
    .logo-go { color: #FFC700; }
    .logo-digit { color: #FFFFFF; }
    .logo-spain { color: #FFC700; margin-left: 10px; font-size: 2.6rem; }

    /* Hero Banner */
    .digit-hero {
        background: linear-gradient(135deg, #18181B 0%, #27272A 100%);
        border: 1px solid #3F3F46;
        border-radius: 20px;
        padding: 2.2rem 2.8rem;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border-bottom: 5px solid #FFC700;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }
    .digit-hero::after {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        background: #FFC700;
        opacity: 0.15;
        border-radius: 50%;
    }
    .digit-hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
    }
    .digit-hero-subtitle {
        color: #A1A1AA !important;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        font-weight: 500;
    }

    /* Digit Price Summary Card */
    .digit-price-card {
        background: linear-gradient(135deg, #1F1F23 0%, #18181B 100%);
        border: 2px solid #FFC700;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 12px 28px rgba(255, 199, 0, 0.15);
        margin-bottom: 1.5rem;
    }
    .digit-price-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #A1A1AA !important;
        font-weight: 700;
    }
    .digit-price-amount {
        font-size: 3.4rem;
        font-weight: 800;
        color: #FFC700 !important;
        margin: 0.3rem 0;
    }
    .digit-price-sub {
        font-size: 0.9rem;
        color: #4ADE80 !important;
        font-weight: 700;
    }

    /* Feature & Stat Cards */
    .digit-card {
        background: #18181B !important;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #27272A;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.2rem;
        transition: all 0.2s ease;
    }
    .digit-card:hover {
        border-color: #FFC700 !important;
    }

    /* Badge Pills */
    .badge-pill-yellow {
        background: #FFC700;
        color: #18181B !important;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 3px 10px;
        border-radius: 10px;
    }

    /* Form Inputs & Dropdown Visibility Fixes */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, input {
        background-color: #27272A !important;
        color: #FFFFFF !important;
        border-color: #3F3F46 !important;
        border-radius: 10px !important;
    }

    /* Tab Header Visibility */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #A1A1AA !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FFC700 !important;
        border-bottom: 3px solid #FFC700 !important;
    }

    /* DataFrame High Contrast */
    .stDataFrame, div[data-testid="stTable"] {
        background-color: #18181B !important;
        border-radius: 12px !important;
    }

    /* Custom Buttons */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"] {
        background-color: #FFC700 !important;
        color: #18181B !important;
        border: 2px solid #FFC700 !important;
        box-shadow: 0 4px 14px rgba(255, 199, 0, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #E6B300 !important;
        border-color: #E6B300 !important;
        color: #18181B !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. Model Loading & Configuration (Supports Streamlit Cloud & Local)
# -------------------------------------------------------------------
import urllib.request

MODEL_DIR = "glm_model_files"
LOCAL_FALLBACK_DIR = r"D:\2_Research\FILES\2 Learning\GLM\glm_model_files"
GITHUB_RELEASE_URL = "https://github.com/EPS-Learns/Motor_Insurance_Premium_Calculator/releases/download/glm_model_pkl_files/"

REQUIRED_MODEL_FILES = [
    "freq_liability_claims.pkl", "sev_liability_incurred.pkl",
    "freq_property_claims.pkl", "sev_property_incurred.pkl",
    "freq_theft_claims.pkl", "sev_theft_incurred.pkl",
    "freq_glass_claims.pkl", "sev_glass_incurred.pkl",
    "freq_legal_protection_claims.pkl", "sev_legal_protection_incurred.pkl",
    "freq_fire_claims.pkl", "sev_fire_incurred.pkl",
    "freq_occupants_claims.pkl", "sev_occupants_incurred.pkl",
    "freq_total_claims.pkl", "sev_total_incurred.pkl"
]

@st.cache_resource
def load_models():
    models = {}
    
    target_dir = MODEL_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    # Check local fallback folder if available on dev machine
    if os.path.isdir(LOCAL_FALLBACK_DIR):
        local_files = os.listdir(LOCAL_FALLBACK_DIR)
        if any(f.endswith(".pkl") for f in local_files):
            target_dir = LOCAL_FALLBACK_DIR
            
    for fname in REQUIRED_MODEL_FILES:
        key = fname[:-4]
        path = os.path.join(target_dir, fname)
        
        # Download from GitHub Releases if missing (e.g. Streamlit Cloud)
        if not os.path.isfile(path):
            download_url = GITHUB_RELEASE_URL + fname
            try:
                urllib.request.urlretrieve(download_url, path)
            except Exception:
                alt_path = os.path.join("glm_model_files", fname)
                if os.path.isfile(alt_path):
                    path = alt_path
                else:
                    continue
                    
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    models[key] = pickle.load(f)
            except Exception as e:
                st.warning(f"Failed to load {fname}: {e}")
                
    # Load any remaining .pkl files if available
    if os.path.isdir(target_dir):
        for fname in os.listdir(target_dir):
            if fname.endswith(".pkl"):
                key = fname[:-4]
                if key not in models:
                    path = os.path.join(target_dir, fname)
                    try:
                        with open(path, "rb") as f:
                            models[key] = pickle.load(f)
                    except Exception:
                        pass
                        
    return models

glm_models = load_models()

if not glm_models:
    st.error("No GLM models loaded. Please check model directory or GitHub release URLs.")
    st.stop()

cat_unique_values = {
    "fuel_type": ["D", "G"],
    "vehicle_brand_grp": ["AUDI", "CITROEN", "FORD", "MERCEDES", "OPEL", "PEUGEOT", "RENAULT", "SEAT", "TOYOTA", "VOLKSWAGEN", "OTHER"],
    "municipality_type": ["C", "I", "IS"],
    "circulation_area": ["R", "U"],
}

# Coverage Definition Map (5 Core Perils)
coverage_map = {
    "Liability": {
        "label": "Third Party Liability",
        "tagline": "Mandatory Legal Cover",
        "description": "Protects against legal liability for injury or property damage to third parties.",
        "freq_key": "freq_liability_claims",
        "sev_key": "sev_liability_incurred",
        "exposure_col": "liability_exposure",
        "family_sev": "gamma",
        "icon": "⚖️",
        "badge": "Mandatory Legal ⚖️"
    },
    "Property": {
        "label": "Own Damage (Property)",
        "tagline": "Accidental Collision Cover",
        "description": "Covers physical damage to your car due to accidents or collisions.",
        "freq_key": "freq_property_claims",
        "sev_key": "sev_property_incurred",
        "exposure_col": "total_exposure",
        "family_sev": "gamma",
        "icon": "🚗",
        "badge": "Super Saver ⚡"
    },
    "Theft": {
        "label": "Theft Protection",
        "tagline": "Total Loss & Parts Protection",
        "description": "Financial protection if your vehicle or its parts get stolen.",
        "freq_key": "freq_theft_claims",
        "sev_key": "sev_theft_incurred",
        "exposure_col": "total_exposure",
        "family_sev": "gamma",
        "icon": "🔑",
        "badge": "Popular 🔑"
    },
    "Glass": {
        "label": "Windscreen & Glass",
        "tagline": "Zero Deductible Glass Repair",
        "description": "Instant replacement for cracked windscreens, windows & sunroofs.",
        "freq_key": "freq_glass_claims",
        "sev_key": "sev_glass_incurred",
        "exposure_col": "total_exposure",
        "family_sev": "gamma",
        "icon": "🪟",
        "badge": "Quick Claim 🪟"
    },
    "LegalProtection": {
        "label": "Legal Defense Cover",
        "tagline": "Advocate Fees & Legal Expenses",
        "description": "Covers advocate fees & litigation expenses during road accidents.",
        "freq_key": "freq_legal_protection_claims",
        "sev_key": "sev_legal_protection_incurred",
        "exposure_col": "total_exposure",
        "family_sev": "gamma",
        "icon": "📜",
        "badge": "Smart Add-on 📜"
    }
}

available_coverages = {
    k: v for k, v in coverage_map.items()
    if v["freq_key"] in glm_models and v["sev_key"] in glm_models
}

# -------------------------------------------------------------------
# 3. Calculation Engine
# -------------------------------------------------------------------
def calculate_digit_quote(user_input_df: pd.DataFrame, selected_coverages: list, expense_load: float, profit_margin: float):
    row = user_input_df.iloc[0]
    results = {}
    total_pure_premium = 0.0
    total_expected_claims = 0.0

    for cov in selected_coverages:
        cfg = available_coverages[cov]
        freq_model = glm_models[cfg["freq_key"]]
        sev_model = glm_models[cfg["sev_key"]]

        exp_val = float(row.get(cfg["exposure_col"], 1.0))
        eval_df = user_input_df.copy()
        eval_df["offset"] = np.log(exp_val)

        lambda_hat = float(freq_model.predict(eval_df)[0])
        sev_hat = float(sev_model.predict(eval_df)[0])

        pure_premium = lambda_hat * sev_hat
        comm_premium = pure_premium / max(0.01, (1.0 - expense_load - profit_margin))

        results[cov] = {
            "label": cfg["label"],
            "icon": cfg["icon"],
            "badge": cfg["badge"],
            "lambda_hat": lambda_hat,
            "sev_hat": sev_hat,
            "pure_premium": pure_premium,
            "commercial_premium": comm_premium,
        }

        total_pure_premium += pure_premium
        total_expected_claims += lambda_hat

    total_commercial_premium = total_pure_premium / max(0.01, (1.0 - expense_load - profit_margin))

    return {
        "coverage_details": results,
        "total_pure_premium": total_pure_premium,
        "total_commercial_premium": total_commercial_premium,
        "total_expected_claims": total_expected_claims,
    }

# -------------------------------------------------------------------
# 4. Top Website Navigation Bar (Centered Go Digit Spain Title Only)
# -------------------------------------------------------------------
st.markdown("""
<div class="digit-nav">
    <div class="digit-logo-text">
        <span class="logo-go">go</span><span class="logo-digit">digit</span><span class="logo-spain">spain</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 5. Digit Header Hero Banner
# -------------------------------------------------------------------
st.markdown("""
<div class="digit-hero">
    <div class="digit-hero-title">
        <span>⚡ Motor Insurance Premium Calculator</span>
    </div>
    <div class="digit-hero-subtitle">
        Simple, Transparent & Powered by Advanced Actuarial Generalized Linear Models (GLMs)
    </div>
</div>
""", unsafe_allow_html=True)

# Default values for main form
def_driver_age, def_veh_age, def_licence = 32, 5, 8
def_fuel, def_brand, def_muni, def_area = "D", "RENAULT", "C", "R"
def_val, def_seats, def_pwr = 18000.0, 5, 11.0
def_exp, def_liab_exp = 1.0, 1.0

# -------------------------------------------------------------------
# 6. Step-by-Step Interactive Form Layout
# -------------------------------------------------------------------
tab_step1, tab_step2, tab_step3, tab_step4, tab_step5 = st.tabs([
    "1️⃣ Car & Driver Details",
    "2️⃣ Driving Location & Use",
    "3️⃣ Coverage Selection",
    "4️⃣ Commercial Tariff Settings",
    "📖 Digit Data Dictionary"
])

with tab_step1:
    st.markdown("#### 🚘 Step 1: Tell Us About Your Vehicle & Driver")
    c1, c2, c3 = st.columns(3)
    with c1:
        vehicle_brand_grp = st.selectbox("Vehicle Brand", cat_unique_values["vehicle_brand_grp"], index=cat_unique_values["vehicle_brand_grp"].index(def_brand))
        vehicle_value = st.number_input("Insured Declared Value (IDV) €", min_value=500.0, max_value=250000.0, value=def_val, step=1000.0, help="Current estimated market value of your vehicle")
    with c2:
        driver_age = st.number_input("Driver Age (Years)", min_value=18, max_value=95, value=def_driver_age)
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=40, value=def_veh_age)
    with c3:
        age_licence = st.number_input("Driving Licence Tenure (Years)", min_value=0, max_value=75, value=def_licence)
        fuel_type = st.selectbox("Fuel Type", cat_unique_values["fuel_type"], index=cat_unique_values["fuel_type"].index(def_fuel), format_func=lambda x: "Diesel" if x=="D" else "Petrol / Gasoline")
    
    col_a, col_b = st.columns(2)
    with col_a:
        seats = st.number_input("Seating Capacity", min_value=1, max_value=9, value=def_seats)
    with col_b:
        pwr_weight = st.number_input("Power-to-Weight Ratio (kg/hp)", min_value=1.0, max_value=100.0, value=def_pwr, step=0.5)

with tab_step2:
    st.markdown("#### 📍 Step 2: Where & How Is the Vehicle Driven?")
    g1, g2 = st.columns(2)
    with g1:
        municipality = st.selectbox(
            "Municipality Classification",
            cat_unique_values["municipality_type"],
            index=cat_unique_values["municipality_type"].index(def_muni),
            format_func=lambda x: {"I": "Inland (I)", "C": "Coastal (C)", "IS": "Islands (IS)"}.get(x, x),
            help="Geographic region of your home municipality"
        )
        area = st.selectbox(
            "Primary Circulation Area",
            cat_unique_values["circulation_area"],
            index=cat_unique_values["circulation_area"].index(def_area),
            format_func=lambda x: "Urban Traffic (U)" if x == "U" else "Rural Highway (R)"
        )
    with g2:
        total_exposure = st.number_input("Total Policy Exposure Term", min_value=0.01, max_value=1.0, value=def_exp, step=0.1, help="1.0 = Full 1 Year Cover")
        liab_exposure = st.number_input("Liability Cover Exposure Term", min_value=0.01, max_value=1.0, value=def_liab_exp, step=0.1)

# Session state initialization for coverage selection
if "selected_covs" not in st.session_state:
    st.session_state.selected_covs = ["Liability", "Property", "Glass"]

with tab_step3:
    all_cov_keys = list(available_coverages.keys())
    curr_covs = st.session_state.selected_covs
    is_comp_active = set(curr_covs) == set(all_cov_keys)
    is_liab_active = curr_covs == ["Liability"]

    hdr_col1, hdr_col2 = st.columns([5, 1])
    with hdr_col1:
        st.markdown("#### 🛡️ Step 3: Choose Your Policy Coverage")
    with hdr_col2:
        if st.button("🧹 Clear", key="master_clear", help="Clear all coverage selections"):
            st.session_state.selected_covs = []
            st.rerun()

    st.caption("Select or deselect coverages below to instantly update your Digit insurance quote:")

    # Quick Preset Action Toggles
    m_col1, m_col2, _ = st.columns([3.5, 3.5, 3])
    with m_col1:
        comp_btn_type = "primary" if is_comp_active else "secondary"
        comp_btn_label = "✔ 🌟 Comprehensive (Selected)" if is_comp_active else "🌟 Comprehensive (Select All)"
        if st.button(comp_btn_label, type=comp_btn_type, key="master_comp", width="stretch"):
            st.session_state.selected_covs = list(all_cov_keys)
            st.rerun()
    with m_col2:
        liab_btn_type = "primary" if is_liab_active else "secondary"
        liab_btn_label = "✔ ⚖️ Liability Only (Selected)" if is_liab_active else "⚖️ Liability Only"
        if st.button(liab_btn_label, type=liab_btn_type, key="master_liab", width="stretch"):
            st.session_state.selected_covs = ["Liability"]
            st.rerun()

    st.markdown("---")

    # Interactive Grid Cards with Primary/Secondary Toggle Buttons
    grid_cols = st.columns(2)
    cov_keys = list(available_coverages.keys())

    for idx, key in enumerate(cov_keys):
        c_info = available_coverages[key]
        is_active = key in st.session_state.selected_covs

        with grid_cols[idx % 2]:
            if is_active:
                border_style = "border: 2.5px solid #FFC700; background: linear-gradient(135deg, #FFFDF0 0%, #FFF9C4 100%); box-shadow: 0 8px 20px rgba(255, 199, 0, 0.25);"
                status_badge = '<span style="background:#FFC700; color:#18181B; font-weight:800; font-size:0.8rem; padding:4px 12px; border-radius:12px;">✔ SELECTED</span>'
                btn_label = "✔ Included in Quote"
                btn_type = "primary"
            else:
                border_style = "border: 1.5px solid #27272A; background: #18181B; opacity: 0.85;"
                status_badge = '<span style="background:#27272A; color:#A1A1AA; font-weight:600; font-size:0.8rem; padding:4px 12px; border-radius:12px;">+ NOT SELECTED</span>'
                btn_label = "+ Add to Quote"
                btn_type = "secondary"

            st.markdown(f"""
            <div style="{border_style} padding:18px 22px; border-radius:18px; margin-bottom:10px; transition: all 0.2s ease;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="badge-pill-yellow">{c_info['badge']}</span>
                    {status_badge}
                </div>
                <div style="font-weight:800; font-size:1.15rem; color:{'#18181B' if is_active else '#FFFFFF'}; margin-top:10px;">
                    {c_info['icon']} {c_info['label']}
                </div>
                <div style="font-size:0.88rem; color:{'#3F3F46' if is_active else '#A1A1AA'}; font-weight:600; margin-top:2px;">
                    {c_info['tagline']}
                </div>
                <div style="font-size:0.82rem; color:{'#71717A' if is_active else '#71717A'}; margin-top:4px;">
                    {c_info['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(btn_label, key=f"btn_cov_{key}", type=btn_type, width="stretch"):
                if is_active:
                    st.session_state.selected_covs.remove(key)
                else:
                    st.session_state.selected_covs.append(key)
                st.rerun()

selected_coverages = st.session_state.selected_covs

with tab_step4:
    st.markdown("#### 🛠️ Commercial Tariff & Underwriting Loadings")
    st.caption("Adjust corporate expense allocations and target underwriting margins directly:")
    tc1, tc2 = st.columns(2)
    with tc1:
        expense_pct = st.slider("Operations & Tech Expenses (%)", min_value=0.0, max_value=30.0, value=10.0, step=1.0) / 100.0
    with tc2:
        profit_pct = st.slider("Target Underwriting Profit (%)", min_value=0.0, max_value=20.0, value=5.0, step=1.0) / 100.0

with tab_step5:
    st.markdown("#### 📖 Go Digit Motor Dataset & Variable Reference")
    dict_data = [
        {"Variable": "insured_id", "Category": "Policy ID", "Description": "Unique policy identifier assigned sequentially."},
        {"Variable": "year", "Category": "Calendar Year", "Description": "Calendar year of policy risk term."},
        {"Variable": "policy_type", "Category": "Coverage Type", "Description": "TP (Third Party), TPG (Third Party+Glass), CC (Combined), COMP_E (Comprehensive excess), COMP_N (Comprehensive no excess)."},
        {"Variable": "policy_status", "Category": "Status", "Description": "Active (A) or Cancelled (C)."},
        {"Variable": "business_type", "Category": "Business Origin", "Description": "NB (New Business) or P (Portfolio Renewal)."},
        {"Variable": "payment_frequency", "Category": "Billing", "Description": "A (Annual), S (Semiannual), Q (Quarterly)."},
        {"Variable": "bonus_score", "Category": "Claims Score", "Description": "G (Favourable claims history), N (Neutral), B (Poor history)."},
        {"Variable": "driver_age", "Category": "Rating Factor", "Description": "Age of the main policy driver (years)."},
        {"Variable": "vehicle_age", "Category": "Rating Factor", "Description": "Age of insured car (years)."},
        {"Variable": "age_driving_licence", "Category": "Rating Factor", "Description": "Licence tenure / years since issuance."},
        {"Variable": "fuel_type", "Category": "Rating Factor", "Description": "D (Diesel) or G (Gasoline/Petrol)."},
        {"Variable": "vehicle_value", "Category": "Rating Factor", "Description": "Insured Declared Value (IDV) in €."},
        {"Variable": "seats", "Category": "Rating Factor", "Description": "Number of passenger seats."},
        {"Variable": "power_to_weight_ratio", "Category": "Rating Factor", "Description": "Power-to-weight ratio (kg per horsepower)."},
        {"Variable": "vehicle_brand", "Category": "Rating Factor", "Description": "Vehicle brand/manufacturer group."},
        {"Variable": "municipality_type", "Category": "Geography", "Description": "I (Inland), C (Coastal), IS (Islands)."},
        {"Variable": "circulation_area", "Category": "Geography", "Description": "U (Urban), R (Rural)."},
        {"Variable": "total_exposure", "Category": "Exposure", "Description": "Effective policy duration in years (0 to 1+)."},
        {"Variable": "*_claims", "Category": "Frequency", "Description": "Reported claim count per risk peril."},
        {"Variable": "*_incurred", "Category": "Severity", "Description": "Total incurred loss cost (paid + reserved) per risk peril."}
    ]
    st.dataframe(pd.DataFrame(dict_data), width="stretch", hide_index=True)

# -------------------------------------------------------------------
# 7. Live Digit Quote Display & Summary Card
# -------------------------------------------------------------------
st.markdown("---")

if not selected_coverages:
    st.warning("⚠️ Please select at least one coverage in Step 3 to view your Digit instant quote.")
    st.stop()

# Prepare Data Frame for Model Input
user_data = {
    "driver_age": driver_age,
    "vehicle_age": vehicle_age,
    "age_driving_licence": age_licence,
    "fuel_type": fuel_type,
    "vehicle_value": vehicle_value,
    "seats": seats,
    "power_to_weight_ratio": pwr_weight,
    "vehicle_brand_grp": vehicle_brand_grp,
    "municipality_type": municipality,
    "circulation_area": area,
    "total_exposure": total_exposure,
    "liability_exposure": liab_exposure,
}
user_input_df = pd.DataFrame([user_data])

try:
    quote = calculate_digit_quote(
        user_input_df=user_input_df,
        selected_coverages=selected_coverages,
        expense_load=expense_pct,
        profit_margin=profit_pct
    )
except Exception as e:
    st.error(f"❌ Underwriting Calculation Error: {e}")
    st.stop()

# -------------------------------------------------------------------
# 8. Prominent Digit Yellow Price Banner & Summary
# -------------------------------------------------------------------
st.markdown("### ⚡ Your Instant Digit Insurance Quote")

col_price, col_stats = st.columns([2, 3])

with col_price:
    st.markdown(f"""
    <div class="digit-price-card">
        <div class="digit-price-label">Total Annual Commercial Premium</div>
        <div class="digit-price-amount">€{quote['total_commercial_premium']:,.2f}</div>
        <div class="digit-price-sub">✔ Instant Policy Issue • Zero Paperwork</div>
    </div>
    """, unsafe_allow_html=True)

with col_stats:
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""
        <div class="digit-card">
            <div style="font-size:0.8rem; color:#A1A1AA; font-weight:700; text-transform:uppercase;">Net Actuarial Risk Cost</div>
            <div style="font-size:1.6rem; font-weight:800; color:#FFFFFF;">€{quote['total_pure_premium']:,.2f}</div>
            <div style="font-size:0.8rem; color:#4ADE80; font-weight:600; margin-top:4px;">Expected Claim Loss Cost</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        exp_claims = quote['total_expected_claims']
        st.markdown(f"""
        <div class="digit-card">
            <div style="font-size:0.8rem; color:#A1A1AA; font-weight:700; text-transform:uppercase;">Expected Annual Claims</div>
            <div style="font-size:1.6rem; font-weight:800; color:#38BDF8;">{exp_claims:.4f}</div>
            <div style="font-size:0.8rem; color:#38BDF8; font-weight:600; margin-top:4px;">claims per policy year</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background:#18181B; border-radius:12px; padding:12px 16px; border:1px solid #27272A; display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.88rem; font-weight:600; color:#A1A1AA;">Digit Commercial Loading: {expense_pct*100:.0f}% Expenses + {profit_pct*100:.0f}% Profit Margin</span>
        <span class="badge-pill-yellow">Verified Actuarial Tariff</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 9. Full Breakdown Table & Graph (Stacked Vertically)
# -------------------------------------------------------------------
st.markdown("#### 📑 Digit Coverage Price Breakup")
table_rows = []
for cov, det in quote["coverage_details"].items():
    table_rows.append({
        "Cover Peril": f"{det['icon']} {det['label']}",
        "Digit Tag": det["badge"],
        "Expected Frequency (λ)": f"{det['lambda_hat']:.5f}",
        "Avg Severity (€)": f"€{det['sev_hat']:,.2f}",
        "Net Risk Premium (€)": f"€{det['pure_premium']:,.2f}",
        "Final Price (€)": f"€{det['commercial_premium']:,.2f}"
    })
df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, width="stretch", hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### 📊 Premium Share by Cover")
chart_data = pd.DataFrame({
    "Cover": [det['label'] for det in quote["coverage_details"].values()],
    "Price (€)": [det['commercial_premium'] for det in quote["coverage_details"].values()]
}).set_index("Cover")
st.bar_chart(chart_data, color="#FFC700", width="stretch")

# -------------------------------------------------------------------
# Helper: FPDF Policy Quote PDF Generator
# -------------------------------------------------------------------
def clean_pdf_str(text_str):
    return str(text_str).replace("€", "EUR ").replace("λ", "lambda ").replace("•", "-").encode('latin-1', 'ignore').decode('latin-1').strip()

def generate_digit_pdf(user_data, quote_data, table_rows, expense_pct, profit_pct):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    DARK_COLOR = (24, 24, 28)
    YELLOW_COLOR = (255, 199, 0)
    SLATE_BG = (248, 250, 252)
    BORDER_GRAY = (220, 224, 230)

    # Top Header Banner
    pdf.set_fill_color(*DARK_COLOR)
    pdf.rect(0, 0, 210, 38, "F")
    
    pdf.set_text_color(*YELLOW_COLOR)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_xy(14, 8)
    pdf.cell(0, 8, "go digit spain", ln=True)
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(14, 20)
    pdf.cell(0, 6, "OFFICIAL MOTOR INSURANCE POLICY QUOTE", ln=True)
    
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(200, 200, 200)
    pdf.set_xy(120, 10)
    quote_ref = f"Quote Ref: GDS-2026-98421"
    pdf.cell(76, 5, quote_ref, align="R", ln=True)
    pdf.set_xy(120, 16)
    pdf.cell(76, 5, f"Issue Date: {pd.Timestamp.now().strftime('%d %b %Y')}", align="R", ln=True)
    pdf.set_xy(120, 22)
    pdf.cell(76, 5, f"Valid Until: {(pd.Timestamp.now() + pd.Timedelta(days=30)).strftime('%d %b %Y')}", align="R", ln=True)
    
    pdf.set_y(46)
    
    # 1. Customer & Risk Profile Matrix
    pdf.set_text_color(*DARK_COLOR)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. Policyholder & Vehicle Risk Specification Matrix", ln=True)
    
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_fill_color(*SLATE_BG)
    
    muni_desc = {"I": "Inland (I)", "C": "Coastal (C)", "IS": "Islands (IS)"}.get(user_data["municipality_type"], user_data["municipality_type"])
    area_desc = "Urban Traffic (U)" if user_data["circulation_area"] == "U" else "Rural Highway (R)"
    fuel_desc = "Diesel" if user_data["fuel_type"] == "D" else "Petrol / Gasoline"

    grid = [
        (f"Driver Age: {user_data['driver_age']} Years", f"Driving Licence Tenure: {user_data['age_driving_licence']} Years"),
        (f"Vehicle Brand: {user_data['vehicle_brand_grp']}", f"Insured Declared Value (IDV): EUR {user_data['vehicle_value']:,.2f}"),
        (f"Fuel Type: {fuel_desc}", f"Vehicle Age: {user_data['vehicle_age']} Years"),
        (f"Seating Capacity: {user_data['seats']} Seats", f"Power-to-Weight Ratio: {user_data['power_to_weight_ratio']:.1f} kg/hp"),
        (f"Municipality Region: {muni_desc}", f"Circulation Zone: {area_desc}"),
        (f"Total Risk Exposure: {user_data['total_exposure']} Term Year", f"Liability Cover Exposure: {user_data['liability_exposure']} Term Year"),
    ]
    
    for left_val, right_val in grid:
        pdf.cell(91, 6.5, f" {clean_pdf_str(left_val)}", border=1, fill=True)
        pdf.cell(91, 6.5, f" {clean_pdf_str(right_val)}", border=1, fill=True, ln=True)
        
    pdf.ln(5)
    
    # 2. Underwriting Commercial Loading Breakdown
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Commercial Premium & Underwriting Loadings Breakdown", ln=True)
    
    # Highlight Banner Box
    pdf.set_fill_color(255, 253, 240)
    pdf.set_draw_color(*YELLOW_COLOR)
    pdf.set_line_width(0.8)
    y_start = pdf.get_y()
    pdf.rect(14, y_start, 182, 24, "DF")
    
    pdf.set_y(y_start + 3)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(113, 113, 122)
    pdf.cell(0, 4, "TOTAL ANNUAL COMMERCIAL PREMIUM (INCL. ALL LOADINGS)", align="C", ln=True)
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*DARK_COLOR)
    pdf.cell(0, 9, f"EUR {quote_data['total_commercial_premium']:,.2f}", align="C", ln=True)
    
    pdf.set_y(y_start + 27)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(*BORDER_GRAY)
    
    # Commercial Loading Details Grid
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_fill_color(245, 247, 250)
    
    pure_cost = quote_data["total_pure_premium"]
    exp_cost = quote_data["total_commercial_premium"] * expense_pct
    profit_cost = quote_data["total_commercial_premium"] * profit_pct

    pdf.cell(60.6, 6, f" Net Risk Loss Cost: EUR {pure_cost:,.2f}", border=1, fill=True)
    pdf.cell(60.6, 6, f" Tech & Ops ({expense_pct*100:.0f}%): EUR {exp_cost:,.2f}", border=1, fill=True)
    pdf.cell(60.6, 6, f" Target Profit ({profit_pct*100:.0f}%): EUR {profit_cost:,.2f}", border=1, fill=True, ln=True)
    
    pdf.ln(5)
    
    # 3. Peril Breakdown Schedule Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Itemized Coverage Peril Tariff Schedule", ln=True)
    
    pdf.set_fill_color(*DARK_COLOR)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.cell(62, 6.5, " Cover Peril", border=1, fill=True)
    pdf.cell(30, 6.5, " Exp. Frequency", border=1, fill=True, align="R")
    pdf.cell(30, 6.5, " Avg Severity", border=1, fill=True, align="R")
    pdf.cell(30, 6.5, " Net Loss", border=1, fill=True, align="R")
    pdf.cell(30, 6.5, " Final Premium", border=1, fill=True, align="R", ln=True)
    
    pdf.set_text_color(*DARK_COLOR)
    pdf.set_font("Helvetica", "", 8.5)
    for r in table_rows:
        c_peril = clean_pdf_str(r.get("Cover Peril", ""))
        c_freq  = clean_pdf_str(r.get("Expected Frequency (λ)", ""))
        c_sev   = clean_pdf_str(r.get("Avg Severity (€)", ""))
        c_pure  = clean_pdf_str(r.get("Net Risk Premium (€)", ""))
        c_final = clean_pdf_str(r.get("Final Price (€)", ""))
        
        pdf.cell(62, 6, f" {c_peril}", border=1)
        pdf.cell(30, 6, f"{c_freq}", border=1, align="R")
        pdf.cell(30, 6, f"{c_sev}", border=1, align="R")
        pdf.cell(30, 6, f"{c_pure}", border=1, align="R")
        pdf.cell(30, 6, f"{c_final}", border=1, align="R", ln=True)
        
    pdf.ln(5)
    
    # 4. Coverage Benefit Scope Summary
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "4. Included Coverage Benefit Scope & Protection Terms", ln=True)
    
    cov_benefits = [
        ("Third Party Liability", "Full legal liability protection for third-party injury, death, and property damage."),
        ("Own Damage (Property)", "Covers physical damage to insured car from collision, overturning, or impact."),
        ("Theft Protection", "Financial compensation up to IDV for total loss due to vehicle theft or stolen parts."),
        ("Windscreen & Glass", "Zero-deductible instant replacement/repair for front windscreen, windows & rear glass."),
        ("Legal Defense Cover", "Covers advocate fees, court litigation expenses, and legal defense costs."),
    ]
    
    for cov_title, cov_desc in cov_benefits:
        if any(cov_title.lower() in clean_pdf_str(r.get("Cover Peril", "")).lower() for r in table_rows):
            pdf.set_x(14)
            pdf.set_font("Helvetica", "", 8)
            pdf.multi_cell(182, 4.5, clean_pdf_str(f"- {cov_title}: {cov_desc}"))
            
    pdf.ln(4)
    
    # 5. General Declarations Footer
    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "General Underwriting Provisions & Regulatory Disclosures:", ln=True)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(100, 100, 100)
    disclaimer_text = (
        "1. Tariffs are generated using Generalized Linear Models (GLMs) parameterized on motor portfolio claims experience.\n"
        "2. Final policy issuance is contingent upon vehicle physical inspection and claims history verification.\n"
        "3. Quote validity is 30 calendar days from issue date. All rates comply with standard insurance regulatory guidelines."
    )
    pdf.set_x(14)
    pdf.multi_cell(182, 3.8, clean_pdf_str(disclaimer_text))
    
    return bytes(pdf.output())

# -------------------------------------------------------------------
# 10. PDF Download Button
# -------------------------------------------------------------------
st.markdown("---")
col_pdf_center = st.columns([1, 2, 1])[1]

with col_pdf_center:
    pdf_bytes = generate_digit_pdf(user_data, quote, table_rows, expense_pct, profit_pct)
    st.download_button(
        label="📄 Download Official Quote (PDF)",
        data=pdf_bytes,
        file_name=f"digit_policy_quote_{driver_age}yo.pdf",
        mime="application/pdf",
        width="stretch",
        type="primary"
    )
