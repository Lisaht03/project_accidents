iimport streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import requests
import os
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
# This must be the very first Streamlit command.
# layout="wide": Uses the full screen width.
# initial_sidebar_state="collapsed": Hides the sidebar to make it look like a standalone app.
st.set_page_config(
    page_title="SafeRoute Île-de-France",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ADVANCED CSS STYLING ---
# We use st.markdown with unsafe_allow_html=True to inject custom CSS.
# This overrides the default Streamlit theme to create a professional "SaaS" look.
st.markdown("""
    <style>
        /* 1. Animated Background Gradient */
        /* Defines the animation movement */
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        /* Applies the gradient and animation to the main app container */
        .stApp {
            background: linear-gradient(-45deg, #1e3a8a, #3b82f6, #06b6d4, #10b981);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            font-family: 'Inter', sans-serif;
        }

        /* 2. Hide Sidebar completely */
        [data-testid="stSidebar"] { display: none; }

        /* 3. Glassmorphism Effect (The white translucent cards) */
        .glass-container {
            background: rgba(255, 255, 255, 0.90);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.6);
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            margin-bottom: 2rem;
        }

        /* 4. Custom Styling for Input Fields */
        .stSelectbox > div > div {
            background-color: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
            color: #334155 !important;
        }

        /* 5. Custom "Calculate" Button Styling */
        .stButton > button {
            background: #0f172a !important;
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
            margin-top: 1rem;
        }
        /* Hover effect (lifts button slightly) */
        .stButton > button:hover {
            transform: translateY(-2px);
            background: #334155 !important;
        }

        /* 6. Typography Classes */
        .custom-title {
            font-size: 3.5rem;
            font-weight: 900;
            letter-spacing: -2px;
            margin: 0;
            line-height: 1;
            background: -webkit-linear-gradient(45deg, #0f172a 30%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .custom-subtitle {
            font-size: 1.1rem;
            font-weight: 700;
            color: #475569;
            margin-bottom: 0px;
            text-transform: uppercase;
            letter-spacing: 2px;
            opacity: 0.9;
        }
        .ver-badge {
            background: #e2e8f0;
            color: #475569;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            white-space: nowrap;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. STATIC DATA ---
# Hardcoded coordinates allow the map to load instantly without calling an external Geocoding API.
dept_coords = {
    'Paris': (48.8566, 2.3522), 'Seine-et-Marne': (48.8411, 2.9994),
    'Yvelines': (48.8049, 1.9090), 'Essonne': (48.5228, 2.2285),
    'Hauts-de-Seine': (48.8306, 2.2215), 'Seine-Saint-Denis': (48.9112, 2.4699),
    'Val-de-Marne': (48.7775, 2.4571), "Val-d'Oise": (49.0560, 2.1467)
}

# Mapping user-friendly names to the format expected by the API
dept_display_map = {
    'Paris': '75 (Paris)', 'Seine-et-Marne': '77 (Seine-et-Marne)',
    'Yvelines': '78 (Yvelines)', 'Essonne': '91 (Essonne)',
    'Hauts-de-Seine': '92 (Hauts-de-Seine)', 'Seine-Saint-Denis': '93 (Seine-Saint-Denis)',
    'Val-de-Marne': '94 (Val-de-Marne)', "Val-d'Oise": "95 (Val-d'Oise)"
}

SEVERITY_TEXT_MAP = {
    "2": "Fatality (High Danger)", "3": "Hospitalized (Severe)", "4": "Slightly Injured (Minor)",
    2: "Fatality (High Danger)", 3: "Hospitalized (Severe)", 4: "Slightly Injured (Minor)",
    "Death": "Fatality (High Danger)", "Hospitalized": "Hospitalized (Severe)",
    "Slightly injured": "Slightly Injured (Minor)"
}

# Dropdown options
days_list, hours_list = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], list(range(24))
surface_list, road_cat_list = ['Normal','Wet / Slippery'], ['Major Roads','Secondary Roads','Local & Access Roads','Other / Off-Network']
speed_list = [10,20,30,40,50,60,70,80,90,100,110,130]

# --- 4. HELPER FUNCTIONS ---

def get_risk_style(prob_severity):
    """Returns color codes and messages based on the risk probability."""
    if prob_severity < 0.15:
        return {"color": "#10B981", "label": "LOW RISK", "msg": "Conditions likely safe."}
    elif prob_severity < 0.40:
        return {"color": "#F59E0B", "label": "MODERATE", "msg": "Increased vigilance required."}
    else:
        return {"color": "#EF4444", "label": "CRITICAL", "msg": "High probability of severe accident."}

@st.cache_data
def load_image_local():
    """
    Loads the logo and uses NumPy to remove the white background dynamically.
    This makes the logo transparent without needing Photoshop.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "image_0.png")
    try:
        img = Image.open(img_path).convert("RGBA")
        data = np.array(img)
        # Extract channels
        red, green, blue, alpha = data.T
        # Define white areas (RGB > 240)
        white_areas = (red > 240) & (green > 240) & (blue > 240)
        # Set Alpha (transparency) to 0 for white pixels
        data[..., 3][white_areas.T] = 0
        return Image.fromarray(data)
    except Exception:
        return None

# --- 5. UI LAYOUT ---
logo_img = load_image_local()

# Header Section (Glass Container)
st.markdown("<div class='glass-container' style='padding-bottom: 1rem;'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1.2, 4, 1])

# Column 1: Logo
with c1:
    if logo_img: st.image(logo_img, width=260)
    else: st.write("SafeRoute")

# Column 2: Titles (Custom HTML for styling)
with c2:
    st.markdown("<div style='margin-top: 50px; margin-left: -60px;'><p class='custom-subtitle'>AI Risk System</p><h1 class='custom-title'>Île-de-France</h1></div>", unsafe_allow_html=True)

# Column 3: Version Badge
with c3:
    st.markdown("<div style='text-align: right; margin-top: 60px;'><span class='ver-badge'>v3.1 Pro</span></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Inputs Section (Glass Container)
st.markdown("<div class='glass-container'><h4 style='margin-bottom: 20px; color: #334155;'>Incident Parameters</h4>", unsafe_allow_html=True)

# Grid Layout for inputs (3 columns)
c1, c2, c3 = st.columns(3)
with c1:
    s_dept = st.selectbox("Department", list(dept_coords.keys()), format_func=lambda x: dept_display_map[x])
    s_road = st.selectbox("Road Category", road_cat_list)
with c2:
    s_day = st.selectbox("Day of Week", days_list)
    s_speed = st.selectbox("Speed Limit (km/h)", speed_list)
with c3:
    s_hour = st.selectbox("Time of Day", hours_list, format_func=lambda x: f"{x:02d}:00")
    s_surf = st.selectbox("Surface Condition", surface_list)

st.write("")
btn = st.button("Calculate Severity Probability ⚡")
st.markdown("</div>", unsafe_allow_html=True)

# --- 6. API INTEGRATION & RESULTS ---
if btn:
    # 1. Prepare Payload
    params = {
        "department": dept_display_map[s_dept],
        "day_of_week": s_day,
        "hour": int(s_hour),
        "road_category": s_road,
        "speed_limit": int(s_speed),
        "surface_condition": s_surf
    }

    ph = st.empty() # Placeholder for the result

    # 2. Define API URL (Production Secret vs Localhost Fallback)
    api_url = st.secrets.get("API_URL", "http://127.0.0.1:8000/predict")

    with st.spinner('Analyzing...'):
        try:
            # 3. Call the Backend
            resp = requests.post(api_url, params=params, timeout=10)

            if resp.status_code == 200:
                res = resp.json()

                # 4. Parse Probabilities
                raw_probs = res.get("probabilities", {})
                # Clean strings (e.g., "12.5%" -> 0.125)
                clean_probs = {k: (float(str(v).strip('%'))/100) for k, v in raw_probs.items()}

                # 5. Calculate Aggregate Risk (Sum of Death + Hospitalized)
                risk = clean_probs.get("Death", 0.0) + clean_probs.get("Hospitalized", 0.0)
                style = get_risk_style(risk)
                txt = SEVERITY_TEXT_MAP.get(res.get("severity_text"), res.get("severity_text"))

                # 6. Display Results
                with ph.container():
                    # Result Card (Left side color bar indicating risk)
                    st.markdown(f"<div class='glass-container' style='border-left: 10px solid {style['color']}; padding: 1.5rem;'><h5 style='color:#64748b'>Severity Risk Level</h5><h1 style='color:{style['color']}'>{risk:.1%}</h1><h3 style='color:#334155'>Most Likely: {txt}</h3><p style='color:#64748b'>{style['msg']}</p></div>", unsafe_allow_html=True)

                    c_map, c_det = st.columns([2, 1])

                    # Detailed Bars (Right Column)
                    with c_det:
                        st.markdown("<h4 style='color: white; border-bottom: 1px solid rgba(255,255,255,0.3)'>Detailed Analysis</h4>", unsafe_allow_html=True)
                        for k, v in raw_probs.items():
                            st.markdown(f"<div style='color:white; display:flex; justify-content:space-between'><span>{k}</span><span>{v}</span></div>", unsafe_allow_html=True)
                            st.progress(clean_probs[k])

                    # Map Visualization (Left Column)
                    with c_map:
                        # Convert Hex color to RGB tuple for Pydeck
                        hc = style['color'].lstrip('#')
                        rgb = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4))

                        coords = dept_coords[s_dept]
                        df_m = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})

                        # Dynamic Layer: Color and Radius depend on Risk Score
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            df_m,
                            get_position='[lon, lat]',
                            get_fill_color=[*rgb, 160], # Dynamic Color
                            get_radius=2000+(risk*10000), # Dynamic Radius
                            pickable=True,
                            filled=True,
                            get_line_color=[255,255,255],
                            line_width_min_pixels=3
                        )

                        st.pydeck_chart(pdk.Deck(
                            layers=[layer],
                            initial_view_state=pdk.ViewState(latitude=coords[0], longitude=coords[1], zoom=10),
                            map_style=pdk.map_styles.LIGHT
                        ))
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# Footer
st.markdown("<div style='text-align: center; margin-top: 3rem; color: rgba(255,255,255,0.5); font-size: 12px;'>SafeRoute AI Engine • Île-de-France Sector • v3.1</div>", unsafe_allow_html=True)
