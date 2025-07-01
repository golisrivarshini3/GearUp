import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import numpy as np
import requests
import json
import torch
from transformers import pipeline
import urllib.parse
import re

# --- CONFIGURATION & INITIALIZATION ---

# Page configuration
st.set_page_config(
    page_title="GearUP",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AI MODEL SETUP (CPU-FRIENDLY) ---
@st.cache_resource
def load_ai_model():
    """Loads a CPU-friendly AI pipeline and caches it."""
    try:
        model_name = "google/flan-t5-base"
        st.info(f"Initializing AI model ({model_name})... This will download the model (~1GB) on the first run.")
        pipe = pipeline("text2text-generation", model=model_name)
        st.success("AI Model loaded successfully!")
        return pipe
    except Exception as e:
        st.error(f"Failed to load AI model. AI features will be disabled. Error: {e}")
        st.error("Please ensure you have an internet connection for the first run and that the 'transformers' and 'torch' libraries are installed.")
        return None

# Initialize the pipeline variable. It will be loaded on demand.
ai_pipe = None

@st.cache_data
def generate_ai_details(_pipe, service_name, vehicle_type, vehicle_model):
    """Generates detailed service information using the AI model."""
    if _pipe is None:
        return "AI model is not available."
    
    prompt = f"""
    As an expert automotive mechanic, provide a detailed explanation in markdown format for a '{service_name}' on a '{vehicle_model}' ({vehicle_type}).
    Your response must include these sections:
    - **Overview**: A brief explanation of the service's purpose.
    - **Key Checks & Replacements**: A bulleted list of main tasks (e.g., Engine oil drained and replaced).
    - **Why It's Crucial**: An explanation of the benefits for the vehicle.
    - **Signs It's Overdue**: A list of common symptoms of neglect.
    """
    
    with st.spinner(f"AI is generating details for {service_name}..."):
        outputs = _pipe(prompt, max_length=512, do_sample=True, temperature=0.7, top_k=50)
    
    if outputs and isinstance(outputs, list) and 'generated_text' in outputs[0]:
        return outputs[0]['generated_text']
    
    return "Could not generate AI response. Please try again."


@st.cache_data
def format_text_with_ai(_pipe, text_to_format, context_prompt):
    """Uses AI to reformat text into a more readable format (e.g., with bullet points)."""
    if _pipe is None:
        return text_to_format

    prompt = f"""
    {context_prompt}
    
    Reformat the following text into a clear, easy-to-read summary using markdown, especially bullet points for key items.

    ---
    TEXT TO FORMAT:
    {text_to_format}
    ---
    """
    
    with st.spinner("AI is reformatting the text for better readability..."):
        outputs = _pipe(prompt, max_length=512, do_sample=True, temperature=0.2, top_k=50)

    if outputs and isinstance(outputs, list) and 'generated_text' in outputs[0]:
        return outputs[0]['generated_text']
    
    return text_to_format

# --- WIKIMEDIA API FUNCTIONS (Wikipedia & Wikibooks) ---

def search_wikimedia(query, project='wikipedia', limit=1):
    """Search Wikipedia or Wikibooks for a given query."""
    base_url = {
        'wikipedia': "https://en.wikipedia.org/w/api.php",
        'wikibooks': "https://en.wikibooks.org/w/api.php"
    }
    
    try:
        search_url = base_url.get(project, 'wikipedia')
        search_params = {
            'action': 'query', 'format': 'json', 'list': 'search',
            'srsearch': query, 'srlimit': limit
        }
        
        response = requests.get(search_url, params=search_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'query' in data and 'search' in data['query'] and data['query']['search']:
            return data['query']['search'][0]['title']
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error searching {project}: {e}")
        return None

def get_wikimedia_summary(title, project='wikipedia'):
    """Get a more detailed Wikipedia or Wikibooks page summary."""
    if not title:
        return "No relevant article found."
    try:
        cache_key = f"{project}_{title}"
        if cache_key in st.session_state.get('parts_cache', {}):
            return st.session_state.parts_cache[cache_key]
        
        base_url = {
            'wikipedia': "https://en.wikipedia.org/w/api.php",
            'wikibooks': "https://en.wikibooks.org/w/api.php"
        }
        api_url = base_url.get(project, 'wikipedia')
        params = {
            'action': 'query', 'format': 'json', 'titles': title,
            'prop': 'extracts', 'explaintext': True,
            'exchars': 1500
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pages = data['query']['pages']
        page_id = list(pages.keys())[0]
        
        if page_id != '-1' and 'extract' in pages[page_id] and pages[page_id]['extract']:
            summary = pages[page_id]['extract']
            summary = re.sub(r'\n+', '\n', summary).strip()
            st.session_state.parts_cache[cache_key] = summary
            return summary
        return "Information not available."
    except requests.exceptions.RequestException as e:
        return f"Network error retrieving information: {e}"

def get_part_information(part_name):
    """Get detailed information about a vehicle part from Wikipedia."""
    search_query = f"Automotive {part_name}"
    title = search_wikimedia(search_query, project='wikipedia')
    
    if title:
        return get_wikimedia_summary(title, project='wikipedia')
    else:
        title = search_wikimedia(part_name, project='wikipedia')
        if title:
            return get_wikimedia_summary(title, project='wikipedia')
    
    return "Information not available for this part."

def get_maintenance_tips(vehicle_type):
    """Get maintenance tips from Wikibooks (preferred) or Wikipedia."""
    query_map = {
        'Car': 'Automobile Maintenance',
        'Bike': 'Motorcycle Maintenance',
        'Scooter': 'Motor scooter maintenance'
    }
    search_term = query_map.get(vehicle_type, f'Automobile Maintenance/{vehicle_type}')

    wikibooks_query = search_term
    title = search_wikimedia(wikibooks_query, project='wikibooks')
    if title:
        summary = get_wikimedia_summary(title, project='wikibooks')
        if summary and "not available" not in summary.lower():
            return summary, "Wikibooks"

    wikipedia_query = f"Automobile maintenance"
    title = search_wikimedia(wikipedia_query, project='wikipedia')
    if title:
        summary = get_wikimedia_summary(title, project='wikipedia')
        if summary and "not available" not in summary.lower():
            return summary, "Wikipedia"
            
    return "General maintenance tips not available.", "N/A"

# --- CONFIGURATION (Continued) ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #DC143C 0%, #FF6B6B 100%); padding: 2rem; border-radius: 10px;
        margin-bottom: 2rem; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #DC143C;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 1rem;
    }
    .service-card {
        background: #FFF5F5; padding: 1.5rem; border-radius: 10px; border: 2px solid #DC143C; margin-bottom: 1rem;
    }
    .info-card {
        background: #F8F9FA; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #007BFF; margin-bottom: 1rem;
    }
    .success-alert {
        background: #E8F5E8; border: 2px solid #4CAF50; border-radius: 10px; padding: 1rem; margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #DC143C 0%, #FF6B6B 100%); color: white; border: none;
        border-radius: 25px; padding: 0.5rem 2rem; font-weight: bold; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #B91C3C 0%, #EF4444 100%); transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .part-info {
        background: #F0F8FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #007BFF; margin: 1rem 0;
    }
    .tips-section {
        background: #FFF9E6; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #FFA500; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'vehicles' not in st.session_state:
    st.session_state.vehicles = []
if 'parts_cache' not in st.session_state:
    st.session_state.parts_cache = {}

# --- EXPANDED DATA DEFINITIONS ---

VEHICLE_DATA = {
    'Car': {
        'Maruti Suzuki': {
            'Swift': {'price_range': (600000, 900000), 'fuel_type': 'Petrol'},
            'Baleno': {'price_range': (700000, 1000000), 'fuel_type': 'Petrol'},
            'Dzire': {'price_range': (650000, 950000), 'fuel_type': 'Petrol'},
            'Ertiga': {'price_range': (850000, 1300000), 'fuel_type': 'Petrol/CNG'},
            'Brezza': {'price_range': (830000, 1400000), 'fuel_type': 'Petrol'},
            'Grand Vitara': {'price_range': (1100000, 2000000), 'fuel_type': 'Petrol/Hybrid'}
        },
        'Hyundai': {
            'Creta': {'price_range': (1100000, 1900000), 'fuel_type': 'Petrol/Diesel'},
            'i20': {'price_range': (750000, 1200000), 'fuel_type': 'Petrol'},
            'Verna': {'price_range': (1100000, 1700000), 'fuel_type': 'Petrol'},
            'Venue': {'price_range': (790000, 1350000), 'fuel_type': 'Petrol/Diesel'},
            'Exter': {'price_range': (610000, 1030000), 'fuel_type': 'Petrol/CNG'},
        },
        'Tata': {
            'Nexon': {'price_range': (815000, 1580000), 'fuel_type': 'Petrol/Diesel/EV'},
            'Punch': {'price_range': (613000, 1020000), 'fuel_type': 'Petrol/CNG'},
            'Altroz': {'price_range': (665000, 1080000), 'fuel_type': 'Petrol/Diesel/CNG'},
            'Harrier': {'price_range': (1550000, 2650000), 'fuel_type': 'Diesel'},
            'Safari': {'price_range': (1620000, 2740000), 'fuel_type': 'Diesel'},
        },
        'Mahindra': {
            'XUV700': {'price_range': (1400000, 2700000), 'fuel_type': 'Petrol/Diesel'},
            'Scorpio-N': {'price_range': (1360000, 2460000), 'fuel_type': 'Petrol/Diesel'},
            'Thar': {'price_range': (1125000, 1760000), 'fuel_type': 'Petrol/Diesel'},
            'XUV300': {'price_range': (800000, 1475000), 'fuel_type': 'Petrol/Diesel'},
        },
        'Kia': {
            'Seltos': {'price_range': (1090000, 2035000), 'fuel_type': 'Petrol/Diesel'},
            'Sonet': {'price_range': (799000, 1570000), 'fuel_type': 'Petrol/Diesel'},
            'Carens': {'price_range': (1045000, 1945000), 'fuel_type': 'Petrol/Diesel'},
        },
    },
    'Bike': {
        'Hero MotoCorp': {
            'Splendor Plus': {'price_range': (75000, 80000), 'fuel_type': 'Petrol'},
            'HF Deluxe': {'price_range': (60000, 70000), 'fuel_type': 'Petrol'},
            'XPulse 200': {'price_range': (145000, 155000), 'fuel_type': 'Petrol'},
            'Glamour': {'price_range': (82000, 88000), 'fuel_type': 'Petrol'},
            'Xtreme 160R': {'price_range': (122000, 133000), 'fuel_type': 'Petrol'},
            'Karizma XMR': {'price_range': (180000, 190000), 'fuel_type': 'Petrol'},
        },
        'Honda': {
            'CB Shine': {'price_range': (80000, 95000), 'fuel_type': 'Petrol'},
            'SP 125': {'price_range': (86000, 91000), 'fuel_type': 'Petrol'},
            'Hornet 2.0': {'price_range': (139000, 141000), 'fuel_type': 'Petrol'},
            'Unicorn': {'price_range': (110000, 112000), 'fuel_type': 'Petrol'},
        },
        'TVS': {
            'Apache RTR 160': {'price_range': (120000, 130000), 'fuel_type': 'Petrol'},
            'Raider 125': {'price_range': (95000, 105000), 'fuel_type': 'Petrol'},
            'Ronin': {'price_range': (149000, 172000), 'fuel_type': 'Petrol'},
            'Apache RR 310': {'price_range': (272000, 275000), 'fuel_type': 'Petrol'},
        },
        'Bajaj': {
            'Pulsar 150': {'price_range': (110000, 120000), 'fuel_type': 'Petrol'},
            'Platina 100': {'price_range': (65000, 70000), 'fuel_type': 'Petrol'},
            'Dominar 400': {'price_range': (230000, 240000), 'fuel_type': 'Petrol'},
            'Pulsar N250': {'price_range': (150000, 152000), 'fuel_type': 'Petrol'},
        },
        'Royal Enfield': {
            'Classic 350': {'price_range': (190000, 220000), 'fuel_type': 'Petrol'},
            'Himalayan': {'price_range': (215000, 230000), 'fuel_type': 'Petrol'},
            'Meteor 350': {'price_range': (205000, 225000), 'fuel_type': 'Petrol'},
            'Hunter 350': {'price_range': (150000, 175000), 'fuel_type': 'Petrol'},
            'Interceptor 650': {'price_range': (303000, 331000), 'fuel_type': 'Petrol'},
            'Super Meteor 650': {'price_range': (364000, 394000), 'fuel_type': 'Petrol'},
        },
        'Yamaha': {
            'FZ-S FI': {'price_range': (120000, 140000), 'fuel_type': 'Petrol'},
            'R15 V4': {'price_range': (180000, 200000), 'fuel_type': 'Petrol'},
            'MT-15': {'price_range': (168000, 174000), 'fuel_type': 'Petrol'},
        },
        'Suzuki': {
            'Gixxer': {'price_range': (135000, 145000), 'fuel_type': 'Petrol'},
            'Gixxer SF': {'price_range': (140000, 150000), 'fuel_type': 'Petrol'},
            'V-Strom SX': {'price_range': (212000, 215000), 'fuel_type': 'Petrol'},
        },
        'KTM': {
            'Duke 200': {'price_range': (197000, 200000), 'fuel_type': 'Petrol'},
            'Duke 390': {'price_range': (311000, 315000), 'fuel_type': 'Petrol'},
            'RC 200': {'price_range': (218000, 220000), 'fuel_type': 'Petrol'},
            'Adventure 390': {'price_range': (339000, 342000), 'fuel_type': 'Petrol'},
        }
    },
    'Scooter': {
        'Honda': {
            'Activa': {'price_range': (75000, 90000), 'fuel_type': 'Petrol'},
            'Dio': {'price_range': (70000, 78000), 'fuel_type': 'Petrol'},
            'Grazia': {'price_range': (85000, 92000), 'fuel_type': 'Petrol'},
        },
        'TVS': {
            'Jupiter': {'price_range': (73000, 89000), 'fuel_type': 'Petrol'},
            'Ntorq 125': {'price_range': (84000, 105000), 'fuel_type': 'Petrol'},
            'iQube': {'price_range': (117000, 156000), 'fuel_type': 'EV'},
        },
        'Suzuki': {
            'Access 125': {'price_range': (80000, 90000), 'fuel_type': 'Petrol'},
            'Burgman Street': {'price_range': (94000, 115000), 'fuel_type': 'Petrol'},
        },
        'Ola Electric': {
            'S1 Pro': {'price_range': (130000, 140000), 'fuel_type': 'EV'},
            'S1 Air': {'price_range': (105000, 115000), 'fuel_type': 'EV'},
            'S1 X': {'price_range': (75000, 100000), 'fuel_type': 'EV'},
        },
        'Ather Energy': {
            '450X': {'price_range': (128000, 148000), 'fuel_type': 'EV'},
            '450S': {'price_range': (118000, 130000), 'fuel_type': 'EV'},
        },
        'Bajaj': {
            'Chetak': {'price_range': (115000, 145000), 'fuel_type': 'EV'},
        },
        'Yamaha': {
            'RayZR 125': {'price_range': (85000, 95000), 'fuel_type': 'Petrol'},
            'Fascino 125': {'price_range': (80000, 92000), 'fuel_type': 'Petrol'},
        }
    }
}

SERVICE_SCHEDULES = {
    'Car': {
        'General Service': {'interval_months': 6, 'interval_km': 10000, 'cost_range': (2500, 5000)},
        'Oil Change': {'interval_months': 12, 'interval_km': 10000, 'cost_range': (2000, 3500)},
        'Brake Service': {'interval_months': 24, 'interval_km': 20000, 'cost_range': (3000, 8000)},
        'Major Service': {'interval_months': 24, 'interval_km': 40000, 'cost_range': (8000, 15000)},
        'Tire Replacement': {'interval_months': 48, 'interval_km': 50000, 'cost_range': (18000, 40000)}
    },
    'Bike': {
        'General Service': {'interval_months': 3, 'interval_km': 3000, 'cost_range': (800, 1500)},
        'Oil Change': {'interval_months': 6, 'interval_km': 4000, 'cost_range': (500, 900)},
        'Chain Service': {'interval_months': 12, 'interval_km': 15000, 'cost_range': (1500, 3000)},
        'Brake Pads': {'interval_months': 12, 'interval_km': 10000, 'cost_range': (700, 1500)},
        'Tire Replacement': {'interval_months': 36, 'interval_km': 30000, 'cost_range': (4000, 8000)}
    },
    'Scooter': {
        'General Service': {'interval_months': 4, 'interval_km': 4000, 'cost_range': (700, 1200)},
        'Oil Change': {'interval_months': 6, 'interval_km': 6000, 'cost_range': (400, 700)},
        'Brake Service': {'interval_months': 12, 'interval_km': 10000, 'cost_range': (600, 1200)},
        'CVT & Transmission Service': {'interval_months': 24, 'interval_km': 20000, 'cost_range': (1000, 2500)},
        'Tire Replacement': {'interval_months': 36, 'interval_km': 25000, 'cost_range': (3000, 6000)}
    }
}

SERVICE_PARTS = {
    'Car': {
        'General Service': ['Engine Oil', 'Oil Filter', 'Air Filter', 'Coolant', 'Wiper Fluid'],
        'Oil Change': ['Engine Oil', 'Oil Filter'],
        'Brake Service': ['Brake Pads', 'Brake Fluid', 'Brake Discs'],
        'Major Service': ['Engine Oil', 'Oil Filter', 'Air Filter', 'Spark Plugs', 'Fuel Filter', 'Coolant'],
        'Tire Replacement': ['Tires', 'Wheel Alignment', 'Wheel Balancing']
    },
    'Bike': {
        'General Service': ['Engine Oil', 'Air Filter', 'Chain Lubrication', 'Brake Check'],
        'Oil Change': ['Engine Oil', 'Oil Filter'],
        'Chain Service': ['Chain', 'Front Sprocket', 'Rear Sprocket'],
        'Brake Pads': ['Front Brake Pads', 'Rear Brake Pads', 'Brake Fluid'],
        'Tire Replacement': ['Front Tire', 'Rear Tire', 'Tube']
    },
    'Scooter': {
        'General Service': ['Engine Oil', 'Air Filter', 'Spark Plug', 'Brake Check'],
        'Oil Change': ['Engine Oil'],
        'Brake Service': ['Brake Pads', 'Brake Shoes', 'Brake Fluid'],
        'CVT & Transmission Service': ['CVT Belt', 'Rollers', 'Gear Oil'],
        'Tire Replacement': ['Front Tire', 'Rear Tire', 'Tube']
    }
}

USAGE_MULTIPLIERS = {
    'City': {'service_frequency': 1.2, 'parts_wear': 1.1},
    'Highway': {'service_frequency': 0.9, 'parts_wear': 0.8},
    'Mixed': {'service_frequency': 1.0, 'parts_wear': 1.0},
    'Commercial': {'service_frequency': 1.5, 'parts_wear': 1.4}
}

SERVICE_CENTERS = {
    'Hyderabad': [
        {'name': 'AutoCare Service Center', 'rating': 4.5, 'address': 'Banjara Hills, Hyderabad, Telangana', 'phone': '+91-9876543210'},
        {'name': 'ProService Hub', 'rating': 4.7, 'address': 'Gachibowli, Hyderabad, Telangana', 'phone': '+91-9876543212'},
        {'name': 'MegaAuto Care', 'rating': 4.2, 'address': 'Kondapur, Hyderabad, Telangana', 'phone': '+91-9876543213'},
    ],
    'Mumbai': [
        {'name': 'Mumbai Motors', 'rating': 4.6, 'address': 'Andheri West, Mumbai, Maharashtra', 'phone': '+91-9123456780'},
        {'name': 'Speedy Service Point', 'rating': 4.3, 'address': 'Bandra, Mumbai, Maharashtra', 'phone': '+91-9123456781'},
    ]
}

SHOWROOMS = {
    'Hyderabad': [
        {'name': 'Honda Pride Showroom', 'rating': 4.6, 'address': 'Jubilee Hills, Hyderabad, Telangana', 'phone': '+91-8888888888'},
        {'name': 'Toyota Pinnacle Motors', 'rating': 4.4, 'address': 'Madhapur, Hyderabad, Telangana', 'phone': '+91-7777777777'},
        {'name': 'Maruti Suzuki Arena', 'rating': 4.8, 'address': 'Somajiguda, Hyderabad, Telangana', 'phone': '+91-6666666666'}
    ],
    'Mumbai': [
        {'name': 'Honda Drive', 'rating': 4.7, 'address': 'Worli, Mumbai, Maharashtra', 'phone': '+91-8123456780'},
        {'name': 'Bajaj Auto Exchange', 'rating': 4.5, 'address': 'Dadar, Mumbai, Maharashtra', 'phone': '+91-8123456781'},
    ]
}

# --- HELPER FUNCTIONS ---
def get_recommended_parts(vehicle_data):
    """Recommend parts based on kilometers and usage."""
    current_km = vehicle_data.get('current_km', 0)
    usage = vehicle_data.get('usage', 'Mixed')
    v_type = vehicle_data.get('type')
    
    recommended_parts = []

    if 50000 < current_km <= 80000: recommended_parts.extend(['Brake Pads', 'Tires'])
    if current_km > 80000: recommended_parts.extend(['Spark Plugs', 'Suspension Components'])
    if usage == 'City': recommended_parts.extend(['Brake Pads', 'Air Filter'])
    elif usage == 'Commercial': recommended_parts.extend(['Engine Oil', 'Brake Pads', 'Suspension Components', 'Tires'])
    
    if v_type == 'Car':
        if current_km > 80000: recommended_parts.append('Timing Belt')
        if current_km > 100000: recommended_parts.append('Water Pump')
        if usage == 'City': recommended_parts.append('Clutch Assembly')
    elif v_type == 'Bike':
        if usage == 'City': recommended_parts.append('Clutch Plates')
    elif v_type == 'Scooter':
        if 20000 < current_km <= 40000: recommended_parts.append('CVT Belt')
    
    return sorted(list(set(recommended_parts)))


def calculate_service_due(vehicle_data):
    current_date = datetime.date.today()
    purchase_date = vehicle_data['purchase_date']
    current_km = vehicle_data.get('current_km', 0)
    usage = vehicle_data.get('usage', 'Mixed')
    
    services_due = []
    vehicle_type = vehicle_data['type']
    multiplier = USAGE_MULTIPLIERS.get(usage, USAGE_MULTIPLIERS['Mixed'])
    
    if vehicle_type not in SERVICE_SCHEDULES:
        return []

    for service_name, service_info in SERVICE_SCHEDULES[vehicle_type].items():
        adjusted_interval_months = int(service_info['interval_months'] / multiplier['service_frequency'])
        adjusted_interval_km = int(service_info['interval_km'] / multiplier['parts_wear'])
        
        last_time_service_km = vehicle_data.get(f'last_service_km_{service_name}', vehicle_data.get('initial_km', 0))
        last_time_service_date = vehicle_data.get(f'last_service_date_{service_name}', purchase_date)
        
        km_since_last = current_km - last_time_service_km
        months_since_last = ((current_date - last_time_service_date).days / 30.44)

        time_due = (adjusted_interval_months > 0 and months_since_last >= adjusted_interval_months)
        km_due = (adjusted_interval_km > 0 and km_since_last >= adjusted_interval_km)
        
        if time_due or km_due:
            days_overdue = 0
            if time_due:
                days_overdue = (current_date - (last_time_service_date + timedelta(days=adjusted_interval_months * 30.44))).days

            reason = []
            if time_due: reason.append("Time Interval")
            if km_due: reason.append("Kilometer Reading")

            services_due.append({
                'service': service_name,
                'due_reason': ' & '.join(reason),
                'days_overdue': max(0, int(days_overdue)),
                'cost_range': service_info['cost_range'],
                'parts': SERVICE_PARTS[vehicle_type].get(service_name, [])
            })
    return services_due


# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>GearUP</h1>
    <p>Intelligent Vehicle Maintenance & Management</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://i.imgur.com/u41Wd31.png", width=80)
    st.markdown("### Navigation")
    page = st.selectbox("Choose a page:", [
        "Dashboard", "Add Vehicle", "Maintenance Schedule", 
        "Parts Encyclopedia", "Service Locator", "Analytics", "Maintenance Tips"
    ])
    st.markdown("---")
    st.info("AI-powered features may take a moment to generate. The AI model will be downloaded on first use.")

# --- PAGE: DASHBOARD ---
if page == "Dashboard":
    st.markdown("## Dashboard Overview")
    if not st.session_state.vehicles:
        st.markdown("""
        <div class="metric-card">
            <h3>Welcome to GearUP!</h3>
            <p>Get started by adding your first vehicle to track maintenance schedules, costs, and get AI-powered insights.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("Please navigate to the 'Add Vehicle' page from the sidebar to add your vehicle.")
    else:
        col1, col2, col3 = st.columns(3)
        total_vehicles = len(st.session_state.vehicles)
        total_due_services = sum(len(calculate_service_due(v)) for v in st.session_state.vehicles)
        avg_vehicle_age = np.mean([(datetime.date.today() - v['purchase_date']).days / 365 for v in st.session_state.vehicles]) if st.session_state.vehicles else 0
        
        with col1: st.metric("Total Vehicles", total_vehicles)
        with col2: st.metric("Services Due", total_due_services)
        with col3: st.metric("Average Vehicle Age", f"{avg_vehicle_age:.1f} years")
        
        st.markdown("### Your Fleet Status")
        for i, vehicle in enumerate(st.session_state.vehicles):
            services_due = calculate_service_due(vehicle)
            recommended_parts = get_recommended_parts(vehicle)
            
            with st.expander(f"**{vehicle['brand']} {vehicle['model']} ({vehicle['year']})** - {vehicle['type']}", expanded=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Purchase Date:** {vehicle['purchase_date']}")
                    st.write(f"**Purchase Price:** ₹{vehicle['price']:,}")
                    st.write(f"**Current Kilometers:** {vehicle.get('current_km', 0):,} km")
                    st.write(f"**Usage Pattern:** {vehicle.get('usage', 'N/A')}")
                with col2:
                    if services_due:
                        st.error(f"**{len(services_due)} Services Due**")
                        for service in services_due[:3]:
                            st.write(f"• {service['service']}")
                    else:
                        st.success("**All services up to date**")
                    
                    if recommended_parts:
                        st.warning("**Recommended Parts**")
                        for part in recommended_parts[:3]:
                            st.write(f"• {part}")

# --- PAGE: ADD VEHICLE ---
elif page == "Add Vehicle":
    st.markdown("## Add New Vehicle")
    with st.form("add_vehicle_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Add a 'key' to the first selectbox. Its value will be stored in st.session_state.
            vehicle_type = st.selectbox(
                "Vehicle Type", 
                list(VEHICLE_DATA.keys()), 
                key='vehicle_type_selector'
            )

            # Use the value from session_state to dynamically populate the options for the next box.
            brand_options = list(VEHICLE_DATA[st.session_state.vehicle_type_selector].keys())
            brand = st.selectbox(
                "Brand", 
                brand_options, 
                key='brand_selector'
            )
            
            # Do the same for the model selector.
            model_options = list(VEHICLE_DATA[st.session_state.vehicle_type_selector][st.session_state.brand_selector].keys())
            model = st.selectbox(
                "Model", 
                model_options,
                key='model_selector'
            )
            
            year = st.number_input("Year", min_value=2000, max_value=datetime.date.today().year + 1, value=datetime.date.today().year)
        
        with col2:
            purchase_date = st.date_input("Purchase Date", value=datetime.date.today(), max_value=datetime.date.today())
            
            # Use the up-to-date session_state values to calculate the default price
            selected_type = st.session_state.vehicle_type_selector
            selected_brand = st.session_state.brand_selector
            selected_model = st.session_state.model_selector
            
            default_price_tuple = VEHICLE_DATA[selected_type][selected_brand].get(selected_model, {'price_range': (50000, 100000)})['price_range']
            default_price = int(np.mean(default_price_tuple))
            
            price = st.number_input("Purchase Price (₹)", min_value=0, value=default_price)
            current_km = st.number_input("Current Kilometers", min_value=0, value=0)
            usage = st.selectbox("Primary Usage Pattern", ["City", "Highway", "Mixed", "Commercial"])
        
        submitted = st.form_submit_button("Add Vehicle")
        if submitted:
            # On submission, retrieve the final values from session_state and other widgets
            new_vehicle = {
                'type': st.session_state.vehicle_type_selector, 
                'brand': st.session_state.brand_selector, 
                'model': st.session_state.model_selector, 
                'year': year,
                'purchase_date': purchase_date, 
                'price': price, 
                'current_km': current_km,
                'initial_km': current_km, 
                'usage': usage
            }
            st.session_state.vehicles.append(new_vehicle)
            st.success(f"**{st.session_state.brand_selector} {st.session_state.model_selector}** added to your fleet!")
            st.balloons()

# --- PAGE: MAINTENANCE SCHEDULE ---
elif page == "Maintenance Schedule":
    st.markdown("## Maintenance Schedule & AI Advisor")
    if not st.session_state.vehicles:
        st.warning("No vehicles added yet. Please add a vehicle first.")
    else:
        vehicle_options = [f"{v['type']} - {v['brand']} {v['model']} ({v['year']})" for v in st.session_state.vehicles]
        selected_vehicle_idx = st.selectbox("Select Vehicle", range(len(vehicle_options)), format_func=lambda x: vehicle_options[x])
        selected_vehicle = st.session_state.vehicles[selected_vehicle_idx]
        
        st.markdown("### Update Vehicle Mileage")
        col1, _ = st.columns([1, 2])
        with col1:
            new_km = st.number_input("Enter Current Kilometers", min_value=selected_vehicle.get('current_km', 0), value=selected_vehicle.get('current_km', 0), key=f"km_update_{selected_vehicle_idx}")
            if st.button("Update Kilometers"):
                st.session_state.vehicles[selected_vehicle_idx]['current_km'] = new_km
                st.success("Kilometers updated!")
                st.rerun()

        services_due = calculate_service_due(selected_vehicle)
        if services_due:
            st.markdown("### Services Due")
            for i, service in enumerate(services_due):
                st.markdown(f"""
                <div class="service-card">
                    <h4>{service['service']}</h4>
                    <p><strong>Due Reason:</strong> {service['due_reason']}</p>
                    <p><strong>Estimated Cost:</strong> ₹{service['cost_range'][0]:,} - ₹{service['cost_range'][1]:,}</p>
                    <p><strong>Parts Involved:</strong> {', '.join(service['parts'])}</p>
                    {f"<p style='color: red;'><strong>Note:</strong> This service might be overdue.</p>" if service['days_overdue'] > 7 else ""}
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Get AI-Powered Details for {service['service']}", key=f"ai_btn_{selected_vehicle_idx}_{i}"):
                    if ai_pipe is None:
                        ai_pipe = load_ai_model()
                    
                    if ai_pipe:
                        ai_details = generate_ai_details(ai_pipe, service['service'], selected_vehicle['type'], selected_vehicle['model'])
                        st.info(f"AI-Powered Advice for {service['service']}")
                        st.markdown(ai_details)
                    else:
                        st.error("AI Model is not available. Could not fetch details.")
        else:
            st.markdown("""
            <div class="success-alert">
                <h4>All Services Up to Date</h4>
                <p>Your vehicle maintenance is current based on standard intervals. Keep up the great work!</p>
            </div>
            """, unsafe_allow_html=True)

# --- PAGE: PARTS ENCYCLOPEDIA ---
elif page == "Parts Encyclopedia":
    st.markdown("## Vehicle Parts Encyclopedia")
    st.info("Learn about common vehicle parts, powered by Wikipedia and enhanced by AI.")
    
    all_parts = set()
    for vehicle_type in SERVICE_PARTS:
        for service in SERVICE_PARTS[vehicle_type]:
            all_parts.update(SERVICE_PARTS[vehicle_type][service])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_part = st.selectbox("Select a part to learn about:", sorted(list(all_parts)))
        get_info_button = st.button("Get Information")
    
    if get_info_button:
        with st.spinner(f"Fetching information for {selected_part}..."):
            wiki_info = get_part_information(selected_part)
            st.session_state.current_part_info = wiki_info
            
            if ai_pipe is None:
                ai_pipe = load_ai_model()
            
            if ai_pipe and "not available" not in wiki_info.lower():
                context = f"You are explaining the automotive part '{selected_part}' to a car owner."
                formatted_info = format_text_with_ai(ai_pipe, wiki_info, context)
                st.session_state.formatted_part_info = formatted_info
            else:
                st.session_state.formatted_part_info = wiki_info
    
    with col2:
        if 'formatted_part_info' in st.session_state and st.session_state.formatted_part_info:
            st.markdown(f"""
            <div class="part-info">
                <h3>{selected_part}</h3>
                {st.session_state.formatted_part_info}
            </div>
            """, unsafe_allow_html=True)
        elif 'current_part_info' in st.session_state and st.session_state.current_part_info:
             st.markdown(f"""
            <div class="part-info">
                <h3>{selected_part}</h3>
                <p>{st.session_state.current_part_info}</p>
            </div>
            """, unsafe_allow_html=True)


# --- PAGE: SERVICE LOCATOR ---
elif page == "Service Locator":
    st.markdown("## Service & Showroom Locator")
    city = st.text_input("Enter your city", value="Hyderabad").strip().title()

    try:
        geolocator = Nominatim(user_agent="gearup_app_v4")
        city_location = geolocator.geocode(city, timeout=10)
    except Exception as e:
        st.error(f"Could not connect to geocoding service. Please check your internet connection. Error: {e}")
        city_location = None

    if city_location:
        st.write(f"Displaying locations in and around **{city}**.")
        m = folium.Map(location=[city_location.latitude, city_location.longitude], zoom_start=12)

        def add_location_marker(location_list, category, color, icon):
            for loc in location_list.get(city, []):
                try:
                    point = geolocator.geocode(loc['address'], timeout=10)
                    if point:
                        gmaps_query = urllib.parse.quote_plus(f"{loc['name']}, {loc['address']}")
                        gmaps_url = f"https://www.google.com/maps/search/?api=1&query={gmaps_query}"
                        
                        popup_html = f"""
                        <b>{loc['name']}</b><br>
                        Rating: {loc['rating']} ⭐<br>
                        Phone: {loc['phone']}<br>
                        <a href="{gmaps_url}" target="_blank">Open in Google Maps</a>
                        """
                        folium.Marker(
                            [point.latitude, point.longitude],
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=loc['name'],
                            icon=folium.Icon(color=color, icon=icon, prefix='fa')
                        ).add_to(m)
                except Exception as e:
                    print(f"Could not geocode address '{loc['address']}': {e}")
        
        st.markdown("#### Map Legend")
        st.markdown("- <span style='color:red;'>**Red Wrench Icon:**</span> Service Centers\n"
                    "- <span style='color:blue;'>**Blue Car Icon:**</span> Showrooms", 
                    unsafe_allow_html=True)
                    
        add_location_marker(SERVICE_CENTERS, "Service Centers", "red", "wrench")
        add_location_marker(SHOWROOMS, "Showrooms", "blue", "car")

        st_folium(m, use_container_width=True, height=500)
    elif city:
        st.error(f"Could not find the city: {city}. Please check the spelling or try a nearby major city.")


# --- PAGE: ANALYTICS ---
elif page == "Analytics":
    st.markdown("## Analytics & Insights")
    if not st.session_state.vehicles:
        st.warning("No vehicles added yet. Please add vehicles to see analytics.")
    else:
        st.markdown("### Fleet Overview")
        col1, col2 = st.columns(2)
        with col1:
            type_counts = pd.Series([v['type'] for v in st.session_state.vehicles]).value_counts()
            fig_pie = px.pie(values=type_counts.values, names=type_counts.index, title="<b>Vehicle Type Distribution</b>",
                             color_discrete_sequence=['#DC143C', '#FF6B6B', '#FFA07A', '#CD5C5C'])
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            brand_counts = pd.Series([v['brand'] for v in st.session_state.vehicles]).value_counts()
            fig_bar = px.bar(x=brand_counts.index, y=brand_counts.values, title="<b>Vehicle Brands in Your Fleet</b>",
                             labels={'x': 'Brand', 'y': 'Count'}, color_discrete_sequence=['#DC143C'])
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("### Service Cost Analysis")
        service_data = []
        for v in st.session_state.vehicles:
            for s in calculate_service_due(v):
                service_data.append({'vehicle': f"{v['brand']} {v['model']}", 'service': s['service'], 'cost': np.mean(s['cost_range'])})
        
        if service_data:
            df_costs = pd.DataFrame(service_data)
            total_estimated_cost = df_costs['cost'].sum()
            avg_cost = total_estimated_cost / len(st.session_state.vehicles) if st.session_state.vehicles else 0
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Services Currently Due", len(df_costs))
            with col2: st.metric("Est. Cost for Due Services", f"₹{total_estimated_cost:,.0f}")
            with col3: st.metric("Avg. Due Cost per Vehicle", f"₹{avg_cost:,.0f}")

            fig_cost_breakdown = px.bar(df_costs, x='service', y='cost', color='vehicle', title="<b>Estimated Cost Breakdown by Service and Vehicle</b>")
            st.plotly_chart(fig_cost_breakdown, use_container_width=True)
        else:
            st.success("No services currently due across your fleet!")

# --- PAGE: MAINTENANCE TIPS ---
elif page == "Maintenance Tips":
    st.markdown("## Maintenance Tips & Best Practices")
    st.info("Practical guides sourced from Wikibooks/Wikipedia and formatted by AI for clarity.")
    
    vehicle_types_in_fleet = sorted(list(set([v['type'] for v in st.session_state.vehicles])))
    if not vehicle_types_in_fleet:
        vehicle_types_in_fleet = ['Car', 'Bike', 'Scooter']
    
    selected_type = st.selectbox("Select Vehicle Type for Tips:", vehicle_types_in_fleet)
    
    if st.button(f"Get Tips for {selected_type}"):
        with st.spinner(f"Loading tips for {selected_type}..."):
            tips, source = get_maintenance_tips(selected_type)
            if "not available" not in tips.lower():
                if ai_pipe is None:
                    ai_pipe = load_ai_model()
                
                if ai_pipe:
                    context = f"You are creating a maintenance guide for a {selected_type.lower()} owner."
                    formatted_tips = format_text_with_ai(ai_pipe, tips, context)
                else:
                    formatted_tips = tips

                st.markdown(f"""
                <div class="tips-section">
                    <h4>Professional {selected_type} Maintenance Guidelines (Source: {source})</h4>
                    {formatted_tips}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Could not find specific maintenance tips for {selected_type}.")


# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>GearUP - Intelligent Vehicle Maintenance & Management</p>
    <p>Powered by Hugging Face, Wikipedia, and Wikibooks APIs</p>
</div>
""", unsafe_allow_html=True)