import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from io import BytesIO

# PDF Generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CAT Transient Stability Analyzer v6.1", 
    page_icon="⚡", 
    layout="wide"
)

# --- CAT BRANDING CSS ---
st.markdown("""

<style>
    /* CAT Color Palette */
    :root {
        --cat-yellow: #FFCD00;
        --cat-black: #1A1A1A;
        --cat-charcoal: #2D2D2D;
        --cat-gray: #666666;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1A1A1A 0%, #2D2D2D 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #FFCD00;
    }
    .main-header h1 {
        color: #FFCD00 !important;
        margin: 0;
    }
    
    /* --- SIDEBAR STYLING FIX (DEFINITIVO) --- */
    [data-testid="stSidebar"] {
        background-color: #1A1A1A !important;
    }
    
    /* Fuerza TODOS los textos del sidebar a ser legibles */
    [data-testid="stSidebar"] * {
        color: #E0E0E0 !important; /* Blanco hueso para texto general */
    }
    
    /* Títulos y Etiquetas en Amarillo CAT */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label { 
        color: #FFCD00 !important;
    }
    
    /* Arreglo específico para Checkboxes y Radios que a veces se quedan oscuros */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label span p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span p {
        color: #FFFFFF !important;
    }

    /* Inputs y Selectbox (Fondo Gris Oscuro + Texto Blanco) */
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background-color: #333333 !important;
        color: white !important;
        border-color: #FFCD00 !important;
    }
    
    /* Texto seleccionado y valores */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: white !important;
    }

    /* Result boxes */
    .cat-box-success { background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; color: #155724; margin-bottom: 10px; }
    .cat-box-fail { background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545; color: #721c24; margin-bottom: 10px; }
    .cat-box-opt { background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 5px solid #0d47a1; color: #084298; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. ENHANCED DATA LIBRARY WITH DYNAMIC PARAMETERS
# ==============================================================================

CAT_LIBRARY = {
    "XGC1900 (1.9 MW)": {
        "mw": 1.9,
        "type": "High Speed Recip",
        "h_def": 1.0,
        "tau_gov": 0.3,
        "tau_mech": 0.5,
        "droop": 4.0,
        "ramp_up": 0.5,
        "ramp_down": 0.8,
        "xd_pp": 0.14,
        "hr": 8.78,
        "capex": 800,
        "step_load_max": 25,
    },
    "G3520FR (2.5 MW)": {
        "mw": 2.5,
        "type": "High Speed Recip",
        "h_def": 1.0,
        "tau_gov": 0.25,
        "tau_mech": 0.6,
        "droop": 4.0,
        "ramp_up": 1.0,
        "ramp_down": 1.2,
        "xd_pp": 0.14,
        "hr": 8.83,
        "capex": 600,
        "step_load_max": 40,
    },
    "G3520K (2.4 MW)": {
        "mw": 2.4,
        "type": "High Speed Recip",
        "h_def": 1.0,
        "tau_gov": 0.35,
        "tau_mech": 0.6,
        "droop": 5.0,
        "ramp_up": 0.4,
        "ramp_down": 0.6,
        "xd_pp": 0.13,
        "hr": 7.64,
        "capex": 600,
        "step_load_max": 15,
    },
    "CG260-16 (3.96 MW)": {
        "mw": 3.96,
        "type": "High Speed Recip",
        "h_def": 1.2,
        "tau_gov": 0.4,
        "tau_mech": 0.8,
        "droop": 4.0,
        "ramp_up": 0.45,
        "ramp_down": 0.7,
        "xd_pp": 0.15,
        "hr": 7.86,
        "capex": 700,
        "step_load_max": 10,
    },
    "G20CM34 (9.76 MW)": {
        "mw": 9.76,
        "type": "Medium Speed Recip",
        "h_def": 1.5,
        "tau_gov": 0.5,
        "tau_mech": 1.0,
        "droop": 4.0,
        "ramp_up": 0.8,
        "ramp_down": 1.0,
        "xd_pp": 0.16,
        "hr": 7.48,
        "capex": 750,
        "step_load_max": 10,
    },
    "Titan 130 (16.5 MW)": {
        "mw": 16.5,
        "type": "Gas Turbine",
        "h_def": 4.0,
        "tau_gov": 0.8,
        "tau_mech": 1.5,
        "droop": 4.0,
        "ramp_up": 1.5,
        "ramp_down": 2.0,
        "xd_pp": 0.18,
        "hr": 9.63,
        "capex": 800,
        "step_load_max": 15,
    },
    "Titan 250 (23.2 MW)": {
        "mw": 23.2,
        "type": "Gas Turbine",
        "h_def": 4.5,
        "tau_gov": 1.0,
        "tau_mech": 2.0,
        "droop": 4.0,
        "ramp_up": 2.0,
        "ramp_down": 2.5,
        "xd_pp": 0.18,
        "hr": 8.67,
        "capex": 800,
        "step_load_max": 15,
    },
    "Titan 350 (38.0 MW)": {
        "mw": 38.0,
        "type": "Gas Turbine",
        "h_def": 5.0,
        "tau_gov": 1.2,
        "tau_mech": 2.5,
        "droop": 4.0,
        "ramp_up": 3.0,
        "ramp_down": 4.0,
        "xd_pp": 0.20,
        "hr": 8.50,
        "capex": 800,
        "step_load_max": 12,
    }
}

# ==============================================================================
# 2. AI WORKLOAD PROFILES
# ==============================================================================

LOAD_PROFILES = {
    "Single Pulse (Inference)": {
        "type": "pulse",
        "description": "Single inference burst - typical API request spike",
        "default_duration": 5.0,
        "configurable_duration": True,
    },
    "Step Load (Training Start)": {
        "type": "step",
        "description": "Sustained load increase - training job startup",
        "default_duration": 9999.0,
        "configurable_duration": False,
    },
    "Checkpoint Pulse Train": {
        "type": "pulse_train",
        "description": "Periodic spikes during model checkpointing",
        "default_duration": 3.0,
        "interval": 60.0,
        "configurable_duration": True,
    },
    "GPU Cluster Staircase": {
        "type": "staircase",
        "description": "Sequential GPU rack power-on sequence",
        "steps": 4,
        "step_interval": 5.0,
        "configurable_duration": False,
    },
    "Random Inference Bursts": {
        "type": "random",
        "description": "Random inference requests (Monte Carlo)",
        "avg_duration": 2.0,
        "avg_interval": 10.0,
        "configurable_duration": False,
    },
}

def create_load_profile(profile_name, t, params):
    """Generate load profile based on selected pattern."""
    profile = LOAD_PROFILES[profile_name]
    P_spike = params['P_spike']
    t_start = params.get('t_start', 1.0)
    duration = params.get('duration', profile.get('default_duration', 5.0))
    
    load = np.zeros_like(t)
    
    if profile['type'] == 'pulse':
        mask = (t >= t_start) & (t < t_start + duration)
        load[mask] = P_spike
        
    elif profile['type'] == 'step':
        mask = t >= t_start
        load[mask] = P_spike
        
    elif profile['type'] == 'pulse_train':
        interval = profile.get('interval', 60.0)
        for t_pulse in np.arange(t_start, t[-1], interval):
            mask = (t >= t_pulse) & (t < t_pulse + duration)
            load[mask] = P_spike
            
    elif profile['type'] == 'staircase':
        steps = profile.get('steps', 4)
        step_interval = profile.get('step_interval', 5.0)
        step_size = P_spike / steps
        
        for i in range(steps):
            t_step = t_start + i * step_interval
            mask = t >= t_step
            load[mask] = step_size * (i + 1)
            
    elif profile['type'] == 'random':
        np.random.seed(42)
        avg_interval = profile.get('avg_interval', 10.0)
        avg_duration = profile.get('avg_duration', 2.0)
        
        t_current = t_start
        while t_current < t[-1]:
            pulse_duration = np.random.exponential(avg_duration)
            pulse_amplitude = P_spike * np.random.uniform(0.5, 1.0)
            
            mask = (t >= t_current) & (t < t_current + pulse_duration)
            load[mask] = pulse_amplitude
            
            t_current += pulse_duration + np.random.exponential(avg_interval)
    
    return load


# ==============================================================================
# 3. PHYSICS ENGINE
# ==============================================================================

def system_dynamics_v2(y, t, params):
    """Enhanced system dynamics with 2nd-order governor and voltage."""
    delta_f, p_mech, p_gov, p_bess, v_pu = y
    
    f0 = 60.0
    H = max(params['H'], 0.1)
    D = params.get('D', 1.0)
    S_base = max(params['S_base'], 1.0)
    
    R = params['droop'] / 100.0
    T_gov = max(params['T_gov'], 0.05)
    T_mech = max(params['T_mech'], 0.1)
    P_max = params['P_max']
    ramp_up = params['ramp_up']
    ramp_down = params['ramp_down']
    
    P_bess_max = params['P_bess_max']
    T_bess = max(params['T_bess'], 0.02)
    bess_enabled = params['bess_enabled']
    
    X_eq = params.get('X_eq', 0.15)
    
    t_profile = params['t_profile']
    load_profile = params['load_profile']
    P_base = params['P_base']
    
    P_load_dev = np.interp(t, t_profile, load_profile)
    P_load = P_base + P_load_dev
    
    alpha_v = params.get('alpha_v', 0.5)
    P_load_actual = P_load * (v_pu ** alpha_v)
    
    freq_error_pu = -delta_f / f0
    P_ref_droop = P_base + (1.0 / R) * freq_error_pu * P_max
    P_ref = np.clip(P_ref_droop, 0, P_max)
    
    dP_gov_dt = (P_ref - p_gov) / T_gov
    
    delta_p_desired = (p_gov - p_mech) / T_mech
    if delta_p_desired > 0:
        delta_p = min(delta_p_desired, ramp_up)
    else:
        delta_p = max(delta_p_desired, -ramp_down)
    dP_mech_dt = delta_p
    
    if bess_enabled and P_bess_max > 0:
        P_deficit = P_load_actual - p_mech
        freq_response = -delta_f * P_bess_max * 2.0
        P_bess_target = P_deficit + freq_response
        P_bess_target = np.clip(P_bess_target, -P_bess_max, P_bess_max)
        dP_bess_dt = (P_bess_target - p_bess) / T_bess
    else:
        dP_bess_dt = -p_bess / 0.1
    
    P_acc = p_mech + p_bess - P_load_actual
    P_damping = D * delta_f
    dF_dt = ((P_acc - P_damping) / S_base) * (f0 / (2 * H))
    
    Q_imbalance = (P_load_actual - p_mech - p_bess) * 0.3
    dV_dt = -X_eq * Q_imbalance / (v_pu * S_base * 10)
    dV_dt = np.clip(dV_dt, -0.5, 0.5)
    
    return [dF_dt, dP_mech_dt, dP_gov_dt, dP_bess_dt, dV_dt]


def calculate_voltage_sag(P_step_mw, X_d_pp, n_gens, S_base):
    """Calculate instantaneous voltage sag during step load."""
    if n_gens <= 0 or S_base <= 0:
        return 0.0
    X_eq = X_d_pp / np.sqrt(n_gens)
    voltage_sag_pct = (P_step_mw / S_base) * X_eq * 100
    return voltage_sag_pct


def calculate_rocof(freq, time):
    """Calculate Rate of Change of Frequency (ROCOF)."""
    if len(freq) < 2:
        return 0.0
    dt = np.diff(time)
    df = np.diff(freq)
    dt = np.where(dt == 0, 1e-6, dt)
    rocof = df / dt
    return np.max(np.abs(rocof))


# ==============================================================================
# 4. SCENARIO MANAGEMENT (SAVE/LOAD)
# ==============================================================================

def get_current_scenario():
    """Capture current configuration as a scenario dictionary."""
    return {
        "version": "6.1",
        "timestamp": datetime.now().isoformat(),
        "source": "CAT Transient Stability Analyzer",
        "load_config": {
            "it_load_mw": p_it,
            "auxiliaries_pct": dc_aux,
            "base_load_pct": base_load_pct,
            "step_load_pct": step_req_pct,
            "gross_load_mw": p_gross_total,
            "spike_mw": step_mw,
        },
        "load_profile": {
            "name": load_profile_name,
            "duration_s": pulse_duration,
        },
        "generation": {
            "model": gen_model,
            "units": n_gens_op,
            "inertia_h": h_const,
            "droop_pct": droop_pct,
            "tau_gov": tau_gov,
            "tau_mech": tau_mech,
        },
        "bess": {
            "enabled": enable_bess,
            "power_mw": bess_cap_manual if enable_bess else 0,
            "response_ms": bess_response_ms,
            "duration_min": bess_duration_min,
        },
        "limits": {
            "freq_min_hz": nadir_limit,
            "freq_max_hz": overshoot_limit,
            "voltage_min_pu": voltage_min,
            "rocof_max_hz_s": rocof_limit,
        },
        "economics": {
            "gas_price_mmbtu": fuel_price,
            "operating_hours": op_hours,
            "capex_gen_kw": capex_gen,
            "capex_bess_kw": capex_bess_kw,
            "capex_bess_kwh": capex_bess_kwh,
            "discount_rate": discount_rate,
            "project_years": project_years,
        }
    }


def scenario_to_json(scenario):
    """Convert scenario to JSON string."""
    return json.dumps(scenario, indent=2)


def parse_size_solution_config(config_text):
    """
    Parse configuration from CAT Size Solution.
    Accepts JSON format with specific fields.
    """
    try:
        config = json.loads(config_text)
        return {
            "valid": True,
            "it_load_mw": config.get("it_load_mw", config.get("load_config", {}).get("it_load_mw", 100)),
            "auxiliaries_pct": config.get("pue_overhead_pct", config.get("load_config", {}).get("auxiliaries_pct", 15)),
            "base_load_pct": config.get("base_load_pct", config.get("load_config", {}).get("base_load_pct", 50)),
            "step_load_pct": config.get("step_load_pct", config.get("load_config", {}).get("step_load_pct", 40)),
            "generator_model": config.get("generator_model", config.get("generation", {}).get("model", "G3520FR (2.5 MW)")),
            "n_generators": config.get("n_generators", config.get("generation", {}).get("units", 50)),
            "bess_power_mw": config.get("bess_power_mw", config.get("bess", {}).get("power_mw", 0)),
            "bess_enabled": config.get("bess_enabled", config.get("bess", {}).get("enabled", True)),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ==============================================================================
# 5. PDF REPORT GENERATION
# ==============================================================================

def generate_pdf_report(results, scenario):
    """Generate comprehensive PDF report."""
    if not HAS_REPORTLAB:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           leftMargin=0.75*inch, rightMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CATTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#FFCD00'),
        spaceAfter=20,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='CATSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=HexColor('#666666'),
        alignment=TA_CENTER,
        spaceAfter=30
    ))
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#1A1A1A'),
        spaceBefore=20,
        spaceAfter=10,
        backColor=HexColor('#FFCD00'),
        borderPadding=5
    ))
    styles.add(ParagraphStyle(
        name='BodyText2',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8
    ))
    
    story = []
    
    # === COVER ===
    story.append(Spacer(1, 1*inch))
    
    # Header bar
    header_data = [['⚡ CAT TRANSIENT STABILITY REPORT']]
    header_table = Table(header_data, colWidths=[6.5*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#FFCD00')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 18),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Project info
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['CATSubtitle']))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive summary
    stability_status = "✅ STABLE" if results['is_stable'] else "❌ UNSTABLE"
    summary_data = [
        ['STABILITY STATUS', stability_status],
        ['IT Load', f"{scenario['load_config']['it_load_mw']:.0f} MW"],
        ['Generator Model', scenario['generation']['model']],
        ['Fleet Size', f"{scenario['generation']['units']} units"],
        ['BESS Power', f"{scenario['bess']['power_mw']:.1f} MW" if scenario['bess']['enabled'] else 'Disabled'],
        ['Load Profile', scenario['load_profile']['name']],
    ]
    summary_table = Table(summary_data, colWidths=[2.5*inch, 4*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#f0f0f0')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    
    story.append(PageBreak())
    
    # === SECTION 1: LOAD CONFIGURATION ===
    story.append(Paragraph("1. LOAD CONFIGURATION", styles['SectionHeader']))
    
    load_data = [
        ['Parameter', 'Value', 'Unit'],
        ['IT Load', f"{scenario['load_config']['it_load_mw']:.1f}", 'MW'],
        ['Auxiliaries', f"{scenario['load_config']['auxiliaries_pct']:.1f}", '%'],
        ['Gross Load', f"{scenario['load_config']['gross_load_mw']:.1f}", 'MW'],
        ['Base Load', f"{scenario['load_config']['base_load_pct']:.1f}", '%'],
        ['AI Spike', f"{scenario['load_config']['step_load_pct']:.1f}", '%'],
        ['Spike Magnitude', f"{scenario['load_config']['spike_mw']:.1f}", 'MW'],
        ['Load Profile', scenario['load_profile']['name'], '-'],
    ]
    load_table = Table(load_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    load_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
    ]))
    story.append(load_table)
    story.append(Spacer(1, 0.3*inch))
    
    # === SECTION 2: GENERATION FLEET ===
    story.append(Paragraph("2. GENERATION FLEET", styles['SectionHeader']))
    
    gen_data = [
        ['Parameter', 'Value'],
        ['Generator Model', scenario['generation']['model']],
        ['Operating Units', f"{scenario['generation']['units']}"],
        ['Unit Rating', f"{specs['mw']:.2f} MW"],
        ['Total Capacity', f"{scenario['generation']['units'] * specs['mw']:.1f} MW"],
        ['Inertia Constant (H)', f"{scenario['generation']['inertia_h']:.2f} s"],
        ['Droop Setting', f"{scenario['generation']['droop_pct']:.1f} %"],
        ['Governor τ', f"{scenario['generation']['tau_gov']:.2f} s"],
        ['Mechanical τ', f"{scenario['generation']['tau_mech']:.2f} s"],
        ['Ramp Rate (per unit)', f"{specs['ramp_up']:.2f} MW/s"],
        ['Step Load Capability', f"{specs['step_load_max']} %"],
    ]
    gen_table = Table(gen_data, colWidths=[3*inch, 3*inch])
    gen_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
    ]))
    story.append(gen_table)
    story.append(Spacer(1, 0.3*inch))
    
    # === SECTION 3: BESS SYSTEM ===
    story.append(Paragraph("3. BESS SYSTEM", styles['SectionHeader']))
    
    if scenario['bess']['enabled']:
        bess_data = [
            ['Parameter', 'Value'],
            ['Status', 'Enabled'],
            ['Power Rating', f"{scenario['bess']['power_mw']:.1f} MW"],
            ['Response Time', f"{scenario['bess']['response_ms']} ms"],
            ['Duration', f"{scenario['bess']['duration_min']} min"],
            ['Energy Capacity', f"{scenario['bess']['power_mw'] * scenario['bess']['duration_min'] / 60:.1f} MWh"],
        ]
    else:
        bess_data = [
            ['Parameter', 'Value'],
            ['Status', 'Disabled'],
        ]
    
    bess_table = Table(bess_data, colWidths=[3*inch, 3*inch])
    bess_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
    ]))
    story.append(bess_table)
    story.append(Spacer(1, 0.3*inch))
    
    # === SECTION 4: STABILITY RESULTS ===
    story.append(Paragraph("4. STABILITY ANALYSIS RESULTS", styles['SectionHeader']))
    
    # Stability status box
    if results['is_stable']:
        status_color = HexColor('#d4edda')
        status_text = '✅ SYSTEM STABLE - All parameters within limits'
    else:
        status_color = HexColor('#f8d7da')
        issues = []
        if not results['freq_stable']:
            issues.append('Frequency')
        if not results['voltage_stable']:
            issues.append('Voltage')
        if not results['rocof_ok']:
            issues.append('ROCOF')
        status_text = f'❌ SYSTEM UNSTABLE - Violations: {", ".join(issues)}'
    
    status_data = [[status_text]]
    status_table = Table(status_data, colWidths=[6*inch])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), status_color),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Results table
    results_data = [
        ['Metric', 'Value', 'Limit', 'Status'],
        ['Frequency Nadir', f"{results['freq_nadir']:.2f} Hz", f"≥ {scenario['limits']['freq_min_hz']:.1f} Hz", 
         '✅' if results['freq_nadir'] >= scenario['limits']['freq_min_hz'] else '❌'],
        ['Frequency Peak', f"{results['freq_peak']:.2f} Hz", f"≤ {scenario['limits']['freq_max_hz']:.1f} Hz",
         '✅' if results['freq_peak'] <= scenario['limits']['freq_max_hz'] else '❌'],
        ['Voltage Minimum', f"{results['voltage_min']:.3f} p.u.", f"≥ {scenario['limits']['voltage_min_pu']:.2f} p.u.",
         '✅' if results['voltage_min'] >= scenario['limits']['voltage_min_pu'] else '❌'],
        ['ROCOF Maximum', f"{results['rocof_max']:.2f} Hz/s", f"≤ {scenario['limits']['rocof_max_hz_s']:.1f} Hz/s",
         '✅' if results['rocof_max'] <= scenario['limits']['rocof_max_hz_s'] else '❌'],
        ['Voltage Sag (initial)', f"{results['voltage_sag_pct']:.1f} %", '< 10%',
         '✅' if results['voltage_sag_pct'] < 10 else '⚠️'],
    ]
    
    results_table = Table(results_data, colWidths=[1.8*inch, 1.5*inch, 1.5*inch, 0.8*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
    ]))
    story.append(results_table)
    
    story.append(PageBreak())
    
    # === SECTION 5: ECONOMICS ===
    story.append(Paragraph("5. ECONOMIC ANALYSIS", styles['SectionHeader']))
    
    lcoe, capex_m = calculate_lcoe_for_report(scenario)
    
    econ_data = [
        ['Metric', 'Value'],
        ['Total CAPEX', f"${capex_m:.1f} M"],
        ['LCOE', f"${lcoe:.4f} /kWh"],
        ['Gas Price', f"${scenario['economics']['gas_price_mmbtu']:.2f} /MMBtu"],
        ['Operating Hours', f"{scenario['economics']['operating_hours']:,} hrs/yr"],
        ['Discount Rate', f"{scenario['economics']['discount_rate']*100:.1f} %"],
        ['Project Life', f"{scenario['economics']['project_years']} years"],
    ]
    
    econ_table = Table(econ_data, colWidths=[3*inch, 3*inch])
    econ_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1A1A1A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f8f8f8')]),
    ]))
    story.append(econ_table)
    story.append(Spacer(1, 0.5*inch))
    
    # === DISCLAIMER ===
    story.append(Paragraph("DISCLAIMER", styles['SectionHeader']))
    disclaimer = """
    This report is generated by CAT Transient Stability Analyzer for preliminary analysis purposes only.
    Results are based on simplified models and should be validated with detailed engineering studies
    before final design decisions. Actual system performance may vary based on site conditions,
    equipment specifications, and installation quality. Caterpillar Inc. makes no warranties
    regarding the accuracy or completeness of this analysis.
    """
    story.append(Paragraph(disclaimer, styles['BodyText2']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def calculate_lcoe_for_report(scenario):
    """Calculate LCOE for PDF report."""
    gen_specs = CAT_LIBRARY.get(scenario['generation']['model'], list(CAT_LIBRARY.values())[0])
    n_gens = scenario['generation']['units']
    bess_mw = scenario['bess']['power_mw'] if scenario['bess']['enabled'] else 0
    
    gen_cap_mw = n_gens * gen_specs['mw']
    capex_gens = gen_cap_mw * 1000 * scenario['economics']['capex_gen_kw']
    
    bess_energy_mwh = bess_mw * (scenario['bess']['duration_min'] / 60.0)
    capex_bess = (bess_mw * 1000 * scenario['economics']['capex_bess_kw']) + \
                 (bess_energy_mwh * 1000 * scenario['economics']['capex_bess_kwh'])
    
    total_capex = capex_gens + capex_bess
    
    dr = scenario['economics']['discount_rate']
    years = scenario['economics']['project_years']
    crf = (dr * (1 + dr)**years) / ((1 + dr)**years - 1)
    capex_annual = total_capex * crf
    
    P_base = scenario['load_config']['gross_load_mw'] * (scenario['load_config']['base_load_pct'] / 100.0)
    hr = gen_specs['hr']
    fuel_annual = P_base * hr * scenario['economics']['gas_price_mmbtu'] * scenario['economics']['operating_hours']
    
    om_annual = total_capex * 0.025
    total_mwh = P_base * scenario['economics']['operating_hours']
    
    lcoe = (capex_annual + fuel_annual + om_annual) / max(1, total_mwh) / 1000
    
    return lcoe, total_capex / 1e6


# ==============================================================================
# 6. INPUT SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown("## ⚡ CAT Transient Analyzer")
    st.caption("v6.1 - With Save/Load & PDF Export")
    
    # === IMPORT FROM SIZE SOLUTION ===
    with st.expander("🔗 Import from Size Solution", expanded=False):
        st.markdown("**Paste JSON from CAT Size Solution:**")
        import_text = st.text_area("Configuration JSON", height=100, 
                                   placeholder='{"it_load_mw": 100, "n_generators": 50, ...}')
        
        if st.button("📥 Import Configuration", use_container_width=True):
            if import_text.strip():
                parsed = parse_size_solution_config(import_text)
                if parsed['valid']:
                    st.session_state['imported_config'] = parsed
                    st.success("✅ Configuration imported!")
                else:
                    st.error(f"❌ Invalid JSON: {parsed['error']}")
        
        # Show imported config status
        if 'imported_config' in st.session_state:
            cfg = st.session_state['imported_config']
            st.markdown(f"""
            <div class="cat-box-import">
            <b>Imported:</b><br>
            • IT Load: {cfg['it_load_mw']} MW<br>
            • Generators: {cfg['n_generators']} units<br>
            • BESS: {cfg['bess_power_mw']:.1f} MW
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Apply Imported Values"):
                st.session_state['apply_import'] = True
                st.rerun()
    
    # Get default values (from import if available)
    defaults = st.session_state.get('imported_config', {})
    
    # --- LOAD CONFIGURATION ---
    with st.expander("📊 1. Load Configuration", expanded=True):
        p_it = st.number_input("IT Load (MW)", 1.0, 5000.0, 
                               float(defaults.get('it_load_mw', 100.0)), step=10.0)
        dc_aux = st.number_input("Auxiliaries (%)", 0.0, 50.0, 
                                 float(defaults.get('auxiliaries_pct', 15.0)), step=1.0)
        base_load_pct = st.number_input("Base Load (%)", 10.0, 100.0, 
                                        float(defaults.get('base_load_pct', 50.0)), step=5.0)
        step_req_pct = st.number_input("AI Load Spike (%)", 0.0, 100.0, 
                                       float(defaults.get('step_load_pct', 40.0)), step=5.0)
        
        p_gross_total = p_it * (1 + dc_aux / 100.0)
        step_mw = p_it * (step_req_pct / 100.0)
        
        st.caption(f"📌 Gross Load: {p_gross_total:.1f} MW | Spike: {step_mw:.1f} MW")
    
    # --- LOAD PROFILE ---
    with st.expander("📈 2. Load Profile (AI Workload)", expanded=True):
        load_profile_name = st.selectbox(
            "Workload Pattern",
            list(LOAD_PROFILES.keys()),
            help="Select AI workload pattern to simulate"
        )
        
        profile_info = LOAD_PROFILES[load_profile_name]
        st.caption(f"ℹ️ {profile_info['description']}")
        
        if profile_info.get('configurable_duration', False):
            pulse_duration = st.number_input(
                "Event Duration (s)", 
                0.5, 120.0, 
                float(profile_info.get('default_duration', 5.0)),
                step=0.5
            )
        else:
            pulse_duration = profile_info.get('default_duration', 9999.0)
            if load_profile_name == "GPU Cluster Staircase":
                st.info(f"⏱️ Total ramp time: {profile_info['steps'] * profile_info['step_interval']:.0f}s")
    
    # --- GENERATION FLEET ---
    with st.expander("🔧 3. Generation Fleet", expanded=True):
        # Find matching model from import
        default_model = defaults.get('generator_model', "G3520FR (2.5 MW)")
        model_list = list(CAT_LIBRARY.keys())
        default_idx = model_list.index(default_model) if default_model in model_list else 1
        
        gen_model = st.selectbox("Generator Model", model_list, index=default_idx)
        specs = CAT_LIBRARY[gen_model]
        
        st.caption(f"Type: {specs['type']} | Rating: {specs['mw']} MW")
        
        n_rec = int(np.ceil(p_gross_total / specs['mw'])) + 1
        default_n = defaults.get('n_generators', n_rec)
        n_gens_op = st.number_input("Operating Units (N)", 1, 500, int(default_n))
        
        with st.container():
            st.markdown("**Advanced Parameters:**")
            col1, col2 = st.columns(2)
            with col1:
                h_const = st.number_input("Inertia H (s)", 0.1, 20.0, float(specs['h_def']), step=0.1)
                droop_pct = st.number_input("Droop (%)", 1.0, 10.0, float(specs['droop']), step=0.5)
            with col2:
                tau_gov = st.number_input("τ_gov (s)", 0.05, 5.0, float(specs['tau_gov']), step=0.05)
                tau_mech = st.number_input("τ_mech (s)", 0.1, 5.0, float(specs['tau_mech']), step=0.1)
        
        ramp_up_total = specs['ramp_up'] * n_gens_op
        st.caption(f"🔺 Fleet Ramp Rate: {ramp_up_total:.1f} MW/s")
    
    # --- BESS CONFIGURATION ---
    with st.expander("🔋 4. BESS Configuration", expanded=True):
        default_bess_enabled = defaults.get('bess_enabled', True)
        enable_bess = st.checkbox("Enable BESS", value=default_bess_enabled)
        
        if enable_bess:
            auto_size_bess = st.checkbox("Auto-Size (Match Spike)", value=True)
            
            if auto_size_bess:
                bess_cap_manual = step_mw
                st.info(f"📐 Auto-sized: {bess_cap_manual:.1f} MW")
            else:
                default_bess = defaults.get('bess_power_mw', step_mw)
                bess_cap_manual = st.number_input("BESS Power (MW)", 0.0, 5000.0, float(default_bess), step=5.0)
            
            bess_response_ms = st.number_input("Response Time (ms)", 10, 2000, 50, step=10)
            bess_duration_min = st.number_input("Duration (minutes)", 5, 240, 30, step=5)
            
            bess_energy_mwh = bess_cap_manual * (bess_duration_min / 60.0)
            st.caption(f"⚡ Energy: {bess_energy_mwh:.1f} MWh")
        else:
            bess_cap_manual = 0.0
            bess_response_ms = 1000
            bess_duration_min = 30
    
    # --- STABILITY LIMITS ---
    with st.expander("⚠️ 5. Stability Limits", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            nadir_limit = st.number_input("Min Freq (Hz)", 55.0, 59.9, 57.0, step=0.1)
            voltage_min = st.number_input("Min Voltage (p.u.)", 0.80, 0.99, 0.90, step=0.01)
        with col2:
            overshoot_limit = st.number_input("Max Freq (Hz)", 60.1, 65.0, 63.0, step=0.1)
            rocof_limit = st.number_input("Max ROCOF (Hz/s)", 0.5, 5.0, 2.0, step=0.1)
    
    # --- ECONOMICS ---
    with st.expander("💰 6. Economics", expanded=False):
        fuel_price = st.number_input("Gas Price ($/MMBtu)", 1.0, 20.0, 6.0, step=0.5)
        op_hours = st.number_input("Operating Hours/Year", 1000, 8760, 7000, step=500)
        capex_gen = st.number_input("Gen CAPEX ($/kW)", 100, 2000, specs['capex'], step=50)
        capex_bess_kw = st.number_input("BESS Power ($/kW)", 50, 500, 150, step=10)
        capex_bess_kwh = st.number_input("BESS Energy ($/kWh)", 100, 500, 250, step=10)
        discount_rate = st.number_input("Discount Rate (%)", 1.0, 20.0, 8.0, step=0.5) / 100
        project_years = st.number_input("Project Life (years)", 5, 30, 20, step=1)
    
    # --- SAVE/LOAD SCENARIOS ---
    with st.expander("💾 7. Save/Load Scenario", expanded=False):
        scenario_name = st.text_input("Scenario Name", "My_Scenario")
        
        col_save, col_load = st.columns(2)
        
        with col_save:
            if st.button("💾 Save", use_container_width=True):
                scenario = get_current_scenario()
                scenario['name'] = scenario_name
                json_str = scenario_to_json(scenario)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    "application/json",
                    use_container_width=True
                )
        
        with col_load:
            uploaded_file = st.file_uploader("Load", type=['json'], label_visibility='collapsed')
            if uploaded_file is not None:
                try:
                    loaded = json.load(uploaded_file)
                    st.session_state['imported_config'] = {
                        'valid': True,
                        'it_load_mw': loaded.get('load_config', {}).get('it_load_mw', 100),
                        'auxiliaries_pct': loaded.get('load_config', {}).get('auxiliaries_pct', 15),
                        'base_load_pct': loaded.get('load_config', {}).get('base_load_pct', 50),
                        'step_load_pct': loaded.get('load_config', {}).get('step_load_pct', 40),
                        'generator_model': loaded.get('generation', {}).get('model', 'G3520FR (2.5 MW)'),
                        'n_generators': loaded.get('generation', {}).get('units', 50),
                        'bess_power_mw': loaded.get('bess', {}).get('power_mw', 0),
                        'bess_enabled': loaded.get('bess', {}).get('enabled', True),
                    }
                    st.success("✅ Loaded! Click 'Apply' above.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ==============================================================================
# 7. SIMULATION FUNCTIONS
# ==============================================================================

def run_simulation(n_gens, bess_mw, specs, sim_time=20.0):
    """Run transient stability simulation."""
    gen_cap_total = n_gens * specs['mw']
    S_base = gen_cap_total / 0.8
    P_base = p_gross_total * (base_load_pct / 100.0)
    P_max = gen_cap_total
    
    dt = 0.01
    t = np.arange(0, sim_time, dt)
    
    load_params = {
        'P_spike': step_mw,
        't_start': 1.0,
        'duration': pulse_duration,
    }
    load_profile = create_load_profile(load_profile_name, t, load_params)
    
    sim_params = {
        'H': h_const * n_gens,
        'D': 1.0,
        'S_base': S_base,
        'P_base': P_base,
        'P_max': P_max,
        'droop': droop_pct,
        'T_gov': tau_gov,
        'T_mech': tau_mech,
        'ramp_up': specs['ramp_up'] * n_gens,
        'ramp_down': specs['ramp_down'] * n_gens,
        'P_bess_max': bess_mw,
        'T_bess': bess_response_ms / 1000.0,
        'bess_enabled': enable_bess and bess_mw > 0,
        'X_eq': specs['xd_pp'] / np.sqrt(max(1, n_gens)),
        'alpha_v': 0.5,
        't_profile': t,
        'load_profile': load_profile,
    }
    
    y0 = [0.0, P_base, P_base, 0.0, 1.0]
    
    try:
        sol = odeint(system_dynamics_v2, y0, t, args=(sim_params,))
        
        freq = 60.0 + sol[:, 0]
        p_mech = sol[:, 1]
        p_gov = sol[:, 2]
        p_bess = sol[:, 3]
        v_pu = sol[:, 4]
        p_load = P_base + load_profile
        
        freq_nadir = np.min(freq)
        freq_peak = np.max(freq)
        voltage_min_result = np.min(v_pu)
        rocof_max = calculate_rocof(freq, t)
        
        freq_stable = (freq_nadir >= nadir_limit) and (freq_peak <= overshoot_limit)
        voltage_stable = voltage_min_result >= voltage_min
        rocof_ok = rocof_max <= rocof_limit
        
        voltage_sag_pct = calculate_voltage_sag(step_mw, specs['xd_pp'], n_gens, S_base)
        
        is_stable = freq_stable and voltage_stable and rocof_ok

        # --- VALIDACIÓN MECÁNICA (CORREGIDA CON BESS) ---
        # 1. Calculamos el golpe NETO que realmente ven los generadores
        # Si el BESS cubre todo el golpe, el motor ve 0 MW de impacto instantáneo.
        step_mw_net = max(0, step_mw - bess_mw)
        
        # 2. Repartimos ese golpe neto entre las máquinas
        step_mw_per_unit = step_mw_net / n_gens
        step_pct_actual = (step_mw_per_unit / specs['mw']) * 100
        step_load_limit = specs['step_load_max']
        
        # 3. Verificamos si VIOLA el límite mecánico
        mechanical_overload = step_pct_actual > step_load_limit
        
        if mechanical_overload:
            is_stable = False
            # Mensaje detallado
            fail_reason = f"Mechanical Step Limit Exceeded ({step_pct_actual:.1f}% > {step_load_limit}%) - BESS insufficient"
        else:
            fail_reason = None
        # ---------------------------------
        
        return {
            'success': True,
            't': t,
            'freq': freq,
            'p_mech': p_mech,
            'p_gov': p_gov,
            'p_bess': p_bess,
            'v_pu': v_pu,
            'p_load': p_load,
            'load_profile': load_profile,
            'freq_nadir': freq_nadir,
            'freq_peak': freq_peak,
            'voltage_min': voltage_min_result,
            'voltage_sag_pct': voltage_sag_pct,
            'rocof_max': rocof_max,
            'is_stable': is_stable,
            'freq_stable': freq_stable,
            'voltage_stable': voltage_stable,
            'rocof_ok': rocof_ok,
            'P_base': P_base,
            'mechanical_overload': mechanical_overload,
            'fail_reason': fail_reason,
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'is_stable': False,
        }


def calculate_lcoe(n_gens, bess_mw, is_stable):
    """Calculate Levelized Cost of Energy."""
    gen_cap_mw = n_gens * specs['mw']
    
    capex_gens = gen_cap_mw * 1000 * capex_gen
    bess_energy_mwh = bess_mw * (bess_duration_min / 60.0)
    capex_bess = (bess_mw * 1000 * capex_bess_kw) + (bess_energy_mwh * 1000 * capex_bess_kwh)
    total_capex = capex_gens + capex_bess
    
    crf = (discount_rate * (1 + discount_rate)**project_years) / ((1 + discount_rate)**project_years - 1)
    capex_annual = total_capex * crf
    
    P_base = p_gross_total * (base_load_pct / 100.0)
    load_factor = P_base / max(0.1, gen_cap_mw)
    
    if load_factor >= 0.75:
        hr_factor = 1.0
    elif load_factor >= 0.50:
        hr_factor = 1.05
    elif load_factor >= 0.25:
        hr_factor = 1.15
    else:
        hr_factor = 1.30
    
    hr_actual = specs['hr'] * hr_factor
    fuel_annual = P_base * hr_actual * fuel_price * op_hours
    
    om_annual = total_capex * 0.025
    total_mwh = P_base * op_hours
    
    lcoe_mwh = (capex_annual + fuel_annual + om_annual) / max(1, total_mwh)
    lcoe_kwh = lcoe_mwh / 1000.0
    
    if not is_stable:
        lcoe_kwh += 1.0
    
    return lcoe_kwh, total_capex / 1e6


# ==============================================================================
# 8. MAIN UI
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1>⚡ CAT Transient Stability Analyzer</h1>
    <p>Advanced simulation of frequency & voltage response for AI Data Center power systems</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Simulation", "🎯 Optimizer", "📊 Comparison"])

# ==============================================================================
# TAB 1: SINGLE SIMULATION
# ==============================================================================
with tab1:
    col_btn, col_info = st.columns([1, 3])
    
    with col_btn:
        run_sim = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
    
    with col_info:
        st.info(f"**Config:** {n_gens_op}× {gen_model} | BESS: {bess_cap_manual:.1f} MW | Profile: {load_profile_name}")
    
    if run_sim:
        with st.spinner("Running simulation..."):
            results = run_simulation(n_gens_op, bess_cap_manual if enable_bess else 0, specs)
        
        if results['success']:
            # Store results for PDF
            st.session_state['last_results'] = results
            st.session_state['last_scenario'] = get_current_scenario()
            
            col_plot, col_metrics = st.columns([2.5, 1])
            
            with col_plot:
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=("Frequency Response", "Power Balance", "Voltage")
                )
                
                t = results['t']
                
                fig.add_trace(
                    go.Scatter(x=t, y=results['freq'], name='Frequency', 
                              line=dict(color='#1f77b4', width=2)),
                    row=1, col=1
                )
                fig.add_hline(y=nadir_limit, line_dash="dash", line_color="red", 
                             annotation_text=f"Min: {nadir_limit} Hz", row=1, col=1)
                fig.add_hline(y=overshoot_limit, line_dash="dash", line_color="orange",
                             annotation_text=f"Max: {overshoot_limit} Hz", row=1, col=1)
                fig.add_hline(y=60.0, line_dash="dot", line_color="gray", row=1, col=1)
                
                fig.add_trace(
                    go.Scatter(x=t, y=results['p_load'], name='Load', 
                              line=dict(color='black', width=2, dash='dash')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=t, y=results['p_mech'], name='Generators', 
                              line=dict(color='#2ca02c', width=2)),
                    row=2, col=1
                )
                if enable_bess and bess_cap_manual > 0:
                    fig.add_trace(
                        go.Scatter(x=t, y=results['p_bess'], name='BESS', 
                                  line=dict(color='#9467bd', width=2)),
                        row=2, col=1
                    )
                
                fig.add_trace(
                    go.Scatter(x=t, y=results['v_pu'], name='Voltage', 
                              line=dict(color='#d62728', width=2)),
                    row=3, col=1
                )
                fig.add_hline(y=voltage_min, line_dash="dash", line_color="red",
                             annotation_text=f"Min: {voltage_min} p.u.", row=3, col=1)
                fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=3, col=1)
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=60, r=20, t=80, b=40),
                )
                
                fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
                fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
                fig.update_yaxes(title_text="Voltage (p.u.)", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_metrics:
                st.markdown("### 📊 Results")
                
                if results['is_stable']:
                    st.markdown('<div class="cat-box-success">✅ <b>SYSTEM STABLE</b></div>', 
                               unsafe_allow_html=True)
                else:
                    issues = []
                    if not results['freq_stable']:
                        issues.append("Frequency")
                    if not results['voltage_stable']:
                        issues.append("Voltage")
                    if not results['rocof_ok']:
                        issues.append("ROCOF")
                    
                    st.markdown(f'<div class="cat-box-fail">❌ <b>UNSTABLE</b><br>{", ".join(issues)} violation</div>', 
                               unsafe_allow_html=True)
                
                st.markdown("---")
                
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    delta_nadir = results['freq_nadir'] - 60.0
                    st.metric("Freq Nadir", f"{results['freq_nadir']:.2f} Hz", 
                             f"{delta_nadir:.2f} Hz")
                    
                    st.metric("Voltage Min", f"{results['voltage_min']:.3f} p.u.",
                             f"{(results['voltage_min']-1)*100:.1f}%")
                
                with col_m2:
                    delta_peak = results['freq_peak'] - 60.0
                    st.metric("Freq Peak", f"{results['freq_peak']:.2f} Hz",
                             f"+{delta_peak:.2f} Hz")
                    
                    st.metric("ROCOF Max", f"{results['rocof_max']:.2f} Hz/s",
                             "OK" if results['rocof_ok'] else "⚠️ HIGH")
                
                st.markdown("---")
                
                st.metric("Initial Voltage Sag", f"{results['voltage_sag_pct']:.1f}%",
                         "Instantaneous")
                
                lcoe, capex_m = calculate_lcoe(n_gens_op, bess_cap_manual if enable_bess else 0, 
                                               results['is_stable'])
                st.metric("LCOE", f"${lcoe:.4f}/kWh")
                st.metric("CAPEX", f"${capex_m:.1f}M")
                
                # === PDF EXPORT BUTTON ===
                st.markdown("---")
                st.markdown("### 📄 Export Report")
                
                if HAS_REPORTLAB:
                    if st.button("📥 Generate PDF Report", type="secondary", use_container_width=True):
                        with st.spinner("Generating PDF..."):
                            scenario = get_current_scenario()
                            pdf_data = generate_pdf_report(results, scenario)
                            
                            if pdf_data:
                                st.download_button(
                                    "📄 Download PDF",
                                    pdf_data,
                                    f"Transient_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    "application/pdf",
                                    use_container_width=True
                                )
                else:
                    st.warning("PDF export requires reportlab. Add to requirements.txt")
                
                # Tips
                if not results['is_stable']:
                    st.markdown("---")
                    st.markdown("**💡 Recommendations:**")
                    
                    if not results['freq_stable']:
                        if not enable_bess:
                            st.info("→ Enable BESS to absorb load spikes")
                        else:
                            st.info("→ Increase BESS power or add generators")
                    
                    if not results['voltage_stable']:
                        st.info("→ Add generators or reduce step load")
                    
                    if not results['rocof_ok']:
                        st.info("→ Increase system inertia")
        
        else:
            st.error(f"Simulation failed: {results.get('error', 'Unknown error')}")


# ==============================================================================
# TAB 2: OPTIMIZER
# ==============================================================================
with tab2:
    st.markdown("### 🎯 Find Optimal Generator + BESS Configuration")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        opt_objective = st.radio(
            "Optimization Objective",
            ["Minimum LCOE", "Minimum CAPEX", "Minimum Generators"],
            horizontal=True
        )
    
    with col_opt2:
        opt_bess_range = st.slider("BESS Search Range (% of spike)", 0, 150, (0, 120), step=10)
    
    if st.button("🔎 Run Optimization", type="primary"):
        progress_bar = st.progress(0, text="Initializing...")
        
        mw_base = p_gross_total * (base_load_pct / 100.0)
        mw_peak = p_gross_total * ((base_load_pct + step_req_pct) / 100.0)
        
        n_min = max(1, int(np.ceil(mw_base / specs['mw'])))
        n_max = int(np.ceil(mw_peak / specs['mw']) * 1.5) + 2
        n_range = range(n_min, n_max + 1)
        
        bess_min = step_mw * (opt_bess_range[0] / 100.0)
        bess_max = step_mw * (opt_bess_range[1] / 100.0)
        bess_steps = 8
        b_range = np.linspace(bess_min, bess_max, bess_steps)
        
        results_list = []
        total_iters = len(n_range) * len(b_range)
        iter_count = 0
        
        for n in n_range:
            for b_mw in b_range:
                sim_result = run_simulation(n, b_mw, specs, sim_time=15.0)
                
                if sim_result['success']:
                    lcoe, capex_m = calculate_lcoe(n, b_mw, sim_result['is_stable'])
                    
                    results_list.append({
                        'Gens': n,
                        'BESS_MW': b_mw,
                        'Stable': sim_result['is_stable'],
                        'Freq_Nadir': sim_result['freq_nadir'],
                        'Voltage_Min': sim_result['voltage_min'],
                        'ROCOF': sim_result['rocof_max'],
                        'LCOE': lcoe,
                        'CAPEX_M': capex_m,
                    })
                
                iter_count += 1
                progress_bar.progress(iter_count / total_iters, 
                                     text=f"Testing {n} gens + {b_mw:.1f} MW BESS...")
        
        progress_bar.empty()
        
        df_opt = pd.DataFrame(results_list)
        df_stable = df_opt[df_opt['Stable'] == True]
        
        if not df_stable.empty:
            if opt_objective == "Minimum LCOE":
                best = df_stable.loc[df_stable['LCOE'].idxmin()]
            elif opt_objective == "Minimum CAPEX":
                best = df_stable.loc[df_stable['CAPEX_M'].idxmin()]
            else:
                best = df_stable.loc[df_stable['Gens'].idxmin()]
            
            st.markdown("### ✅ Optimal Configuration Found")
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                st.markdown(f"""
                <div class="cat-box-opt">
                    <div style="font-size:12px">GENERATORS</div>
                    <div style="font-size:24px; font-weight:bold">{int(best['Gens'])} Units</div>
                    <div style="font-size:11px">{specs['mw']} MW each</div>
                </div>""", unsafe_allow_html=True)
            
            with col_r2:
                st.markdown(f"""
                <div class="cat-box-opt">
                    <div style="font-size:12px">BESS POWER</div>
                    <div style="font-size:24px; font-weight:bold">{best['BESS_MW']:.1f} MW</div>
                    <div style="font-size:11px">{best['BESS_MW']/step_mw*100:.0f}% of spike</div>
                </div>""", unsafe_allow_html=True)
            
            with col_r3:
                st.markdown(f"""
                <div class="cat-box-success">
                    <div style="font-size:12px">LCOE</div>
                    <div style="font-size:24px; font-weight:bold">${best['LCOE']:.4f}</div>
                    <div style="font-size:11px">per kWh</div>
                </div>""", unsafe_allow_html=True)
            
            with col_r4:
                st.markdown(f"""
                <div class="cat-box-opt">
                    <div style="font-size:12px">CAPEX</div>
                    <div style="font-size:24px; font-weight:bold">${best['CAPEX_M']:.1f}M</div>
                    <div style="font-size:11px">Total investment</div>
                </div>""", unsafe_allow_html=True)
            
            df_gen_only = df_stable[df_stable['BESS_MW'] == 0]
            
            if not df_gen_only.empty:
                gen_only_best = df_gen_only.loc[df_gen_only['LCOE'].idxmin()]
                lcoe_savings = gen_only_best['LCOE'] - best['LCOE']
                gen_savings = gen_only_best['Gens'] - best['Gens']
                
                st.markdown("---")
                col_cmp1, col_cmp2 = st.columns(2)
                
                with col_cmp1:
                    st.info(f"**Gen-Only:** {int(gen_only_best['Gens'])} units @ ${gen_only_best['LCOE']:.4f}/kWh")
                
                with col_cmp2:
                    if lcoe_savings > 0:
                        st.success(f"**Hybrid Savings:** ${lcoe_savings:.4f}/kWh | {int(gen_savings)} fewer generators")
                    else:
                        st.warning("Gen-only solution is more economical")
            else:
                st.warning("⚠️ No stable gen-only configuration found. BESS is required.")
            
            st.markdown("### 🗺️ Optimization Landscape")
            
            fig_opt = go.Figure()
            
            unstable = df_opt[~df_opt['Stable']]
            fig_opt.add_trace(go.Scatter(
                x=unstable['Gens'], y=unstable['BESS_MW'],
                mode='markers',
                marker=dict(symbol='x', size=8, color='lightgray'),
                name='Unstable'
            ))
            
            fig_opt.add_trace(go.Scatter(
                x=df_stable['Gens'], y=df_stable['BESS_MW'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=df_stable['LCOE'],
                    colorscale='Viridis_r',
                    showscale=True,
                    colorbar=dict(title='LCOE ($/kWh)')
                ),
                name='Stable',
                text=[f"LCOE: ${l:.4f}" for l in df_stable['LCOE']],
                hoverinfo='text+x+y'
            ))
            
            fig_opt.add_trace(go.Scatter(
                x=[best['Gens']], y=[best['BESS_MW']],
                mode='markers',
                marker=dict(symbol='star', size=20, color='red', line=dict(width=2, color='white')),
                name='Optimal'
            ))
            
            fig_opt.update_layout(
                xaxis_title="Number of Generators",
                yaxis_title="BESS Capacity (MW)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig_opt, use_container_width=True)
            
            with st.expander("📋 All Results"):
                st.dataframe(df_opt.sort_values('LCOE'), use_container_width=True)
        
        else:
            st.error("❌ No stable configuration found.")


# ==============================================================================
# TAB 3: COMPARISON
# ==============================================================================
with tab3:
    st.markdown("### 📊 Technology Comparison")
    
    models_to_compare = st.multiselect(
        "Select Generator Models",
        list(CAT_LIBRARY.keys()),
        default=list(CAT_LIBRARY.keys())[:3]
    )
    
    if st.button("📈 Run Comparison", type="primary"):
        comparison_results = []
        
        for model_name in models_to_compare:
            model_specs = CAT_LIBRARY[model_name]
            n_gens = int(np.ceil(p_gross_total / model_specs['mw'])) + 1
            
            for bess_enabled_cmp in [False, True]:
                bess_mw_cmp = step_mw if bess_enabled_cmp else 0
                
                sim_result = run_simulation(n_gens, bess_mw_cmp, model_specs, sim_time=15.0)
                
                if sim_result['success']:
                    lcoe, capex_m = calculate_lcoe(n_gens, bess_mw_cmp, sim_result['is_stable'])
                    
                    comparison_results.append({
                        'Model': model_name,
                        'Type': model_specs['type'],
                        'Configuration': 'Hybrid' if bess_enabled_cmp else 'Gen-Only',
                        'Generators': n_gens,
                        'BESS (MW)': bess_mw_cmp,
                        'Installed (MW)': n_gens * model_specs['mw'],
                        'Stable': '✅' if sim_result['is_stable'] else '❌',
                        'Freq Nadir (Hz)': sim_result['freq_nadir'],
                        'Voltage Min (p.u.)': sim_result['voltage_min'],
                        'LCOE ($/kWh)': lcoe,
                        'CAPEX ($M)': capex_m,
                    })
        
        if comparison_results:
            df_comparison = pd.DataFrame(comparison_results)
            
            st.dataframe(
                df_comparison.style.format({
                    'Freq Nadir (Hz)': '{:.2f}',
                    'Voltage Min (p.u.)': '{:.3f}',
                    'LCOE ($/kWh)': '${:.4f}',
                    'CAPEX ($M)': '${:.1f}',
                    'BESS (MW)': '{:.1f}',
                    'Installed (MW)': '{:.1f}',
                }),
                use_container_width=True
            )
            
            st.markdown("#### LCOE Comparison")
            
            df_stable_cmp = df_comparison[df_comparison['Stable'] == '✅']
            
            if not df_stable_cmp.empty:
                fig_cmp = go.Figure()
                
                for config_type in ['Gen-Only', 'Hybrid']:
                    df_config = df_stable_cmp[df_stable_cmp['Configuration'] == config_type]
                    
                    fig_cmp.add_trace(go.Bar(
                        x=df_config['Model'],
                        y=df_config['LCOE ($/kWh)'],
                        name=config_type,
                        text=[f"${x:.4f}" for x in df_config['LCOE ($/kWh)']],
                        textposition='outside'
                    ))
                
                fig_cmp.update_layout(
                    barmode='group',
                    yaxis_title='LCOE ($/kWh)',
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                
                st.plotly_chart(fig_cmp, use_container_width=True)
            else:
                st.warning("No stable configurations found")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>⚡ CAT Transient Stability Analyzer v6.1</b></p>
    <p>Enhanced with PDF Export, Save/Load Scenarios, and Size Solution Integration</p>
    <p>Caterpillar Electric Power | 2026</p>
</div>
""", unsafe_allow_html=True)
