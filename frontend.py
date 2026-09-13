import streamlit as st
import requests
import base64
import json
import os
import time
import numpy as np
import pandas as pd
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def render_html(html_str: str):
    """Elimina los espacios iniciales de cada línea para evitar que el parser Markdown de Streamlit lo trate como bloque de código."""
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y METADATOS
# ==============================================================================
st.set_page_config(
    page_title="ALTUR | Voice AI Defense",
    page_icon="https://img.icons8.com/ios-filled/50/6366f1/shield.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Cargar umbral si existe
threshold_val = 0.50
if os.path.exists("threshold.json"):
    try:
        with open("threshold.json") as f:
            threshold_val = json.loads(f.read().lstrip("\ufeff")).get("threshold", 0.50)
    except Exception:
        pass

# ==============================================================================
# ESTILOS CYBER-BANKING DARK THEME (RÉPLICA EXACTA)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Variables y Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #080C14 !important;
        color: #F3F4F6;
    }

    code, pre, .stCodeBlock, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu, header, footer {visibility: hidden;}

    /* Barra de Navegación Superior */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 20px;
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-icon-box {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        border-radius: 12px;
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }

    .brand-title-group {
        display: flex;
        flex-direction: column;
    }

    .brand-title-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-name {
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }

    .brand-tag {
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #64748B;
        margin-top: 2px;
    }

    .nav-status-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .status-pill {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 7px 14px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #94A3B8;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    .docs-btn {
        background: #1E1B4B;
        border: 1px solid #4338CA;
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12px;
        font-weight: 600;
        color: #C7D2FE !important;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease;
    }

    .docs-btn:hover {
        background: #312E81;
        border-color: #6366F1;
    }

    /* Grid de 4 Modelos Superiores */
    .models-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }

    @media (max-width: 900px) {
        .models-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    .model-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .model-icon {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .icon-purple { background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); color: #818CF8; }
    .icon-violet { background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); color: #A78BFA; }
    .icon-blue   { background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #60A5FA; }
    .icon-teal   { background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.3); color: #2DD4BF; }

    .model-info-label {
        font-size: 10.5px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.04em;
    }

    .model-info-value {
        font-size: 12.5px;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 2px;
    }

    /* Paneles de Contenido */
    .cyber-panel {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }

    .panel-title {
        font-size: 12.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #F1F5F9;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Zona de Carga / Drop Zone */
    .drop-zone {
        border: 1.5px dashed #2A384C;
        background: rgba(11, 15, 25, 0.6);
        border-radius: 10px;
        padding: 30px 16px;
        text-align: center;
        margin-bottom: 12px;
    }

    .cloud-icon-box {
        width: 44px;
        height: 44px;
        background: #1E293B;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 12px auto;
        color: #818CF8;
    }

    .drop-zone-title {
        font-size: 13.5px;
        font-weight: 600;
        color: #E2E8F0;
    }

    .drop-zone-link {
        color: #6366F1;
        text-decoration: underline;
        cursor: pointer;
    }

    .drop-zone-sub {
        font-size: 11.5px;
        color: #64748B;
        margin-top: 4px;
    }

    /* Preview de Audio */
    .waveform-box {
        background: #080C14;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 20px;
        min-height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #475569;
        font-size: 12.5px;
        margin-bottom: 14px;
    }

    .audio-bottom-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .audio-playback-info {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 12px;
        color: #94A3B8;
    }

    .play-btn-circle {
        width: 32px;
        height: 32px;
        background: #1E293B;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94A3B8;
    }

    /* Botón Run Detection */
    .stButton > button {
        background: linear-gradient(135deg, #4338CA 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #4F46E5 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 12.5px !important;
        letter-spacing: 0.05em !important;
        padding: 9px 20px !important;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        box-shadow: 0 0 24px rgba(99, 102, 241, 0.6) !important;
        transform: translateY(-1px);
    }

    /* Panel de Veredicto */
    .verdict-idle-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 50px 20px;
    }

    .verdict-shield-icon {
        width: 58px;
        height: 58px;
        background: #111B2E;
        border: 1px solid #1E293B;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6366F1;
        margin-bottom: 16px;
    }

    .verdict-idle-title {
        font-size: 15px;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 6px;
    }

    .verdict-idle-sub {
        font-size: 12px;
        color: #64748B;
        max-width: 290px;
        line-height: 1.5;
    }

    .card-footer-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: #64748B;
        padding-top: 14px;
        border-top: 1px solid #1E293B;
        margin-top: 20px;
    }

    /* Styling Streamlit's native file uploader to match dark cyber-banking design */
    [data-testid="stFileUploader"] {
        width: 100%;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(11, 15, 25, 0.7) !important;
        border: 1.5px dashed #2A384C !important;
        border-radius: 10px !important;
        padding: 20px 16px !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #6366F1 !important;
        background: rgba(99, 102, 241, 0.06) !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #1E1B4B !important;
        border: 1px solid #4338CA !important;
        color: #C7D2FE !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 6px 16px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #312E81 !important;
        border-color: #6366F1 !important;
        color: #FFFFFF !important;
        transform: none !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #94A3B8 !important;
        font-size: 12.5px !important;
        margin-bottom: 6px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #64748B !important;
        font-size: 11px !important;
    }

    /* Pestañas Estilizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        padding: 0;
        border: none;
        margin-bottom: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 32px;
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        color: #94A3B8;
        font-size: 11.5px;
        font-weight: 600;
        padding: 0 14px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        border-color: #6366F1 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ENCABEZADO INSTITUCIONAL
# ==============================================================================
render_html(f"""
<div class="top-nav">
    <div class="brand-container">
        <div class="brand-icon-box">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        <div class="brand-title-group">
            <div class="brand-title-row">
                <span class="brand-name">ALTUR</span>
                <span class="brand-tag">VOICE AI DEFENSE</span>
            </div>
            <span class="brand-subtitle">Bank Telephony Deepfake Detection Engine</span>
        </div>
    </div>
    <div class="nav-status-group">
        <div class="status-pill">
            <span class="status-dot"></span>
            <strong style="color: #F3F4F6;">API Live</strong>
            <span style="color: #4B5563;">|</span>
            <span>Threshold: <strong style="color: #818CF8;">{threshold_val:.2f}</strong></span>
        </div>
        <a href="http://localhost:8000/docs" target="_blank" class="docs-btn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="16 18 22 12 16 6"></polyline>
                <polyline points="8 6 2 12 8 18"></polyline>
            </svg>
            Swagger Docs
        </a>
    </div>
</div>
""")

# ==============================================================================
# 4 TARJETAS DE ARQUITECTURA DE MODELOS (SUB-HEADER)
# ==============================================================================
render_html("""
<div class="models-grid">
    <div class="model-card">
        <div class="model-icon icon-purple">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="18" cy="5" r="3"></circle>
                <circle cx="6" cy="12" r="3"></circle>
                <circle cx="18" cy="19" r="3"></circle>
                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
            </svg>
        </div>
        <div>
            <div class="model-info-label">MODEL 1: AASIST</div>
            <div class="model-info-value">Graph Latents (132-dim)</div>
        </div>
    </div>
    <div class="model-card">
        <div class="model-icon icon-violet">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
            </svg>
        </div>
        <div>
            <div class="model-info-label">MODEL 2: RAWNET2</div>
            <div class="model-info-value">Raw Waveforms (4-dim)</div>
        </div>
    </div>
    <div class="model-card">
        <div class="model-icon icon-blue">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
        </div>
        <div>
            <div class="model-info-label">MODEL 3: ACOUSTIC DSP</div>
            <div class="model-info-value">Spectral/MFCCs (136-dim)</div>
        </div>
    </div>
    <div class="model-card">
        <div class="model-icon icon-teal">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="4" y1="21" x2="4" y2="14"></line>
                <line x1="4" y1="10" x2="4" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12" y2="3"></line>
                <line x1="20" y1="21" x2="20" y2="16"></line>
                <line x1="20" y1="12" x2="20" y2="3"></line>
                <line x1="1" y1="14" x2="7" y2="14"></line>
                <line x1="9" y1="8" x2="15" y2="8"></line>
                <line x1="17" y1="16" x2="23" y2="16"></line>
            </svg>
        </div>
        <div>
            <div class="model-info-label">META-LEARNER: XGBOOST</div>
            <div class="model-info-value">EER Calibrated (272-dim)</div>
        </div>
    </div>
</div>
""")

# ==============================================================================
# ÁREA DE TRABAJO PRINCIPAL (2 BLOQUES LADO A LADO)
# ==============================================================================
col_left, col_right = st.columns([1.25, 1.0], gap="large")

# Variables de estado de audio para la sesión
if "selected_audio_bytes" not in st.session_state:
    st.session_state.selected_audio_bytes = None
if "selected_audio_name" not in st.session_state:
    st.session_state.selected_audio_name = None
if "last_verdict" not in st.session_state:
    st.session_state.last_verdict = None

with col_left:
    # --------------------------------------------------------------------------
    # PANEL 1: AUDIO INPUT SOURCE
    # --------------------------------------------------------------------------
    render_html("""
    <div class="cyber-panel">
        <div class="panel-header">
            <div class="panel-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                AUDIO INPUT SOURCE
            </div>
        </div>
    </div>
    """)

    input_tab1, input_tab2, input_tab3 = st.tabs(["Upload / Drop", "Sample Presets", "Batch Ingestion"])

    with input_tab1:
        uploaded_file = st.file_uploader(
            "Arrastra y suelta tu archivo de audio aquí (Formatos: WAV, MP3, M4A, OGG)",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            key="main_file_uploader",
            help="Compatible con grabaciones telefónicas bancarias a 8kHz estéreo."
        )

        if uploaded_file is not None:
            st.session_state.selected_audio_bytes = uploaded_file.getvalue()
            st.session_state.selected_audio_name = uploaded_file.name

    with input_tab2:
        render_html("<div style='font-size: 12px; color: #94A3B8; margin-bottom: 8px;'>Selecciona una grabación del corpus telefónico bancario:</div>")
        sample_options = []
        if os.path.exists("audio"):
            try:
                available_samples = [f for f in sorted(os.listdir("audio")) if f.endswith(".wav")][:15]
                sample_options = available_samples
            except Exception:
                sample_options = []
        
        if not sample_options:
            sample_options = ["call_0181ce113ebe.wav", "call_018c9d3823ac.wav", "1.wav", "2.wav"]

        selected_sample = st.selectbox("Muestra de llamada", sample_options, label_visibility="collapsed")
        if st.button("Cargar Muestra Seleccionada", use_container_width=True):
            sample_path = os.path.join("audio", selected_sample)
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as sf:
                    st.session_state.selected_audio_bytes = sf.read()
                    st.session_state.selected_audio_name = selected_sample
                    st.session_state.last_verdict = None
                    st.rerun()

    with input_tab3:
        render_html("<div style='font-size: 12px; color: #94A3B8; margin-bottom: 8px;'>Ingesta masiva de llamadas y exportación CSV:</div>")
        batch_files = st.file_uploader(
            "Archivos del lote",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_uploader"
        )
        if batch_files and st.button("Ejecutar Análisis de Lote", use_container_width=True):
            with st.spinner(f"Analizando {len(batch_files)} grabaciones..."):
                batch_res = []
                for bf in batch_files:
                    b64 = base64.b64encode(bf.read()).decode('utf-8')
                    try:
                        r = requests.post(f"{API_URL}/detect", json={"call_id": bf.name, "audio_base64": b64, "sample_rate": 8000, "channels": 2}, timeout=30)
                        if r.status_code == 200:
                            rj = r.json()
                            batch_res.append({
                                "Call ID": bf.name,
                                "Veredicto": "SINTÉTICO" if rj.get("is_synthetic") else "AUTÉNTICO",
                                "Confianza": f"{rj.get('confidence', 0.5):.4f}",
                                "Score GBM": f"{rj.get('desglose', {}).get('acustica_timing', 0):.4f}",
                                "Score CNN": f"{rj.get('desglose', {}).get('neuronal_cnn', 0):.4f}"
                            })
                    except Exception as ex:
                        batch_res.append({"Call ID": bf.name, "Veredicto": "ERROR", "Confianza": str(ex)[:20], "Score GBM": "-", "Score CNN": "-"})
                df_b = pd.DataFrame(batch_res)
                st.dataframe(df_b, use_container_width=True)
                st.download_button("Exportar Reporte CSV", data=df_b.to_csv(index=False).encode('utf-8'), file_name="batch_report.csv", mime="text/csv")

    # --------------------------------------------------------------------------
    # PANEL 2: AUDIO STREAM PREVIEW
    # --------------------------------------------------------------------------
    render_html("""
    <div class="cyber-panel" style="margin-top: 16px;">
        <div class="panel-header">
            <div class="panel-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 18v-6a9 9 0 0 1 18 0v6"></path>
                    <path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"></path>
                </svg>
                Audio Stream Preview
            </div>
            <div style="background: #080C14; border: 1px solid #1E293B; border-radius: 6px; padding: 4px 10px; font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                8000 Hz &bull; Stereo (Ch0: Caller)
            </div>
        </div>
    </div>
    """)

    if st.session_state.selected_audio_bytes is None:
        render_html("""
        <div class="waveform-box">
            <span>Carga un archivo de audio o selecciona una muestra para reproducir</span>
        </div>
        """)
        audio_name_display = "Ningún audio cargado"
        audio_time_display = "0:00 / 0:00"
    else:
        st.audio(st.session_state.selected_audio_bytes)
        size_kb = len(st.session_state.selected_audio_bytes) / 1024
        audio_name_display = st.session_state.selected_audio_name or "stream_audio.wav"
        audio_time_display = f"{size_kb:.1f} KB &bull; Listo para análisis"

    # Barra Inferior con Botón RUN DETECTION
    btn_col1, btn_col2 = st.columns([1.3, 1.0])
    with btn_col1:
        render_html(f"""
        <div class="audio-playback-info" style="padding-top: 8px;">
            <div class="play-btn-circle">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
            </div>
            <div>
                <div style="font-weight: 600; color: #F1F5F9; font-size: 12.5px; font-family: 'JetBrains Mono', monospace;">{audio_name_display}</div>
                <div style="color: #64748B; font-size: 11px;">{audio_time_display}</div>
            </div>
        </div>
        """)

    with btn_col2:
        if st.button("⚡ RUN DETECTION", use_container_width=True):
            if st.session_state.selected_audio_bytes is None:
                st.warning("Por favor sube un archivo de audio o selecciona una muestra primero.")
            else:
                with st.spinner("Procesando extracción acústica y análisis neuronal..."):
                    try:
                        audio_b64 = base64.b64encode(st.session_state.selected_audio_bytes).decode('utf-8')
                        call_id = st.session_state.selected_audio_name or "sample_call"

                        req_payload = {
                            "call_id": call_id,
                            "audio_base64": audio_b64,
                            "sample_rate": 8000,
                            "channels": 2
                        }

                        t0 = time.perf_counter()
                        res = requests.post(f"{API_URL}/detect", json=req_payload, timeout=30)
                        latency = time.perf_counter() - t0

                        if res.status_code == 200:
                            data = res.json()
                            data["latency"] = latency
                            data["call_id"] = call_id
                            st.session_state.last_verdict = data
                            st.rerun()
                        else:
                            st.error(f"Error en API ({res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"Error de conexión con el servicio: {str(e)}")

with col_right:
    # --------------------------------------------------------------------------
    # PANEL DERECHO: RESULTADO DE EVALUACIÓN FORENSE
    # --------------------------------------------------------------------------
    verdict_data = st.session_state.last_verdict

    if verdict_data is None:
        # Estado Inicial cuando no se ha corrido la detección
        idle_box_html = """
        <div class="cyber-panel" style="min-height: 480px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="panel-header">
                    <div class="panel-title" style="color: #94A3B8;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        </svg>
                        RESULTADO DE SEGURIDAD
                    </div>
                </div>
                <div class="verdict-idle-box" style="padding: 70px 20px;">
                    <div class="verdict-shield-icon">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        </svg>
                    </div>
                    <div class="verdict-idle-title">Listo para Evaluación de Seguridad</div>
                    <div class="verdict-idle-sub">Selecciona o sube una grabación y presiona <strong>RUN DETECTION</strong> para visualizar el resultado forense y gráficos de decisión.</div>
                </div>
            </div>
            <div class="card-footer-info">
                <span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline; vertical-align: middle; margin-right: 4px;">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    </svg>
                    Bank Security Level 4
                </span>
                <span style="font-family: 'JetBrains Mono', monospace;">call_id: <span style="color: #94A3B8;">-</span></span>
            </div>
        </div>
        """
        render_html(idle_box_html)
    else:
        # Estado con Inferencia Ejecutada y Datos Reales del Endpoint
        is_syn = verdict_data.get("is_synthetic", False)
        confidence = float(verdict_data.get("confidence", 0.0))
        desglose = verdict_data.get("desglose", {})
        latency = float(verdict_data.get("latency", 0.0))
        call_id_display = verdict_data.get("call_id", "-")
        score_gbm = float(desglose.get("acustica_timing", 0.0))
        score_cnn = float(desglose.get("neuronal_cnn", 0.0))
        p_syn = float(desglose.get("p_sintetico", score_gbm if is_syn else 1.0 - confidence))

        gbm_pct = max(0.0, min(100.0, score_gbm * 100))
        cnn_pct = max(0.0, min(100.0, score_cnn * 100))
        conf_pct = max(0.0, min(100.0, confidence * 100))

        if is_syn:
            banner_html = """
            <div style="background: rgba(127, 29, 29, 0.25); border: 1px solid #DC2626; border-radius: 10px; padding: 18px; text-align: center; margin-bottom: 14px;">
                <div style="width: 44px; height: 44px; background: rgba(220, 38, 38, 0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto; color: #EF4444;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                    </svg>
                </div>
                <div style="font-size: 15px; font-weight: 800; color: #FCA5A5; letter-spacing: -0.01em;">FRAUDE DETECTADO: VOZ SINTÉTICA</div>
                <div style="font-size: 11.5px; color: #F87171; margin-top: 4px;">Nivel de Riesgo: Crítico &bull; Patrón Generativo TTS Detectado</div>
            </div>
            """
            bar_color_gbm = "#EF4444"
            bar_color_cnn = "#F87171"
        else:
            banner_html = """
            <div style="background: rgba(6, 78, 59, 0.25); border: 1px solid #059669; border-radius: 10px; padding: 18px; text-align: center; margin-bottom: 14px;">
                <div style="width: 44px; height: 44px; background: rgba(16, 185, 129, 0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px auto; color: #10B981;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                </div>
                <div style="font-size: 15px; font-weight: 800; color: #6EE7B7; letter-spacing: -0.01em;">VOZ AUTÉNTICA: HUMANO VERIFICADO</div>
                <div style="font-size: 11.5px; color: #34D399; margin-top: 4px;">Nivel de Riesgo: Mínimo &bull; Resonancia Natural del Tracto Vocal</div>
            </div>
            """
            bar_color_gbm = "#10B981"
            bar_color_cnn = "#34D399"

        result_box_html = f"""
        <div class="cyber-panel" style="min-height: 480px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="panel-header">
                    <div class="panel-title" style="color: #94A3B8;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#6366F1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        </svg>
                        RESULTADO DE SEGURIDAD
                    </div>
                </div>
                
                {banner_html}
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;">
                    <div style="background: #080C14; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <div style="font-size: 10px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">CONFIANZA</div>
                        <div style="font-size: 20px; font-weight: 800; color: #F8FAFC; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{conf_pct:.2f}%</div>
                    </div>
                    <div style="background: #080C14; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <div style="font-size: 10px; font-weight: 700; color: #94A3B8; text-transform: uppercase;">LATENCIA</div>
                        <div style="font-size: 20px; font-weight: 800; color: #60A5FA; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{latency:.3f}s</div>
                    </div>
                </div>
                
                <div style="background: #080C14; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 10px;">Desglose Gráfico de Modelos</div>
                    
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; font-size: 11.5px; margin-bottom: 4px;">
                            <span style="color: #94A3B8;">Modelo Acústico (HistGBM):</span>
                            <span style="font-weight: 700; color: #E2E8F0; font-family: 'JetBrains Mono', monospace;">{score_gbm:.4f}</span>
                        </div>
                        <div style="background: #1E293B; border-radius: 4px; height: 6px; overflow: hidden;">
                            <div style="background: {bar_color_gbm}; width: {gbm_pct}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>

                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 11.5px; margin-bottom: 4px;">
                            <span style="color: #94A3B8;">Modelo Neuronal (CNN):</span>
                            <span style="font-weight: 700; color: #E2E8F0; font-family: 'JetBrains Mono', monospace;">{score_cnn:.4f}</span>
                        </div>
                        <div style="background: #1E293B; border-radius: 4px; height: 6px; overflow: hidden;">
                            <div style="background: {bar_color_cnn}; width: {cnn_pct}%; height: 100%; border-radius: 4px;"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card-footer-info">
                <span>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline; vertical-align: middle; margin-right: 4px;">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    </svg>
                    Bank Security Level 4
                </span>
                <span style="font-family: 'JetBrains Mono', monospace;">call_id: <span style="color: #94A3B8;">{call_id_display}</span></span>
            </div>
        </div>
        """
        render_html(result_box_html)
        with st.expander("Contrato de Respuesta API (POST /detect)", expanded=False):
            st.code(json.dumps({"is_synthetic": is_syn, "confidence": confidence}, indent=2), language="json")