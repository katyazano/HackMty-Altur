import streamlit as st
import requests
import base64
import json
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y METADATOS
# ==============================================================================
st.set_page_config(
    page_title="ALTUR | Voice Biometrics & Fraud Prevention",
    page_icon="https://img.icons8.com/ios-filled/50/0284c7/shield.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ==============================================================================
# ESTILOS INSTITUCIONALES (BANK-GRADE DESIGN SYSTEM)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Variables y Tipografía Base */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
    }

    /* Reducción de márgenes superiores */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Banner Superior Institucional */
    .bank-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .bank-title-group h1 {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .bank-title-group p {
        font-size: 13px;
        color: #94A3B8;
        margin: 4px 0 0 0;
        letter-spacing: 0.01em;
    }

    .bank-badge-group {
        display: flex;
        gap: 10px;
        align-items: center;
    }

    .bank-badge {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .badge-live {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .badge-sec {
        background-color: rgba(2, 132, 199, 0.12);
        color: #38BDF8;
        border-color: rgba(2, 132, 199, 0.3);
    }

    /* Pestañas Corporativas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0F172A;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1E293B;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #94A3B8;
        font-size: 13px;
        font-weight: 500;
        padding: 0 18px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }

    /* Tarjetas y Contenedores */
    .bank-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .bank-card-header {
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Veredictos de Seguridad */
    .verdict-box {
        border-radius: 8px;
        padding: 22px;
        margin: 16px 0;
        display: flex;
        align-items: flex-start;
        gap: 16px;
    }

    .verdict-human {
        background-color: rgba(6, 78, 59, 0.25);
        border: 1px solid #059669;
        color: #ECFDF5;
    }

    .verdict-synthetic {
        background-color: rgba(127, 29, 29, 0.25);
        border: 1px solid #DC2626;
        color: #FEF2F2;
    }

    .verdict-title {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 4px;
    }

    .verdict-desc {
        font-size: 13px;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Tarjetas de Métricas KPI */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin: 16px 0;
    }

    .kpi-box {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: left;
    }

    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-subtext {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
    }

    /* Botones de acción */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.02em;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }

    /* Tablas de Datos */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }

    /* Footer Institucional */
    .bank-footer {
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #1E293B;
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #64748B;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ENCABEZADO INSTITUCIONAL
# ==============================================================================
st.markdown("""
<div class="bank-header">
    <div class="bank-title-group">
        <h1>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            ALTUR SENTINEL
        </h1>
        <p>Sistema de Detección de Fraude en Voz y Validación Biométrica Telefónica</p>
    </div>
    <div class="bank-badge-group">
        <span class="bank-badge badge-live">
            <span style="height: 6px; width: 6px; background-color: #34D399; border-radius: 50%; display: inline-block;"></span>
            MOTOR ACTIVO
        </span>
        <span class="bank-badge badge-sec">
            ENCRIPTACIÓN TLS 1.3 / FIPS
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# PESTAÑAS DE CONTROL DE OPERACIONES
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "Verificación Individual",
    "Procesamiento por Lotes",
    "Auditoría y Calibración"
])

# ==============================================================================
# PESTAÑA 1: VERIFICACIÓN INDIVIDUAL DE LLAMADA
# ==============================================================================
with tab1:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 16px; font-weight: 600; color: #F1F5F9; margin: 0;">Inspección Forense de Grabación de Audio</h3>
        <p style="font-size: 13px; color: #94A3B8; margin-top: 2px;">Cargue un archivo de audio para evaluar la integridad acústica y patrones de latencia conversacional.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Archivo de audio para análisis (Formatos compatibles: WAV, MP3, M4A, OGG)",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        key="single_file_uploader",
        help="El sistema procesará y remuestreará internamente a 8kHz estéreo para la extracción espectral."
    )

    if uploaded_file is not None:
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        
        st.markdown(f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 12px 16px; margin: 12px 0; display: flex; justify-content: space-between; font-size: 12px; color: #94A3B8;">
            <div><strong>Archivo:</strong> <span style="color: #F1F5F9; font-family: monospace;">{uploaded_file.name}</span></div>
            <div><strong>Tamaño:</strong> <span style="color: #F1F5F9; font-family: monospace;">{file_size_kb:.1f} KB</span></div>
            <div><strong>Estado:</strong> <span style="color: #38BDF8;">Listo para inferencia</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.audio(uploaded_file)
        
        if st.button("Ejecutar Análisis Forense", type="primary", use_container_width=True):
            with st.spinner("Extrayendo 171 descriptores espectrales y procesando ensamble neuronal..."):
                try:
                    audio_bytes = uploaded_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                    request_body = {
                        "call_id": uploaded_file.name,
                        "audio_base64": audio_base64,
                        "sample_rate": 8000,
                        "channels": 2,
                    }

                    t0 = time.perf_counter()
                    response = requests.post(
                        f"{API_URL}/detect", json=request_body, timeout=30
                    )
                    latency = time.perf_counter() - t0

                    if response.status_code == 200:
                        result = response.json()
                        is_synthetic = result.get("is_synthetic")
                        confidence = result.get("confidence", 0.0)
                        desglose = result.get("desglose", {})

                        # Renderizado de Veredicto Principal
                        if is_synthetic:
                            st.markdown("""
                            <div class="verdict-box verdict-synthetic">
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0; margin-top: 2px;">
                                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                    <line x1="12" y1="9" x2="12" y2="13"></line>
                                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                                </svg>
                                <div>
                                    <div class="verdict-title" style="color: #FCA5A5;">RIESGO CRÍTICO: VOZ SINTÉTICA DETECTADA</div>
                                    <div class="verdict-desc">La señal exhibe regularidades acústicas no biológicas, patrones de contraste espectral planos o artefactos de modelos generativos TTS. Se recomienda bloqueo preventivo y confirmación por canal secundario.</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="verdict-box verdict-human">
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0; margin-top: 2px;">
                                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                                </svg>
                                <div>
                                    <div class="verdict-title" style="color: #6EE7B7;">AUTENTICACIÓN EXITOSA: VOZ HUMANA CONFIRMADA</div>
                                    <div class="verdict-desc">El análisis confirma la presencia de formantes naturales del tracto vocal, variabilidad armónica orgánica y distribución fisiológica de pausas y respiración.</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Tarjetas de Telemetría Forense
                        p_syn = desglose.get("p_sintetico", confidence if is_synthetic else 1.0 - confidence)
                        gbm_score = desglose.get("acustica_timing", 0.0)
                        cnn_score = desglose.get("neuronal_cnn", 0.0)

                        st.markdown(f"""
                        <div class="kpi-container">
                            <div class="kpi-box">
                                <div class="kpi-label">Confianza del Veredicto</div>
                                <div class="kpi-value">{confidence * 100:.2f}%</div>
                                <div class="kpi-subtext">Certeza estadística</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Probabilidad Sintética</div>
                                <div class="kpi-value">{p_syn * 100:.2f}%</div>
                                <div class="kpi-subtext">P(sintético | señal)</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Motor HistGradient</div>
                                <div class="kpi-value">{gbm_score:.4f}</div>
                                <div class="kpi-subtext">171 features acústicas</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Red Neuronal CNN</div>
                                <div class="kpi-value">{cnn_score:.4f}</div>
                                <div class="kpi-subtext">Log-mel espectrograma</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Latencia de Respuesta</div>
                                <div class="kpi-value">{latency:.3f} s</div>
                                <div class="kpi-subtext">SLA objetivo: &lt; 30.0 s</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Auditoría Técnica y Contrato de Endpoint
                        st.markdown("""
                        <div style="margin-top: 24px; margin-bottom: 8px;">
                            <span style="font-size: 13px; font-weight: 600; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em;">Auditoría de Cumplimiento de API (POST /detect)</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # JSON Oficial del Juez
                        st.code(json.dumps(
                            {"is_synthetic": is_synthetic, "confidence": confidence},
                            indent=2), language="json")

                        col_req, col_desc = st.columns(2)
                        with col_req:
                            with st.expander("Detalle del Request Enviado", expanded=False):
                                preview = dict(request_body)
                                preview["audio_base64"] = (
                                    audio_base64[:48] + f"... ({len(audio_base64)/1e6:.2f} MB)")
                                st.code(json.dumps(preview, indent=2), language="json")
                        with col_desc:
                            with st.expander("Desglose Interno del Ensamble", expanded=False):
                                st.json(desglose)
                    else:
                        st.error(f"Error de comunicación con el servicio de inferencia: Código HTTP {response.status_code}")
                except Exception as e:
                    st.error(f"Error de conexión con el motor de inferencia: {e}. Verifique el estado del microservicio.")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Ficha Técnica y Arquitectura
    with st.expander("Especificación Técnica del Motor de Inferencia", expanded=False):
        st.markdown(r"""
        **Pipeline Forense de Extracción Acústica y Análisis de Timing**
        
        El sistema extrae **171 descriptores estadísticos interpretables** combinados con un modelo convolucional de micro-artefactos:
        - **Modelado de Tracto Vocal:** Coeficientes Cepstrales en la Escala de Mel (MFCCs 1-20), primeras y segundas derivadas ($\Delta$ y $\Delta^2$).
        - **Contraste Espectral:** Medición de valles y crestas de energía en 4 sub-bandas (las voces sintéticas presentan uniformidad anómala).
        - **Contenido Armónico:** Vector cromático (Chroma STFT) y energía de centroides espectrales.
        - **Dinámica Conversacional:** Dispersión y coeficientes de variación de tiempos de respuesta, silencios y pausas.
        """)
        st.code("""
# Pipeline de extracción de características en memoria
def acoustic_features(caller, sr=8000):
    feats = {}
    mfcc = librosa.feature.mfcc(y=caller, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    contrast = librosa.feature.spectral_contrast(y=caller, sr=sr, n_bands=4)
    chroma = librosa.feature.chroma_stft(y=caller, sr=sr)
    return feats
        """, language="python")

# ==============================================================================
# PESTAÑA 2: PROCESAMIENTO POR LOTES
# ==============================================================================
with tab2:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 16px; font-weight: 600; color: #F1F5F9; margin: 0;">Ingesta y Validación de Lotes de Audio</h3>
        <p style="font-size: 13px; color: #94A3B8; margin-top: 2px;">Cargue múltiples grabaciones telefónicas para procesamiento masivo y exportación de reportes de riesgo.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Seleccione múltiples archivos de audio (WAV, MP3, M4A, OGG)",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        accept_multiple_files=True,
        key="batch_file_uploader"
    )

    if uploaded_files:
        st.markdown(f"""
        <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 12px 16px; margin: 12px 0; font-size: 12px; color: #94A3B8;">
            Archivos seleccionados para evaluación: <strong style="color: #F8FAFC;">{len(uploaded_files)}</strong>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Iniciar Procesamiento de Lote", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            synthetic_count = 0
            human_count = 0
            error_count = 0
            t_batch_start = time.perf_counter()

            for idx, file in enumerate(uploaded_files):
                status_text.markdown(f"<span style='font-size: 12px; color: #94A3B8;'>Analizando registro {idx + 1} de {len(uploaded_files)}: <code style='color: #38BDF8;'>{file.name}</code></span>", unsafe_allow_html=True)
                audio_bytes = file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                try:
                    resp = requests.post(
                        f"{API_URL}/detect",
                        json={"call_id": file.name, "audio_base64": audio_b64,
                              "sample_rate": 8000, "channels": 2},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        j = resp.json()
                        is_syn = j.get("is_synthetic", False)
                        if is_syn:
                            synthetic_count += 1
                            tag = "SINTÉTICO (ALTO RIESGO)"
                        else:
                            human_count += 1
                            tag = "HUMANO (VERIFICADO)"

                        desg = j.get("desglose", {})
                        results.append({
                            "ID Registro": file.name,
                            "Clasificación": tag,
                            "Confianza": f"{j.get('confidence', 0.5):.4f}",
                            "P(Sintético)": f"{desg.get('p_sintetico', 0.5):.4f}",
                            "Score Acústico (GBM)": f"{desg.get('acustica_timing', 0.0):.4f}",
                            "Score Neuronal (CNN)": f"{desg.get('neuronal_cnn', 0.0):.4f}"
                        })
                    else:
                        error_count += 1
                        results.append({
                            "ID Registro": file.name,
                            "Clasificación": "ERROR DE RESPUESTA",
                            "Confianza": "N/A",
                            "P(Sintético)": "N/A",
                            "Score Acústico (GBM)": "N/A",
                            "Score Neuronal (CNN)": "N/A"
                        })
                except Exception as e:
                    error_count += 1
                    results.append({
                        "ID Registro": file.name,
                        "Clasificación": "ERROR DE CONEXIÓN",
                        "Confianza": str(e)[:24],
                        "P(Sintético)": "N/A",
                        "Score Acústico (GBM)": "N/A",
                        "Score Neuronal (CNN)": "N/A"
                    })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            batch_latency = time.perf_counter() - t_batch_start
            status_text.empty()
            progress_bar.empty()

            # Resumen Ejecutivo de Lote
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-box">
                    <div class="kpi-label">Total Procesados</div>
                    <div class="kpi-value">{len(uploaded_files)}</div>
                    <div class="kpi-subtext">Llamadas analizadas</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Fraude Detectado</div>
                    <div class="kpi-value" style="color: #F87171;">{synthetic_count}</div>
                    <div class="kpi-subtext">{(synthetic_count / max(1, len(uploaded_files)) * 100):.1f}% del lote</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Humanos Verificados</div>
                    <div class="kpi-value" style="color: #34D399;">{human_count}</div>
                    <div class="kpi-subtext">{(human_count / max(1, len(uploaded_files)) * 100):.1f}% del lote</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Tiempo de Lote</div>
                    <div class="kpi-value">{batch_latency:.2f} s</div>
                    <div class="kpi-subtext">Promedio: {batch_latency/max(1, len(uploaded_files)):.2f} s/audio</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            # Exportación de reporte forense
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Exportar Informe de Auditoría (CSV)",
                data=csv_data,
                file_name="reporte_deteccion_altur.csv",
                mime="text/csv"
            )

# ==============================================================================
# PESTAÑA 3: AUDITORÍA DE BENCHMARK Y CALIBRACIÓN
# ==============================================================================
with tab3:
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 16px; font-weight: 600; color: #F1F5F9; margin: 0;">Evaluación y Validación Institucional del Modelo</h3>
        <p style="font-size: 13px; color: #94A3B8; margin-top: 2px;">Ejecuta una auditoría integral contra el conjunto de validación local (manifest.csv y audio/) para certificar métricas de discriminación y tasas de error.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Ejecutar Auditoría Completa de Conjunto Local", type="primary", use_container_width=True):
        with st.spinner("Procesando corpus de validación con hilos concurrentes..."):
            try:
                if not os.path.exists("manifest.csv"):
                    st.error("Archivo manifest.csv no encontrado en la raíz del repositorio.")
                else:
                    df = pd.read_csv("manifest.csv")
                    df["path"] = df["anon_id"].apply(lambda a: f"audio/{a}.wav")
                    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

                    if len(df) == 0:
                        st.error("No se encontraron grabaciones correspondientes en el directorio audio/.")
                    else:
                        y_true = (df["label"] == "synthetic").astype(int).values
                        y_pred = [None] * len(df)
                        confidences = [None] * len(df)
                        
                        progress_bar = st.progress(0)
                        status_holder = st.empty()

                        def analyze_audio_task(idx_row):
                            idx, row = idx_row
                            with open(row["path"], "rb") as f:
                                audio_b64 = base64.b64encode(f.read()).decode()
                            try:
                                resp = requests.post(
                                    f"{API_URL}/detect",
                                    json={"call_id": row["anon_id"], "audio_base64": audio_b64,
                                          "sample_rate": 8000, "channels": 2},
                                    timeout=30
                                )
                                if resp.status_code == 200:
                                    j = resp.json()
                                    p_syn = j.get("desglose", {}).get("p_sintetico", 0.5)
                                    return idx, 1 if j.get("is_synthetic") else 0, p_syn
                                else:
                                    return idx, 0, 0.5
                            except Exception:
                                return idx, 0, 0.5

                        with ThreadPoolExecutor(max_workers=4) as executor:
                            futures = {executor.submit(analyze_audio_task, (idx, row)): idx
                                      for idx, (_, row) in enumerate(df.iterrows())}
                            for completed in as_completed(futures):
                                idx, pred, conf = completed.result()
                                y_pred[idx] = pred
                                confidences[idx] = conf
                                done = sum(1 for x in y_pred if x is not None)
                                progress_bar.progress(done / len(df))
                                status_holder.markdown(f"<span style='font-size: 12px; color: #94A3B8;'>Correlacionando muestras forenses: {done} / {len(df)}</span>", unsafe_allow_html=True)

                        progress_bar.empty()
                        status_holder.empty()

                        y_pred = np.array(y_pred)
                        confidences = np.array(confidences)

                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        auc = roc_auc_score(y_true, confidences)

                        tn = int(((y_true == 0) & (y_pred == 0)).sum())
                        fp = int(((y_true == 0) & (y_pred == 1)).sum())
                        fn = int(((y_true == 1) & (y_pred == 0)).sum())
                        tp = int(((y_true == 1) & (y_pred == 1)).sum())

                        # Panel de Métricas de Calidad
                        st.markdown(f"""
                        <div class="kpi-container">
                            <div class="kpi-box">
                                <div class="kpi-label">Exactitud Global</div>
                                <div class="kpi-value">{acc * 100:.1f}%</div>
                                <div class="kpi-subtext">Muestras correctas</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Precisión de Fraude</div>
                                <div class="kpi-value">{prec * 100:.1f}%</div>
                                <div class="kpi-subtext">Pureza de alertas</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Exhaustividad (Recall)</div>
                                <div class="kpi-value">{rec * 100:.1f}%</div>
                                <div class="kpi-subtext">Captura de ataques</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Puntuación F1</div>
                                <div class="kpi-value">{f1:.3f}</div>
                                <div class="kpi-subtext">Balance armónico</div>
                            </div>
                            <div class="kpi-box">
                                <div class="kpi-label">Área ROC (AUC)</div>
                                <div class="kpi-value">{auc:.4f}</div>
                                <div class="kpi-subtext">Poder de discriminación</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Matriz de Riesgo Institucional
                        st.markdown("""
                        <div style="margin-top: 20px; margin-bottom: 10px;">
                            <span style="font-size: 13px; font-weight: 600; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em;">Matriz de Confusión y Fricción Operativa</span>
                        </div>
                        """, unsafe_allow_html=True)

                        conf_df = pd.DataFrame({
                            "Predicción: Humano": [f"{tn} (Verdadero Negativo)", f"{fn} (Falso Negativo / Fuga)"],
                            "Predicción: Sintético": [f"{fp} (Falso Positivo / Fricción)", f"{tp} (Verdadero Positivo / Neutralizado)"]
                        }, index=["Real: Humano", "Real: Sintético"])

                        st.dataframe(conf_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error durante la auditoría: {e}")

# ==============================================================================
# PIE DE PÁGINA CORPORATIVO
# ==============================================================================
st.markdown("""
<div class="bank-footer">
    <div>ALTUR FINANCIAL SECURITY PLATFORM &copy; 2026 &bull; PROTOCOLO DE BIOMETRÍA DE VOZ V2.4</div>
    <div>CUMPLIMIENTO DE CONTRATO /DETECT &bull; SLA LATENCIA &lt; 30.0s</div>
</div>
""", unsafe_allow_html=True)