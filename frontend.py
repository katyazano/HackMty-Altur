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

# Configuración de la página
st.set_page_config(page_title="Escudo Altur | Anti-Deepfake", page_icon="🛡️", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🛡️ Escudo Altur: Detección de Deepfakes en Llamadas")
st.markdown("Sube una grabación de voz para que nuestro modelo HistGradientBoosting analice la acústica y el ritmo conversacional.")

tab1, tab2, tab3 = st.tabs(["📁 Archivo Individual", "📦 Carpeta Múltiple", "📊 Desde Carpeta Local"])

with tab1:
    st.subheader("Sube un solo audio")
    uploaded_file = st.file_uploader("Sube tu audio (WAV, MP3, M4A)", type=['wav', 'mp3', 'm4a', 'ogg'])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Analizar Audio", type="primary", use_container_width=True):
            with st.spinner("Procesando features acústicas y timing..."):
                try:
                    audio_bytes = uploaded_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                    # Contrato EXACTO del juez (scripts/check_endpoint.py)
                    request_body = {
                        "call_id": uploaded_file.name,
                        "audio_base64": audio_base64,
                        "sample_rate": 8000,
                        "channels": 2,
                    }

                    t0 = time.perf_counter()
                    response = requests.post(
                        f"{API_URL}/detect", json=request_body, timeout=30)
                    latency = time.perf_counter() - t0

                    if response.status_code == 200:
                        result = response.json()
                        is_synthetic = result.get("is_synthetic")
                        confidence = result.get("confidence", 0.0)

                        st.markdown("---")
                        if is_synthetic:
                            st.error("🚨 **FRAUDE DETECTADO: Voz Sintética**")
                            st.caption("El modelo detectó latencias perfectas o espectros uniformes típicos de motores TTS.")
                        else:
                            st.success("✅ **HUMANO VERIFICADO**")
                            st.caption("Respiración, armónicos y tiempos de respuesta naturales validados.")

                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Confianza del veredicto", f"{confidence * 100:.2f}%")
                        with m2:
                            st.metric("Latencia", f"{latency:.2f} s", help="Límite del juez: 30 s")

                        st.markdown("---")
                        st.subheader("Respuesta del endpoint `POST /detect`")
                        st.caption("Este es el JSON exacto que recibe el benchmark de Altur.")

                        # Solo las dos llaves del contrato oficial
                        st.code(json.dumps(
                            {"is_synthetic": is_synthetic, "confidence": confidence},
                            indent=2), language="json")

                        req_col, res_col = st.columns(2)
                        with req_col:
                            with st.expander("Request enviado"):
                                preview = dict(request_body)
                                preview["audio_base64"] = (
                                    audio_base64[:48] + f"... ({len(audio_base64)/1e6:.1f} MB)")
                                st.code(json.dumps(preview, indent=2), language="json")
                        with res_col:
                            with st.expander("Desglose interno (no va al juez)"):
                                st.json(result.get("desglose", {}))
                    else:
                        st.warning(f"Error del servidor: {response.status_code}")
                except Exception as e:
                    st.error(f"Error de conexión: {e}. ¿Está corriendo el servidor FastAPI?")

    st.markdown("---")
    st.subheader("Arquitectura del Modelo")
    st.markdown("""
    A diferencia de modelos de caja negra, nuestro sistema extrae **171 características interpretables**:
    - **Señal Acústica:** MFCCs, Spectral Contrast y Chroma.
    - **Timing Conversacional:** Coeficiente de variación de latencia y pausas.
    """)

    with st.expander("Ver el motor de extracción (features.py)", expanded=False):
        st.code("""
# Fragmento clave de nuestro features.py
def acoustic_features(caller, sr):
    feats = {}

    # 1. Señal temporal del tracto vocal
    mfcc = librosa.feature.mfcc(y=caller, sr=sr, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    # 2. Spectral Contrast (TTS tiende a ser uniforme)
    contrast = librosa.feature.spectral_contrast(y=caller, sr=sr, n_bands=4)

    # 3. Chroma (Contenido armónico)
    chroma = librosa.feature.chroma_stft(y=caller, sr=sr)

    return feats
        """, language="python")

    st.info("💡 **Backend robusto:** Si subes un audio a 44.1kHz desde tu celular, nuestra API lo remuestrea automáticamente a 8kHz estéreo antes de la inferencia.")

with tab2:
    st.subheader("Sube múltiples audios")
    uploaded_files = st.file_uploader(
        "Sube varios audios (WAV, MP3, M4A)",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Analizar Todos", type="primary", use_container_width=True):
        with st.spinner(f"Analizando {len(uploaded_files)} audios..."):
            try:
                results = []
                for idx, file in enumerate(uploaded_files):
                    audio_bytes = file.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode()

                    try:
                        resp = requests.post(
                            f"{API_URL}/detect",
                            json={"call_id": file.name, "audio_base64": audio_b64,
                                  "sample_rate": 8000, "channels": 2},
                            timeout=30
                        )
                        if resp.status_code == 200:
                            j = resp.json()
                            results.append({
                                "Archivo": file.name,
                                "Sintético": "✓ SÍ" if j.get("is_synthetic") else "✗ NO",
                                "Confianza": f"{j.get('confidence', 0.5):.4f}",
                                "P(sintético)": f"{j.get('desglose', {}).get('p_sintetico', 0.5):.4f}",
                                "GBM": f"{j.get('desglose', {}).get('acustica_timing', 0):.4f}",
                                "CNN": f"{j.get('desglose', {}).get('neuronal_cnn', 0):.4f}"
                            })
                    except Exception as e:
                        results.append({
                            "Archivo": file.name,
                            "Sintético": "⚠️ ERROR",
                            "Confianza": str(e)[:30],
                            "GBM": "N/A",
                            "CNN": "N/A"
                        })

                    if (idx + 1) % 10 == 0:
                        st.info(f"Procesados {idx + 1}/{len(uploaded_files)}")

                df_results = pd.DataFrame(results)
                st.success(f"✅ Análisis completado: {len(uploaded_files)} audios")
                st.dataframe(df_results, use_container_width=True)

                # Descargar CSV
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="Descargar resultados (CSV)",
                    data=csv,
                    file_name="resultados_analisis.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.subheader("Análisis de Carpeta Local (audio/)")

    if st.button("📊 Analizar todo audio/", type="primary", use_container_width=True):
        with st.spinner("Analizando carpeta audio/ (esto toma ~1-2 min)..."):
            try:
                df = pd.read_csv("manifest.csv")
                df["path"] = df["anon_id"].apply(lambda a: f"audio/{a}.wav")
                df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

                if len(df) == 0:
                    st.error("No hay audios en audio/")
                else:
                    y_true = (df["label"] == "synthetic").astype(int).values
                    y_pred = [None] * len(df)
                    confidences = [None] * len(df)
                    progress_placeholder = st.empty()

                    def analyze_audio(idx_row):
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
                                # para AUC hace falta P(sintetico), no la confianza del veredicto
                                p_syn = j.get("desglose", {}).get("p_sintetico", 0.5)
                                return idx, 1 if j.get("is_synthetic") else 0, p_syn
                            else:
                                return idx, 0, 0.5
                        except Exception:
                            return idx, 0, 0.5

                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(analyze_audio, (idx, row)): idx
                                  for idx, (_, row) in enumerate(df.iterrows())}
                        for completed in as_completed(futures):
                            idx, pred, conf = completed.result()
                            y_pred[idx] = pred
                            confidences[idx] = conf
                            done = sum(1 for x in y_pred if x is not None)
                            progress_placeholder.info(f"Procesados {done}/{len(df)}")

                    y_pred = np.array(y_pred)
                    confidences = np.array(confidences)

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    auc = roc_auc_score(y_true, confidences)

                    st.success(f"✅ Análisis completado: {len(df)} audios")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Accuracy", f"{acc*100:.1f}%")
                    with col2:
                        st.metric("Precision", f"{prec*100:.1f}%")
                    with col3:
                        st.metric("Recall", f"{rec*100:.1f}%")
                    with col4:
                        st.metric("F1", f"{f1:.3f}")
                    with col5:
                        st.metric("AUC", f"{auc:.4f}")

                    tn = ((y_true == 0) & (y_pred == 0)).sum()
                    fp = ((y_true == 0) & (y_pred == 1)).sum()
                    fn = ((y_true == 1) & (y_pred == 0)).sum()
                    tp = ((y_true == 1) & (y_pred == 1)).sum()

                    st.write("**Matriz de Confusión:**")
                    conf_data = pd.DataFrame({
                        "Predicho Humano": [tn, fn],
                        "Predicho Sintético": [fp, tp]
                    }, index=["Real Humano", "Real Sintético"])
                    st.dataframe(conf_data, use_container_width=True)

            except FileNotFoundError:
                st.error("No encontrado manifest.csv o carpeta audio/")
            except Exception as e:
                st.error(f"Error: {e}")