"""
app/app.py
Web app Streamlit — Predição de No-Show em Consultas Médicas
Executar: streamlit run app/app.py
"""

import pickle, pathlib, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "noshow_model.pkl"

# ── Carregar modelo ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

artifact = load_model()

# ── Config da página ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicAI — Predição de No-Show",
    page_icon="🏥",
    layout="wide",
)

# ── CSS mínimo ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-high   { background:#fee2e2; border-left:5px solid #dc2626; padding:1rem; border-radius:8px; }
.risk-medium { background:#fef9c3; border-left:5px solid #ca8a04; padding:1rem; border-radius:8px; }
.risk-low    { background:#dcfce7; border-left:5px solid #16a34a; padding:1rem; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🏥 ClinicAI — Predição de No-Show")
st.caption("Sistema de apoio à decisão para otimização da agenda de consultas")

if artifact is None:
    st.error("⚠️ Modelo não encontrado. Corre primeiro `python models/train_model.py` na raiz do projeto.")
    st.stop()

pipeline = artifact["pipeline"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_info = st.tabs(["📋 Consulta Individual", "📊 Análise em Lote", "ℹ️ Sobre o Modelo"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Consulta individual
# ════════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Prever risco de no-show para uma consulta")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Dados do Paciente**")
        age            = st.slider("Idade", 5, 90, 35)
        gender         = st.selectbox("Género", ["F", "M"])
        distance_km    = st.number_input("Distância à clínica (km)", 0.5, 60.0, 5.0, step=0.5)
        chronic_disease= st.toggle("Doença crónica?", value=False)
        previous_noshow_rate = st.slider("Taxa histórica de no-show", 0.0, 1.0, 0.1, step=0.05,
                                         help="Proporção de consultas anteriores em que o paciente não compareceu")

    with col2:
        st.markdown("**Dados da Consulta**")
        specialty      = st.selectbox("Especialidade",
                                      ["Clínica Geral", "Pediatria", "Cardiologia",
                                       "Dermatologia", "Ortopedia"])
        day_of_week    = st.selectbox("Dia da semana",
                                      ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"],
                                      index=0)
        day_num        = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"].index(day_of_week)
        appointment_hour = st.slider("Hora da consulta", 8, 17, 10)
        lead_days      = st.slider("Dias desde a marcação", 0, 60, 7)
        is_first_visit = st.toggle("Primeira consulta?", value=False)

    with col3:
        st.markdown("**Contacto / Lembretes**")
        sms_received   = st.toggle("SMS de lembrete enviado?", value=True)

    # ── Predição ──────────────────────────────────────────────────────────────
    sample = pd.DataFrame([{
        "age": age,
        "distance_km": distance_km,
        "previous_noshow_rate": previous_noshow_rate,
        "lead_days": lead_days,
        "appointment_hour": appointment_hour,
        "gender": gender,
        "specialty": specialty,
        "chronic_disease": int(chronic_disease),
        "sms_received": int(sms_received),
        "is_first_visit": int(is_first_visit),
        "day_of_week": day_num,
    }])

    prob = pipeline.predict_proba(sample)[0, 1]

    st.divider()
    r1, r2, r3 = st.columns([1, 2, 1])
    with r2:
        st.metric("Probabilidade de No-Show", f"{prob:.1%}")

        if prob >= 0.55:
            st.markdown(f'<div class="risk-high">🔴 <b>Risco Elevado</b> — Recomenda-se contacto proativo com o paciente e/ou overbooking controlado.</div>', unsafe_allow_html=True)
        elif prob >= 0.30:
            st.markdown(f'<div class="risk-medium">🟡 <b>Risco Moderado</b> — Considerar envio de lembrete adicional.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">🟢 <b>Risco Baixo</b> — Sem ação necessária.</div>', unsafe_allow_html=True)

        # Gauge simples com matplotlib
        fig, ax = plt.subplots(figsize=(4, 0.6))
        ax.barh(0, 1, color="#e5e7eb", height=0.5)
        color = "#dc2626" if prob >= 0.55 else "#ca8a04" if prob >= 0.30 else "#16a34a"
        ax.barh(0, prob, color=color, height=0.5)
        ax.set_xlim(0, 1); ax.axis("off")
        st.pyplot(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Análise em lote
# ════════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Simular agenda com múltiplos pacientes")

    if st.button("🎲 Gerar 20 pacientes aleatórios"):
        from data.generate_data import generate_appointments
        batch = generate_appointments(n=20, seed=np.random.randint(0, 9999))
        feat_cols = ["age", "distance_km", "previous_noshow_rate", "lead_days",
                     "appointment_hour", "gender", "specialty", "chronic_disease",
                     "sms_received", "is_first_visit", "day_of_week"]
        probs = pipeline.predict_proba(batch[feat_cols])[:, 1]
        batch["Prob. No-Show"] = probs.round(3)
        batch["Risco"] = pd.cut(probs, bins=[0, 0.30, 0.55, 1],
                                labels=["🟢 Baixo", "🟡 Moderado", "🔴 Elevado"])
        batch["Real"] = batch["no_show"].map({0: "Presente", 1: "No-Show"})

        display_cols = ["age", "gender", "specialty", "lead_days",
                        "sms_received", "Prob. No-Show", "Risco", "Real"]
        st.dataframe(batch[display_cols].sort_values("Prob. No-Show", ascending=False),
                     use_container_width=True)

        # Distribuição de risco
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = batch["Risco"].value_counts()
        colors_map = {"🟢 Baixo": "#16a34a", "🟡 Moderado": "#ca8a04", "🔴 Elevado": "#dc2626"}
        bars = ax.bar(counts.index, counts.values,
                      color=[colors_map.get(k, "#888") for k in counts.index])
        ax.set_ylabel("Nº de consultas"); ax.set_title("Distribuição de Risco na Agenda")
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Clica no botão para gerar uma agenda simulada e ver a distribuição de risco.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sobre o modelo
# ════════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("Sobre o Modelo")
    st.markdown(f"""
    **Modelo treinado:** `{artifact['model_name']}`

    **Problema:** Prever se um paciente irá comparecer à sua consulta marcada.

    **Variáveis utilizadas:**
    | Variável | Tipo | Descrição |
    |---|---|---|
    | `age` | Numérica | Idade do paciente |
    | `distance_km` | Numérica | Distância à clínica |
    | `previous_noshow_rate` | Numérica | Histórico de faltas |
    | `lead_days` | Numérica | Dias entre marcação e consulta |
    | `appointment_hour` | Numérica | Hora da consulta |
    | `gender` | Categórica | Género |
    | `specialty` | Categórica | Especialidade médica |
    | `chronic_disease` | Binária | Tem doença crónica? |
    | `sms_received` | Binária | Recebeu SMS de lembrete? |
    | `is_first_visit` | Binária | É a primeira consulta? |
    | `day_of_week` | Ordinal | Dia da semana (0=2ª … 4=6ª) |

    **Pipeline de ML:**
    1. Normalização das variáveis numéricas (StandardScaler)
    2. One-Hot Encoding das variáveis categóricas
    3. Classificador selecionado por cross-validation (ROC-AUC)
    """)

    img_eval = ROOT / "models" / "evaluation.png"
    img_feat = ROOT / "models" / "feature_importance.png"
    if img_eval.exists():
        st.image(str(img_eval), caption="Matriz de Confusão e Curva ROC", use_container_width=True)
    if img_feat.exists():
        st.image(str(img_feat), caption="Importância das Features", use_container_width=True)
    if not img_eval.exists():
        st.info("Corre `python models/train_model.py` para gerar os gráficos de avaliação.")
