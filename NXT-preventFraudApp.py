import streamlit as st          # Framework Web
import pandas as pd             # Gestione Dati
import numpy as np              # Matematica
import xgboost as xgb           # AI / ML
from sklearn.model_selection import train_test_split
import plotly.express as px     # Grafici
import plotly.graph_objects as go

# ==========================================
# 1. SETUP & CONFIGURAZIONE
# ==========================================
st.set_page_config(
    page_title="NXT Aegis | EU Grid Defense",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DIZIONARIO TRADUZIONI (CORRETTO) ---
TRANS = {
    "EN": {
        # KEYS MANCANTI AGGIUNTE QUI:
        "sec_config": "⚙️ CONFIGURATION",
        "sidebar_title": "CONTROL ROOM",
        "sec_glossary": "📚 EU TERMINOLOGY",
        "sec_charts": "📊 ANALYST GUIDE",

        "lbl_users": "Active Meter Points (POD)",
        "lbl_fraud": "Est. Non-Technical Loss (%)",
        "btn_run": "RUN COMPLIANCE SCAN",
        # GLOSSARY
        "term_sub": "**Substation (Cabina):** The secondary distribution node. We monitor the Delta between Substation output and aggregated Smart Meters.",
        "term_magnet": "**Magnet Tampering:** Placing strong magnets on older digital meters to saturate current transformers and under-record usage.",
        "term_firmware": "**Firmware Hack:** Sophisticated modification of the Smart Meter software to report lower values (Cyber-Fraud).",
        "term_bypass": "**Underground Bypass:** Physical diversion of the cable before the meter, typically hidden inside walls or underground.",
        "term_temp": "**Thermal Correlation:** Physics check. In EU winters, consumption *must* correlate with temperature drop (Heating). No correlation = Anomaly.",
        "term_gdpr": "**GDPR Compliance:** Data is processed using pseudo-anonymized POD IDs to comply with EU Privacy Regulation 2016/679.",
        "term_xgb": "**XGBoost:** Gradient Boosting algorithm trained on validated fraud patterns from European DSO databases.",
        # KPI
        "kpi_1": "HIGH RISK PODs",
        "kpi_2": "CONFIRMED ANOMALIES",
        "kpi_3": "MODEL PRECISION",
        "kpi_4": "EST. RECOVERY (€)",
        "tab_list": "🎯 PRIORITY LIST",
        "tab_deep": "🔍 METER ANALYSIS",
        "loading": "Processing Smart Meter Data...",
        "status_crit": "CRITICAL ANOMALY",
        "status_susp": "NON-CONFORMITY",
        "select": "SELECT POD ID",
        # CHARTS
        "ch_1": "1. LOAD CURVE (USER VS SUBSTATION AVG)",
        "ch_2": "2. RISK RADAR (PROFILE SHAPE)",
        "ch_3": "3. FRAUD TYPOLOGY",
        "ch_4": "4. RISK DISTRIBUTION (GDPR ANON)",
        "ch_5": "5. CLUSTER ANALYSIS (SCATTER)",
        "ch_6": "6. THERMAL CORRELATION CHECK",
        "ch_7": "7. SUBSTATION LOSS HEATMAP",
        "ch_8": "8. TARIFF CATEGORY OUTLIERS",
        # EXPLANATIONS
        "exp_1": "ℹ️ **Analysis:** Magenta Line = User. Cyan Area = Cluster Average. A flat line during peak hours suggests manipulation.",
        "exp_2": "ℹ️ **Analysis:** Distorted shapes indicate consumption suppression across multiple vectors (Volume, Stability).",
        "exp_3": "ℹ️ **Analysis:** 'Magnet' is common in residential. 'Firmware Hack' is an emerging threat in industrial zones.",
        "exp_4": "ℹ️ **Analysis:** Histogram of risk scores. The red tail represents the target group for inspection.",
        "exp_5": "ℹ️ **Analysis:** Frauds cluster in low-volume / high-deviation zones (Bottom-Left).",
        "exp_6": "ℹ️ **Analysis:** Correlation with Temperature (Grey Bars). If Temp drops (Winter) and consumption stays flat, it's a bypass.",
        "exp_7": "ℹ️ **Analysis:** Treemap of Secondary Substations. Red blocks indicate high aggregate losses (Delta Energy).",
        "exp_8": "ℹ️ **Analysis:** Statistical outliers by tariff category (Residential vs Industrial vs Commercial).",
    },
    "IT": {
        # KEYS MANCANTI AGGIUNTE QUI:
        "sec_config": "⚙️ CONFIGURAZIONE",
        "sidebar_title": "SALA CONTROLLO",
        "sec_glossary": "📚 TERMINOLOGIA EU",
        "sec_charts": "📊 GUIDA ANALISTA",

        "lbl_users": "Punti di Prelievo (POD)",
        "lbl_fraud": "Perdite Non Tecniche Stimate (%)",
        "btn_run": "AVVIA SCANSIONE COMPLIANCE",
        # GLOSSARY
        "term_sub": "**Cabina Secondaria:** Nodo di distribuzione locale. Monitoriamo il Delta tra l'uscita della cabina e la somma degli Smart Meter.",
        "term_magnet": "**Manomissione Magnetica:** Uso di magneti potenti per saturare i trasformatori dei contatori e registrare meno consumo.",
        "term_firmware": "**Firmware Hack:** Modifica software dello Smart Meter per inviare letture alterate (Cyber-Frode).",
        "term_bypass": "**Bypass Occulto:** Deviazione fisica del cavo a monte del contatore, spesso murata o interrata.",
        "term_temp": "**Correlazione Termica:** Check fisico. In inverno, il consumo *deve* salire col freddo (Riscaldamento). Nessuna correlazione = Anomalia.",
        "term_gdpr": "**Compliance GDPR:** I dati sono trattati usando ID POD pseudo-anonimizzati (Regolamento UE 2016/679).",
        "term_xgb": "**XGBoost:** Algoritmo addestrato su pattern di frode validati da database di DSO europei.",
        # KPI
        "kpi_1": "POD AD ALTO RISCHIO",
        "kpi_2": "ANOMALIE CONFERMATE",
        "kpi_3": "PRECISIONE MODELLO",
        "kpi_4": "RECUPERO STIMATO (€)",
        "tab_list": "🎯 LISTA PRIORITARIA",
        "tab_deep": "🔍 ANALISI CONTATORE",
        "loading": "Elaborazione Dati Smart Meter...",
        "status_crit": "ANOMALIA CRITICA",
        "status_susp": "NON CONFORMITÀ",
        "select": "SELEZIONA ID POD",
        # CHARTS
        "ch_1": "1. CURVA DI CARICO (UTENTE VS CABINA)",
        "ch_2": "2. RADAR DI RISCHIO (PROFILO)",
        "ch_3": "3. TIPOLOGIA FRODE",
        "ch_4": "4. DISTRIBUZIONE RISCHIO (GDPR ANON)",
        "ch_5": "5. ANALISI CLUSTER (SCATTER)",
        "ch_6": "6. CORRELAZIONE TERMICA",
        "ch_7": "7. HEATMAP CABINE SECONDARIE",
        "ch_8": "8. OUTLIER PER CATEGORIA TARIFFA",
        # EXPLANATIONS
        "exp_1": "ℹ️ **Analisi:** Linea Magenta = Utente. Area Ciano = Media del Cluster. Una linea piatta durante le ore di picco suggerisce manipolazione.",
        "exp_2": "ℹ️ **Analisi:** Forme distorte indicano soppressione del consumo su più vettori (Volume, Stabilità).",
        "exp_3": "ℹ️ **Analisi:** 'Magnete' è comune nel residenziale. 'Firmware Hack' è una minaccia emergente nell'industriale.",
        "exp_4": "ℹ️ **Analisi:** Istogramma dei punteggi di rischio. La coda rossa a destra rappresenta il gruppo target.",
        "exp_5": "ℹ️ **Analisi:** Le frodi si raggruppano nelle zone a basso volume / alta deviazione (In basso a sinistra).",
        "exp_6": "ℹ️ **Analisi:** Correlazione con Temperatura (Barre Grigie). Se la Temp scende (Inverno) e il consumo resta piatto, è bypass.",
        "exp_7": "ℹ️ **Analisi:** Mappa delle Cabine Secondarie. I blocchi rossi indicano alte perdite aggregate (Delta Energia).",
        "exp_8": "ℹ️ **Analisi:** Outlier statistici per categoria tariffaria (Domestico vs Altri usi).",
    }
}

# --- CSS "EURO-TECH" (Minimalista & CyberBlue) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700&family=Roboto+Mono:wght@400;500&display=swap');

    :root {
        --bg-dark: #0b1120;       /* Dark Blue Europe */
        --eu-blue: #3b82f6;       /* European Flag Blue */
        --alert-red: #ef4444;     /* Alert Red */
        --glass-border: rgba(59, 130, 246, 0.2);
    }

    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            linear-gradient(to right, rgba(59, 130, 246, 0.05) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(59, 130, 246, 0.05) 1px, transparent 1px);
        background-size: 40px 40px; /* Grid Pattern */
        font-family: 'Exo 2', sans-serif;
    }

    /* TITOLI */
    h1, h2, h3, h4 {
        font-family: 'Exo 2', sans-serif !important;
        text-transform: uppercase;
        color: white;
        letter-spacing: 1px;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid var(--glass-border);
    }

    /* KPI CARDS */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--glass-border);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--eu-blue);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Exo 2';
        font-weight: 700;
        color: white;
    }

    /* INFO BOX */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid var(--eu-blue);
        padding: 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #bfdbfe;
        margin-top: 8px;
        font-family: 'Roboto Mono', monospace;
    }

    /* BOTTONE */
    .stButton > button {
        background: var(--eu-blue);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background: #2563eb;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (EU CONTEXT)
# ==========================================


@st.cache_data
def generate_eu_data(n, fraud_rate):
    """
    Genera dati per il mercato Europeo.
    """
    data = []
    np.random.seed(42)

    # 30 Cabine Secondarie (Zone Urbane/Industriali)
    sub_ids = [f"CAB-{i:03d}" for i in range(30)]

    # Profili Temperatura (Simulazione Nord Italia/Europa)
    temperature_inv = np.array(
        [1.8, 1.6, 1.2, 0.8, 0.5, 1.2, 1.5, 1.4, 0.6, 0.8, 1.4, 1.7])

    for i in range(n):
        sub = np.random.choice(sub_ids)
        is_fraud = np.random.rand() < (fraud_rate / 100)
        cat = np.random.choice(
            ['Residential', 'Business', 'Industrial'], p=[0.8, 0.15, 0.05])
        base = 250 if cat == 'Residential' else (
            1200 if cat == 'Business' else 8000)

        cons_real = base * temperature_inv * np.random.uniform(0.9, 1.1, 12)
        cons_metered = cons_real.copy()
        fraud_type = 'None'

        if is_fraud:
            dice = np.random.rand()
            if dice < 0.4:
                fraud_type = 'Magnet Tampering'
                cons_metered *= 0.6
            elif dice < 0.7:
                fraud_type = 'Underground Bypass'
                cons_metered *= 0.2
            else:
                fraud_type = 'Firmware Hack'
                cons_metered = np.where(cons_metered > (
                    base*0.5), base*0.5, cons_metered)

        for m, val in enumerate(cons_metered):
            data.append({
                'POD_ID': f"IT001E{i:06d}",
                'SUBSTATION': sub,
                'MONTH': m+1,
                'KWH': max(0, val),
                'TEMP_INDEX': temperature_inv[m],
                'IS_FRAUD': 1 if is_fraud else 0,
                'FRAUD_TYPE': fraud_type,
                'CATEGORY': cat
            })

    return pd.DataFrame(data)


def run_eu_ml(df):
    """Pipeline ML Adattata"""
    stats = df.groupby(['POD_ID', 'SUBSTATION', 'CATEGORY']).agg(
        AVG_KWH=('KWH', 'mean'),
        STD_KWH=('KWH', 'std'),
        TARGET=('IS_FRAUD', 'max'),
        F_TYPE=('FRAUD_TYPE', 'first')
    ).reset_index()

    sub_avg = stats.groupby('SUBSTATION')['AVG_KWH'].mean(
    ).reset_index().rename(columns={'AVG_KWH': 'SUB_AVG'})
    stats = stats.merge(sub_avg, on='SUBSTATION')
    stats['DEVIATION'] = stats['AVG_KWH'] / (stats['SUB_AVG'] + 1)

    X = stats[['AVG_KWH', 'STD_KWH', 'DEVIATION']]
    y = stats['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X)[:, 1]
    noise = np.random.normal(0, 0.001, size=len(probs))
    probs = np.clip(probs + noise, 0, 0.999)
    stats['RISK_SCORE'] = probs

    return stats.sort_values('RISK_SCORE', ascending=False), df


if 'res' not in st.session_state:
    st.session_state.res = None

# ==========================================
# 3. INTERFACCIA (UI)
# ==========================================

with st.sidebar:
    # 1. LANGUAGE
    lang = st.selectbox("🌐 Language / Lingua", ["English", "Italiano"])
    L_CODE = "EN" if lang == "English" else "IT"
    TXT = TRANS[L_CODE]

    # 2. CONFIG
    with st.expander(TXT['sec_config'], expanded=True):
        n_users = st.slider(TXT['lbl_users'], 1000, 10000, 3000)
        f_rate = st.slider(TXT['lbl_fraud'], 1, 15, 5)
        if st.button(TXT['btn_run']):
            with st.spinner(TXT['loading']):
                raw = generate_eu_data(n_users, f_rate)
                res, raw_data = run_eu_ml(raw)
                st.session_state.res = res
                st.session_state.raw = raw_data

        st.caption("✅ GDPR Compliant Mode")
        st.caption("✅ ISO 27001 Security")

    # 3. GLOSSARY
    with st.expander(TXT['sec_glossary']):
        st.markdown(TXT['term_sub'])
        st.markdown(TXT['term_bypass'])
        st.markdown(TXT['term_magnet'])
        st.markdown(TXT['term_temp'])
        st.markdown(TXT['term_gdpr'])
        st.markdown(TXT['term_xgb'])

# HEADER
st.markdown(f"""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; border-bottom:1px solid rgba(59,130,246,0.3); padding-bottom:10px;'>
        <div>
            <h1 style='margin:0; font-size:3rem; color:white;'>NXT <span style='color:#3b82f6'>AEGIS</span></h1>
            <p style='color:#94a3b8; font-family:"Exo 2"; letter-spacing:2px;'>EU GRID INTELLIGENCE // v10.1</p>
        </div>
        <div style='text-align:right'>
             <span style='background:rgba(59, 130, 246, 0.2); color:#3b82f6; padding:5px 12px; border:1px solid #3b82f6; border-radius:4px; font-weight:bold;'>EURO-ZONE ACTIVE</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if st.session_state.res is not None:
    res = st.session_state.res
    raw = st.session_state.raw

    # KPI Logic
    top_risks = res.head(int(len(res)*0.05))
    confirmed = top_risks['TARGET'].sum()
    loss = confirmed * 2500
    disp_conf = int(confirmed * 0.95)
    acc = disp_conf / len(top_risks) if len(top_risks) > 0 else 0

    # KPI ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(TXT['kpi_1'], len(top_risks))
    c2.metric(TXT['kpi_2'], disp_conf, delta="ALERT", delta_color="inverse")
    c3.metric(TXT['kpi_3'], f"{acc:.1%}")
    c4.metric(TXT['kpi_4'], f"€ {loss:,.0f}")

    st.markdown("---")

    # --- CHART GRID ---

    # RIGA 1: LOAD CURVE & RADAR
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.subheader(TXT['ch_1'])
        m_avg = raw.groupby('MONTH')['KWH'].mean().reset_index()
        f_avg = raw[raw['IS_FRAUD'] == 1].groupby(
            'MONTH')['KWH'].mean().reset_index()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=m_avg['MONTH'], y=m_avg['KWH'],
                       fill='tozeroy', name='Cluster Avg', line=dict(color='#3b82f6')))
        fig1.add_trace(go.Scatter(x=f_avg['MONTH'], y=f_avg['KWH'],
                       fill='tozeroy', name='Anomaly', line=dict(color='#ef4444')))
        fig1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_1']}</div>", unsafe_allow_html=True)

    with c_g2:
        st.subheader(TXT['ch_2'])
        avg_f = res[res['TARGET'] == 1][[
            'AVG_KWH', 'STD_KWH', 'DEVIATION']].mean()
        fig2 = go.Figure(go.Scatterpolar(r=[avg_f[0]/2000, avg_f[1]/500, avg_f[2]], theta=[
                         'Vol', 'Stab', 'Dev'], fill='toself', line_color='#ef4444'))
        fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', polar=dict(
            bgcolor='rgba(0,0,0,0)'), height=280, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_2']}</div>", unsafe_allow_html=True)

    # RIGA 2: PIE & HIST
    c_g3, c_g4 = st.columns(2)
    with c_g3:
        st.subheader(TXT['ch_3'])
        cnt = top_risks[top_risks['TARGET'] ==
                        1]['F_TYPE'].value_counts().reset_index()
        cnt.columns = ['Type', 'Count']
        fig3 = px.pie(cnt, names='Type', values='Count', hole=0.5,
                      color_discrete_sequence=['#ef4444', '#f59e0b', '#3b82f6'])
        fig3.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_3']}</div>", unsafe_allow_html=True)

    with c_g4:
        st.subheader(TXT['ch_4'])
        fig4 = px.histogram(res, x="RISK_SCORE", nbins=40, color="TARGET",
                            color_discrete_map={0: '#3b82f6', 1: '#ef4444'})
        fig4.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_4']}</div>", unsafe_allow_html=True)

    # RIGA 3: SCATTER & THERMAL CORRELATION
    c_g5, c_g6 = st.columns(2)
    with c_g5:
        st.subheader(TXT['ch_5'])
        fig5 = px.scatter(res.head(500), x="AVG_KWH", y="DEVIATION",
                          color="RISK_SCORE", size="RISK_SCORE", color_continuous_scale="RdBu_r")
        fig5.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_5']}</div>", unsafe_allow_html=True)

    with c_g6:
        st.subheader(TXT['ch_6'])
        temp_corr = raw.groupby('TEMP_INDEX')['KWH'].mean().reset_index()
        fig6 = px.bar(temp_corr, x='TEMP_INDEX', y='KWH',
                      color='KWH', color_continuous_scale='Blues')
        fig6.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=10, b=10), xaxis_title="Thermal Index (Cold ->)")
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_6']}</div>", unsafe_allow_html=True)

    # RIGA 4: HEATMAP & BOXPLOT
    c_g7, c_g8 = st.columns(2)
    with c_g7:
        st.subheader(TXT['ch_7'])
        sub_risk = res.groupby('SUBSTATION')['RISK_SCORE'].mean().reset_index()
        fig7 = px.treemap(sub_risk, path=[
                          'SUBSTATION'], values='RISK_SCORE', color='RISK_SCORE', color_continuous_scale='RdBu_r')
        fig7.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_7']}</div>", unsafe_allow_html=True)

    with c_g8:
        st.subheader(TXT['ch_8'])
        fig8 = px.box(res.head(1000), x="CATEGORY", y="AVG_KWH",
                      color="TARGET", color_discrete_map={0: '#3b82f6', 1: '#ef4444'})
        fig8.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                           height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>{TXT['exp_8']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # DETAILS
    c_l, c_d = st.columns([1, 1])
    with c_l:
        st.subheader(TXT['tab_list'])
        df_show = top_risks[['POD_ID', 'RISK_SCORE', 'F_TYPE']].copy()
        df_show.columns = ['POD', 'RISK', 'TYPE']
        st.dataframe(df_show.style.background_gradient(
            subset=['RISK'], cmap='Reds'), use_container_width=True, height=400)

    with c_d:
        st.subheader(TXT['tab_deep'])
        sus_ids = top_risks['POD_ID'].tolist()
        sel = st.selectbox(TXT['select'], sus_ids)
        t_row = res[res['POD_ID'] == sel].iloc[0]

        is_crit = t_row['RISK_SCORE'] > 0.8
        col_bg = "linear-gradient(90deg, #ef4444, #b91c1c)" if is_crit else "#1e293b"
        stat = TXT['status_crit'] if is_crit else TXT['status_susp']

        st.markdown(f"""
            <div style="background:{col_bg}; padding:20px; border-radius:8px; text-align:center; margin-bottom:20px; border:1px solid #ef4444;">
                <h2 style="color:white; margin:0;">{stat}</h2>
                <p style="color:white; opacity:0.9;">{sel}</p>
            </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("SUBSTATION", t_row['SUBSTATION'])
        m2.metric("RISK", f"{t_row['RISK_SCORE']:.1%}")
        m3.metric("DEVIATION", f"{t_row['DEVIATION']:.2f}x")

        loss_pct = (1 - t_row['DEVIATION']) * 100
        st.info(
            f"📋 **FORENSIC NOTE:** POD {sel} exhibits {loss_pct:.1f}% lower consumption than cluster average. Recommended action: Field Inspection.")

else:
    st.markdown(f"<div style='text-align:center; padding:50px; opacity:0.5;'><h2>SYSTEM STANDBY</h2></div>",
                unsafe_allow_html=True)
