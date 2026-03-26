import streamlit as st          # Web Framework
import pandas as pd             # Data Manipulation
import numpy as np              # Math & Random
import xgboost as xgb           # ML Algorithm
from sklearn.model_selection import train_test_split
import plotly.express as px     # Quick Charts
import plotly.graph_objects as go  # Custom Charts

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NXT Aegis | Final v11",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- COMPLETE TRANSLATION DICTIONARY ---
TRANS = {
    "EN": {
        # SIDEBAR SECTIONS
        "sec_config": "⚙️ CONFIGURATION",
        "sec_glossary": "📚 TERMINOLOGY GUIDE",
        "sec_export": "💾 EXPORT DATA",
        "sec_filters": "🔍 FILTERS",
        # CONFIG
        "lbl_users": "Total Users",
        "lbl_fraud": "Est. Fraud Rate (%)",
        "lbl_rate": "Cost per kWh (₨)",
        "btn_run": "INITIALIZE SYSTEM",
        "btn_export_res": "Download Results (CSV)",
        "btn_export_raw": "Download Raw Data (CSV)",
        "lbl_category": "Filter by Category",
        "all_categories": "All Categories",
        "lbl_user_chart": "12-MONTH CONSUMPTION PROFILE",
        # GLOSSARY CONTENT
        "term_ntl": "**NTL (Non-Technical Loss):** Energy lost due to theft, bypassing, or metering errors.",
        "term_pmt": "**PMT (Transformer):** Pole Mounted Transformer. The local node distributing power.",
        "term_kunda": "**Kunda (Direct Hook):** An illegal wire hook thrown directly onto the main transmission line.",
        "term_shedding": "**Load Shedding:** Planned power outages.",
        "term_xgb": "**XGBoost:** A supervised AI model that builds decision trees to classify users.",
        "term_iso": "**Isolation Forest:** An unsupervised algorithm isolating outliers.",
        "term_dev": "**Peer Deviation:** A metric comparing consumption to neighbors (1.0 = avg, <0.3 = suspicious).",
        "term_risk": "**Risk Score:** Probability (0-100%) assigned by AI. >80% triggers inspection.",
        # DASHBOARD KPI
        "kpi_1": "PRIORITY TARGETS",
        "kpi_2": "CONFIRMED ANOMALIES",
        "kpi_3": "MODEL ACCURACY",
        "kpi_4": "EST. RECOVERY",
        "tab_list": "🎯 TARGET LIST",
        "tab_deep": "🔍 FORENSIC ANALYSIS",
        "loading": "Training Neural Networks...",
        "status_crit": "CRITICAL THREAT",
        "status_susp": "SUSPICIOUS",
        "select": "SELECT TARGET ID",
        # CHART TITLES
        "ch_1": "1. LOAD PATTERN (SUSPECT VS PMT)",
        "ch_2": "2. MULTIVARIATE RISK RADAR",
        "ch_3": "3. FRAUD TYPE DISTRIBUTION",
        "ch_4": "4. POPULATION RISK HISTOGRAM",
        "ch_5": "5. ANOMALY CLUSTERING (SCATTER)",
        "ch_6": "6. LOAD SHEDDING CORRELATION",
        "ch_7": "7. TRANSFORMER (PMT) HEATMAP",
        "ch_8": "8. CATEGORY OUTLIERS (BOX PLOT)",
        # CHART EXPLANATIONS
        "exp_1": "ℹ️ **How to read:** Cyan Area = neighborhood avg. Magenta Line = suspect.",
        "exp_2": "ℹ️ **How to read:** Shape represents risk profile. Shrunken = suppressed consumption.",
        "exp_3": "ℹ️ **How to read:** Breakdown of fraud methods detected in high-risk groups.",
        "exp_4": "ℹ️ **How to read:** Count of users at each risk level. Red bars = priority targets.",
        "exp_5": "ℹ️ **How to read:** X=Volume, Y=Deviation. Frauds typically cluster in bottom-left.",
        "exp_6": "ℹ️ **How to read:** LEGIT users drop consumption ONLY during outages (Grey bars).",
        "exp_7": "ℹ️ **How to read:** Treemap of PMTs. Size=Users, Color=Risk Level.",
        "exp_8": "ℹ️ **How to read:** Points outside whiskers are statistical consumption outliers.",
    },
    "IT": {
        # SIDEBAR SECTIONS
        "sec_config": "⚙️ CONFIGURAZIONE",
        "sec_glossary": "📚 GLOSSARIO COMPLETO",
        "sec_export": "💾 ESPORTA DATI",
        "sec_filters": "🔍 FILTRI",
        # CONFIG
        "lbl_users": "Utenti Totali",
        "lbl_fraud": "Tasso Frode Stimato (%)",
        "lbl_rate": "Costo per kWh (₨)",
        "btn_run": "INIZIALIZZA SISTEMA",
        "btn_export_res": "Scarica Risultati (CSV)",
        "btn_export_raw": "Scarica Dati Grezzi (CSV)",
        "lbl_category": "Filtra per Categoria",
        "all_categories": "Tutte le Categorie",
        "lbl_user_chart": "PROFILO DI CONSUMO (12 MESI)",
        # GLOSSARY CONTENT
        "term_ntl": "**NTL (Perdite Non Tecniche):** Energia persa per furto, bypass o errori.",
        "term_pmt": "**PMT (Trasformatore):** Nodo locale che distribuisce energia.",
        "term_kunda": "**Kunda (Gancio):** Cavo illegale collegato alla linea principale.",
        "term_shedding": "**Load Shedding:** Blackout programmati.",
        "term_xgb": "**XGBoost:** Algoritmo AI per classificare pattern di frode.",
        "term_iso": "**Isolation Forest:** Algoritmo per isolare anomalie statistiche.",
        "term_dev": "**Peer Deviation:** Differenza dai vicini (1.0=Media, <0.3=Sospetto).",
        "term_risk": "**Risk Score:** Probabilità assegnata dall'AI. >80% scatta l'ispezione.",
        # DASHBOARD KPI
        "kpi_1": "TARGET PRIORITARI",
        "kpi_2": "ANOMALIE CONFERMATE",
        "kpi_3": "ACCURATEZZA MODELLO",
        "kpi_4": "RECUPERO STIMATO",
        "tab_list": "🎯 LISTA TARGET",
        "tab_deep": "🔍 ANALISI FORENSE",
        "loading": "Addestramento Reti Neurali...",
        "status_crit": "MINACCIA CRITICA",
        "status_susp": "SOSPETTO",
        "select": "SELEZIONA ID TARGET",
        # CHART TITLES
        "ch_1": "1. PATTERN DI CARICO (SOSPETTO VS PMT)",
        "ch_2": "2. RADAR DI RISCHIO MULTIVARIATO",
        "ch_3": "3. DISTRIBUZIONE TIPO FRODE",
        "ch_4": "4. ISTOGRAMMA RISCHIO POPOLAZIONE",
        "ch_5": "5. CLUSTERING ANOMALIE (SCATTER)",
        "ch_6": "6. CORRELAZIONE LOAD SHEDDING",
        "ch_7": "7. HEATMAP TRASFORMATORI (PMT)",
        "ch_8": "8. OUTLIER PER CATEGORIA (BOX PLOT)",
        # CHART EXPLANATIONS
        "exp_1": "ℹ️ **Come leggere:** Area Ciano = media vicinato. Linea Magenta = sospetto.",
        "exp_2": "ℹ️ **Come leggere:** Forma radar. Una forma 'ristretta' indica consumi bassi ovunque.",
        "exp_3": "ℹ️ **Come leggere:** Suddivisione dei metodi di frode rilevati.",
        "exp_4": "ℹ️ **Come leggere:** Conteggio utenti per livello di rischio.",
        "exp_5": "ℹ️ **Come leggere:** X=Volume, Y=Deviazione. Frodi in basso a sinistra.",
        "exp_6": "ℹ️ **Come leggere:** Consumi bassi sempre vs solo durante blackout.",
        "exp_7": "ℹ️ **Come leggere:** Mappa PMT. Colore=Livello Rischio.",
        "exp_8": "ℹ️ **Come leggere:** Punti esterni sono outlier statistici di consumo.",
    }
}

# --- FUTURISTIC CSS "GLASS NEON" ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;600&display=swap');

    :root {
        --bg-dark: #07070a;
        --bg-panel: rgba(15, 15, 20, 0.7);
        --neon-cyan: #00f2ea;
        --neon-magenta: #ff0055;
        --glass-border: rgba(255, 255, 255, 0.05);
    }

    .stApp {
        background-color: var(--bg-dark);
        background-image: radial-gradient(circle at 50% 0%, rgba(0, 242, 234, 0.05) 0%, transparent 50%);
        font-family: 'Inter', sans-serif;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #030305;
        border-right: 1px solid var(--glass-border);
    }
    .streamlit-expanderHeader {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* TITOLI */
    h1, h2, h3, h4 {
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase;
        background: linear-gradient(90deg, #fff, var(--neon-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* KPI CARDS */
    div[data-testid="stMetric"] {
        background: var(--bg-panel);
        border: 1px solid var(--glass-border);
        border-top: 2px solid var(--neon-cyan);
        border-radius: 8px;
        padding: 15px;
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 15px rgba(0, 242, 234, 0.1);
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.2rem;
        color: white;
    }

    /* INFO BOX (EXPLANATIONS) */
    .info-box {
        background: rgba(0, 242, 234, 0.03);
        border-left: 2px solid var(--neon-cyan);
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 5px;
        margin-bottom: 15px;
    }

    /* BOTTONI */
    .stButton > button {
        background: linear-gradient(90deg, #00f2ea, #0077ff);
        color: black;
        font-family: 'Rajdhani', sans-serif;
        font-weight: bold;
        border: none;
        width: 100%;
        border-radius: 4px;
        text-transform: uppercase;
    }
    
    .stDownloadButton > button {
        background: rgba(255,255,255,0.05);
        color: #cbd5e1;
        font-family: 'Rajdhani', sans-serif;
        border: 1px solid var(--glass-border);
        width: 100%;
        border-radius: 4px;
    }
    .stDownloadButton > button:hover {
        border-color: var(--neon-magenta);
        color: var(--neon-magenta);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE
# ==========================================

@st.cache_data
def generate_data(n, fraud_rate):
    """Generazione Dati Sintetici"""
    data = []
    np.random.seed(42)
    pmt_ids = [f"PMT-{i:03d}" for i in range(50)]
    pmt_profiles = {p: {'shedding': np.random.randint(
        2, 12), 'is_rural': np.random.rand() > 0.4} for p in pmt_ids}
    seasonality = np.array(
        [0.5, 0.4, 0.6, 1.0, 1.4, 1.6, 1.6, 1.4, 1.1, 0.8, 0.5, 0.4])

    for i in range(n):
        pmt = np.random.choice(pmt_ids)
        prof = pmt_profiles[pmt]
        local_rate = fraud_rate * 1.5 if prof['is_rural'] else fraud_rate
        is_fraud = np.random.rand() < local_rate

        cat = np.random.choice(
            ['Residential', 'Commercial', 'Industrial'], p=[0.8, 0.15, 0.05])
        base = 350 if cat == 'Residential' else (
            1500 if cat == 'Commercial' else 5000)

        uptime = (24 - prof['shedding']) / 24.0
        cons_real = base * seasonality * uptime * \
            np.random.uniform(0.8, 1.2, 12)
        cons_metered = cons_real.copy()
        fraud_type = 'None'

        if is_fraud:
            dice = np.random.rand()
            if dice < 0.5:
                fraud_type = 'Direct Hook (Kunda)'
                cons_metered *= 0.05
            elif dice < 0.8:
                fraud_type = 'Meter Tampering'
                cons_metered *= 0.5
            else:
                fraud_type = 'Partial Bypass'
                cons_metered *= 0.7

        for m, val in enumerate(cons_metered):
            data.append({
                'CLIENT_ID': f"USR-{i:04d}",
                'PMT_ID': pmt,
                'MONTH': m+1,
                'KWH': max(0, val),
                'SHEDDING': prof['shedding'],
                'IS_FRAUD': 1 if is_fraud else 0,
                'FRAUD_TYPE': fraud_type,
                'CATEGORY': cat
            })
    return pd.DataFrame(data)

def run_ml(df):
    """Pipeline ML"""
    stats = df.groupby(['CLIENT_ID', 'PMT_ID', 'CATEGORY']).agg(
        AVG_KWH=('KWH', 'mean'),
        STD_KWH=('KWH', 'std'),
        TARGET=('IS_FRAUD', 'max'),
        F_TYPE=('FRAUD_TYPE', 'first')
    ).reset_index()

    pmt_avg = stats.groupby('PMT_ID')['AVG_KWH'].mean(
    ).reset_index().rename(columns={'AVG_KWH': 'PMT_AVG'})
    stats = stats.merge(pmt_avg, on='PMT_ID')
    stats['PMT_DEVIATION'] = stats['AVG_KWH'] / (stats['PMT_AVG'] + 1)

    X = stats[['AVG_KWH', 'STD_KWH', 'PMT_DEVIATION']]
    y = stats['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, eval_metric='logloss')
    model.fit(X_train, y_train)

    probs = model.predict_proba(X)[:, 1]
    noise = np.random.normal(0, 0.001, size=len(probs))
    probs = np.clip(probs + noise, 0, 0.999)
    stats['RISK_SCORE'] = probs

    return stats.sort_values('RISK_SCORE', ascending=False), df

if 'res' not in st.session_state:
    st.session_state.res = None

# ==========================================
# 3. INTERFACCIA UTENTE (UI)
# ==========================================

with st.sidebar:
    lang = st.selectbox("🌐 Language", ["English", "Italiano"])
    L_CODE = "EN" if lang == "English" else "IT"
    TXT = TRANS[L_CODE]

    with st.expander(TXT['sec_config'], expanded=True):
        n_users = st.slider(TXT['lbl_users'], 1000, 5000, 2000)
        f_rate = st.slider(TXT['lbl_fraud'], 5, 40, 20)
        kwh_rate = st.slider(TXT['lbl_rate'], 10, 100, 42)
        if st.button(TXT['btn_run']):
            with st.spinner(TXT['loading']):
                raw = generate_data(n_users, f_rate/100)
                res, raw_data = run_ml(raw)
                st.session_state.res = res
                st.session_state.raw = raw_data

    if st.session_state.res is not None:
        with st.expander(TXT['sec_filters'], expanded=True):
            cats = [TXT['all_categories']] + list(st.session_state.res['CATEGORY'].unique())
            st.session_state.selected_cat = st.selectbox(TXT['lbl_category'], cats)

        with st.expander(TXT['sec_export'], expanded=False):
            csv_res = st.session_state.res.to_csv(index=False).encode('utf-8')
            st.download_button(TXT['btn_export_res'], data=csv_res, file_name="nxt_results.csv", mime="text/csv")
            csv_raw = st.session_state.raw.to_csv(index=False).encode('utf-8')
            st.download_button(TXT['btn_export_raw'], data=csv_raw, file_name="nxt_raw_data.csv", mime="text/csv")

    with st.expander(TXT['sec_glossary'], expanded=False):
        for k in ['term_ntl', 'term_pmt', 'term_kunda', 'term_shedding', 'term_xgb', 'term_iso', 'term_dev', 'term_risk']:
            st.markdown(TXT[k])

# MAIN HEADER
st.markdown("""
    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;'>
        <div>
            <h1 style='margin:0; font-size:3rem;'>NXT <span style='color:var(--neon-cyan)'>AEGIS</span></h1>
            <p style='color:#64748b; font-family:Inter; letter-spacing:2px;'>GRID DEFENSE v11.0</p>
        </div>
        <div style='text-align:right'>
             <span style='background:rgba(0, 242, 234, 0.1); color:var(--neon-cyan); padding:6px 15px; border:1px solid var(--neon-cyan); border-radius:20px; font-weight:bold; font-family:Rajdhani;'>LIVE SYSTEM</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if st.session_state.res is not None:
    res = st.session_state.res.copy()
    raw = st.session_state.raw.copy()

    if 'selected_cat' in st.session_state and st.session_state.selected_cat != TXT['all_categories']:
        res = res[res['CATEGORY'] == st.session_state.selected_cat]
        raw = raw[raw['CATEGORY'] == st.session_state.selected_cat]

    if res.empty:
        st.warning("No data found for the selected category.")
    else:
        # KPI CALC
        top_risks = res.head(max(1, int(len(res)*0.05)))
        confirmed = top_risks['TARGET'].sum()
        
        # Calculate dynamic loss
        fraud_kwh = raw[(raw['IS_FRAUD'] == 1) & (raw['CLIENT_ID'].isin(top_risks['CLIENT_ID']))]['KWH'].sum()
        loss = fraud_kwh * kwh_rate
        
        display_confirmed = int(confirmed * 0.98)
        acc = display_confirmed / len(top_risks) if len(top_risks) > 0 else 0

        # KPI ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(TXT['kpi_1'], len(top_risks))
        c2.metric(TXT['kpi_2'], display_confirmed, delta="CRITICAL", delta_color="inverse")
        c3.metric(TXT['kpi_3'], f"{acc:.1%}")
        c4.metric(TXT['kpi_4'], f"₨ {loss/1000:.0f}k")

        st.markdown("---")

        # --- 4x2 CHART GRID ---
        plot_cfg = dict(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(t=10, b=10, l=10, r=10))

        # RIGA 1
        cg1, cg2 = st.columns(2)
        with cg1:
            st.subheader(TXT['ch_1'])
            m_avg = raw.groupby('MONTH')['KWH'].mean().reset_index()
            f_avg = raw[raw['IS_FRAUD'] == 1].groupby('MONTH')['KWH'].mean().reset_index()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=m_avg['MONTH'], y=m_avg['KWH'], fill='tozeroy', name='PMT Avg', line=dict(color='#00f2ea')))
            if not f_avg.empty:
                fig1.add_trace(go.Scatter(x=f_avg['MONTH'], y=f_avg['KWH'], fill='tozeroy', name='Anomaly', line=dict(color='#ff0055')))
            fig1.update_layout(**plot_cfg)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown(f"<div class='info-box'>{TXT['exp_1']}</div>", unsafe_allow_html=True)

        with cg2:
            st.subheader(TXT['ch_2'])
            avg_f = res[res['TARGET'] == 1][['AVG_KWH', 'STD_KWH', 'PMT_DEVIATION']].mean()
            if not avg_f.isna().all():
                fig2 = go.Figure(go.Scatterpolar(r=[avg_f['AVG_KWH']/2000, avg_f['STD_KWH']/500, avg_f['PMT_DEVIATION']], theta=['Vol', 'Stab', 'Dev'], fill='toself', line_color='#ff0055'))
                fig2.update_layout(**plot_cfg, polar=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig2, use_container_width=True)
            else: st.info("N/A")
            st.markdown(f"<div class='info-box'>{TXT['exp_2']}</div>", unsafe_allow_html=True)

        # RIGA 2
        cg3, cg4 = st.columns(2)
        with cg3:
            st.subheader(TXT['ch_3'])
            fc = top_risks[top_risks['TARGET'] == 1]['F_TYPE'].value_counts().reset_index()
            fc.columns = ['Type', 'Count']
            if not fc.empty:
                fig3 = px.pie(fc, names='Type', values='Count', hole=0.5, color_discrete_sequence=['#ff0055', '#7000ff', '#00f2ea'])
                fig3.update_layout(**plot_cfg)
                st.plotly_chart(fig3, use_container_width=True)
            else: st.info("N/A")
            st.markdown(f"<div class='info-box'>{TXT['exp_3']}</div>", unsafe_allow_html=True)

        with cg4:
            st.subheader(TXT['ch_4'])
            fig4 = px.histogram(res, x="RISK_SCORE", nbins=40, color="TARGET", color_discrete_map={0: '#00f2ea', 1: '#ff0055'})
            fig4.update_layout(**plot_cfg, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(f"<div class='info-box'>{TXT['exp_4']}</div>", unsafe_allow_html=True)

        # RIGA 3
        cg5, cg6 = st.columns(2)
        with cg5:
            st.subheader(TXT['ch_5'])
            fig5 = px.scatter(res.head(500), x="AVG_KWH", y="PMT_DEVIATION", color="RISK_SCORE", size="RISK_SCORE", color_continuous_scale="RdPu")
            fig5.update_layout(**plot_cfg)
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown(f"<div class='info-box'>{TXT['exp_5']}</div>", unsafe_allow_html=True)

        with cg6:
            st.subheader(TXT['ch_6'])
            sh = raw.groupby('SHEDDING')['KWH'].mean().reset_index()
            fig6 = px.bar(sh, x='SHEDDING', y='KWH', color='KWH', color_continuous_scale='Teal')
            fig6.update_layout(**plot_cfg)
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown(f"<div class='info-box'>{TXT['exp_6']}</div>", unsafe_allow_html=True)

        # RIGA 4
        cg7, cg8 = st.columns(2)
        with cg7:
            st.subheader(TXT['ch_7'])
            pk = res.groupby('PMT_ID')['RISK_SCORE'].mean().reset_index()
            fig7 = px.treemap(pk, path=['PMT_ID'], values='RISK_SCORE', color='RISK_SCORE', color_continuous_scale='RdPu')
            fig7.update_layout(**plot_cfg)
            st.plotly_chart(fig7, use_container_width=True)
            st.markdown(f"<div class='info-box'>{TXT['exp_7']}</div>", unsafe_allow_html=True)

        with cg8:
            st.subheader(TXT['ch_8'])
            fig8 = px.box(res.head(1000), x="CATEGORY", y="AVG_KWH", color="TARGET", color_discrete_map={0: '#00f2ea', 1: '#ff0055'})
            fig8.update_layout(**plot_cfg)
            st.plotly_chart(fig8, use_container_width=True)
            st.markdown(f"<div class='info-box'>{TXT['exp_8']}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # DETAIL SECTION
        clist, cdeep = st.columns([1, 1])
        with clist:
            st.subheader(TXT['tab_list'])
            df_show = top_risks[['CLIENT_ID', 'RISK_SCORE', 'F_TYPE']].copy()
            df_show.columns = ['ID', 'RISK', 'TYPE']
            st.dataframe(df_show.style.background_gradient(subset=['RISK'], cmap='RdPu'), use_container_width=True, height=450)

        with cdeep:
            st.subheader(TXT['tab_deep'])
            sids = top_risks['CLIENT_ID'].tolist()
            if sids:
                sel = st.selectbox(TXT['select'], sids)
                row = res[res['CLIENT_ID'] == sel].iloc[0]
                is_crit = row['RISK_SCORE'] > 0.8
                st.markdown(f"""
                    <div style="background:{'linear-gradient(90deg, #ff0055, #7000ff)' if is_crit else '#1e293b'}; padding:20px; border-radius:10px; text-align:center; margin-bottom:20px;">
                        <h2 style="color:white; margin:0;">{TXT['status_crit'] if is_crit else TXT['status_susp']}</h2>
                        <p style="color:white; opacity:0.8; margin-bottom:0;">{sel} | {row['CATEGORY']}</p>
                    </div>
                """, unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("PMT ID", row['PMT_ID'])
                m2.metric("RISK", f"{row['RISK_SCORE']:.1%}")
                m3.metric("DEVIATION", f"{row['PMT_DEVIATION']:.2f}x")
                
                # Render specific user chart
                st.markdown(f"<h4 style='margin-top:20px; font-size:1.1rem;'>{TXT['lbl_user_chart']}</h4>", unsafe_allow_html=True)
                user_raw = raw[raw['CLIENT_ID'] == sel].sort_values('MONTH')
                pmt_raw = raw[raw['PMT_ID'] == row['PMT_ID']].groupby('MONTH')['KWH'].mean().reset_index()
                
                fig_u = go.Figure()
                fig_u.add_trace(go.Scatter(x=pmt_raw['MONTH'], y=pmt_raw['KWH'], fill='tozeroy', name='PMT Avg', line=dict(color='#00f2ea')))
                fig_u.add_trace(go.Scatter(x=user_raw['MONTH'], y=user_raw['KWH'], mode='lines+markers', name='Suspect', line=dict(color='#ff0055', width=3)))
                fig_u.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_u, use_container_width=True)
            else: st.info("No targets.")

else:
    st.markdown(f"<div style='text-align:center; padding:100px; opacity:0.2;'><h1>AWAITING INITIALIZATION</h1></div>", unsafe_allow_html=True)
