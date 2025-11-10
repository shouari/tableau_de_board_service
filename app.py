import os
import io
import json
import re
import difflib
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
import openai


st.set_page_config(page_title="SAV KPI Dashboard", page_icon="📊", layout="wide")
st.title("📊 Service Calls — KPI Dashboard")
st.caption("Analyse et classification des tickets SAV, avec OpenAI GPT-4o")

openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

# -----------------------
# 📦 Cache persistant à 2 niveaux
# -----------------------
CACHE_PATH = Path("classification_cache.json")
REPORT_CACHE_PATH = Path("report_cache.json")

def _load_cache() -> dict:
    if "CLASS_CACHE" in st.session_state and isinstance(st.session_state.CLASS_CACHE, dict):
        return st.session_state.CLASS_CACHE
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                st.session_state.CLASS_CACHE = json.load(f)
                return st.session_state.CLASS_CACHE
        except Exception:
            pass
    st.session_state.CLASS_CACHE = {}
    return st.session_state.CLASS_CACHE

def _save_cache(cache: dict) -> None:
    st.session_state.CLASS_CACHE = cache
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _load_report_cache() -> dict:
    if "REPORT_CACHE" in st.session_state:
        return st.session_state.REPORT_CACHE
    if REPORT_CACHE_PATH.exists():
        try:
            with REPORT_CACHE_PATH.open("r", encoding="utf-8") as f:
                st.session_state.REPORT_CACHE = json.load(f)
                return st.session_state.REPORT_CACHE
        except Exception:
            pass
    st.session_state.REPORT_CACHE = {}
    return st.session_state.REPORT_CACHE

def _save_report_cache(cache: dict) -> None:
    st.session_state.REPORT_CACHE = cache
    try:
        with REPORT_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _compute_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

CLASS_CACHE = _load_cache()
REPORT_CACHE = _load_report_cache()

def _cache_key_for_text(text: str) -> str:
    norm = (text or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

# -----------------------
# Taxonomie et règles locales
# -----------------------
TAXO_FR = {
    "type_probleme": [
        "Hors ligne", "Pas de son", "Pas d'image", "Pas de contrôle",
        "Programmation", "Installation/Remplacement", "Mécanique/Batterie",
        "Logiciel/App", "Autre"
    ],
    "categorie": [
        "Audio", "Vidéo", "Éclairage", "Réseau", "Sécurité",
        "Stores", "Système de contrôle", "Autres"
    ],
    "systeme": [
        "Control4", "Unifi", "Lutron", "Somfy", "Hikvision", "Clare", "CDVI",
        "Apple TV", "Helix", "Sonos", "QSC", "MyQ", "Polycom", "NAS", "Générique", "Non spécifié"
    ],
}

KEYWORDS_FR = {
    "type_probleme": {
        "offline": "Hors ligne", "hors ligne": "Hors ligne", "pas d'internet": "Hors ligne", "no internet": "Hors ligne",
        "pas de son": "Pas de son", "no sound": "Pas de son",
        "pas d'image": "Pas d'image", "no image": "Pas d'image",
        "ne fonctionne pas": "Pas de contrôle", "ne marche pas": "Pas de contrôle", "contrôle": "Pas de contrôle",
        "programmation": "Programmation", "ajustement": "Programmation",
        "installer": "Installation/Remplacement", "installation": "Installation/Remplacement", "remplacer": "Installation/Remplacement",
        "batterie": "Mécanique/Batterie", "défectueux": "Mécanique/Batterie", "defectueux": "Mécanique/Batterie",
        "app ": "Logiciel/App", "application": "Logiciel/App", "login": "Logiciel/App", "mot de passe": "Logiciel/App",
    },
    "categorie": {
        "audio": "Audio", "son": "Audio", "subwoofer": "Audio", "ampli": "Audio", "qsc": "Audio", "sonos": "Audio",
        "vidéo": "Vidéo", "video": "Vidéo", "tv": "Vidéo", "hdmi": "Vidéo", "splitter": "Vidéo",
        "apple tv": "Vidéo", "helix": "Vidéo", "polycom": "Vidéo",
        "éclairage": "Éclairage", "eclairage": "Éclairage", "lumière": "Éclairage", "scene": "Éclairage", "scène": "Éclairage",
        "réseau": "Réseau", "reseau": "Réseau", "internet": "Réseau", "wifi": "Réseau", "unifi": "Réseau", "usg": "Réseau", "uxg": "Réseau",
        "sécurité": "Sécurité", "securite": "Sécurité", "caméra": "Sécurité", "camera": "Sécurité", "hikvision": "Sécurité", "alarme": "Sécurité", "cdvi": "Sécurité", "myq": "Sécurité", "clare": "Sécurité",
        "store": "Stores", "toile": "Stores", "lutron": "Stores", "somfy": "Stores",
        "control4": "Système de contrôle", "domotique": "Système de contrôle",
    },
    "systeme": {
        "control4": "Control4", "unifi": "Unifi", "usg": "Unifi", "uxg": "Unifi",
        "lutron": "Lutron", "somfy": "Somfy",
        "hikvision": "Hikvision", "clare": "Clare", "cdvi": "CDVI",
        "apple tv": "Apple TV", "helix": "Helix", "sonos": "Sonos",
        "qsc": "QSC", "myq": "MyQ", "polycom": "Polycom", "nas": "NAS",
    },
}

# -----------------------
# Utils
# -----------------------
def clean_html(text):
    if not isinstance(text, str):
        return ""
    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)

def json_to_df(file_like):
    data = json.load(file_like)
    rows = data.get("serviceCalls", data) if isinstance(data, dict) else data
    df = pd.DataFrame(rows)
    
    expected_cols = ["number", "client", "issueReported", "price", "createdOn", "stateName", "paymentBillingDate"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    
    df["issue_text"] = df["issueReported"].apply(clean_html)
    df["createdOn"] = pd.to_datetime(df["createdOn"], errors="coerce")
    df["paymentBillingDate"] = pd.to_datetime(df["paymentBillingDate"], errors="coerce")
    
    # Calcul du délai de clôture (en jours) - SEULEMENT pour les SC "Completed"
    df["closure_days"] = None
    completed_mask = df["stateName"].astype(str).str.lower() == "completed"
    df.loc[completed_mask, "closure_days"] = (
        df.loc[completed_mask, "paymentBillingDate"] - df.loc[completed_mask, "createdOn"]
    ).dt.days
    
    df.rename(columns={"number": "sc_number"}, inplace=True)
    month_period = df["createdOn"].dt.to_period("M")
    df["month_label"] = month_period.astype(str)
    df.loc[month_period.isna(), "month_label"] = "Sans date"
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    return df

# -----------------------
# Classification (robuste)
# -----------------------

def _snap(value: str, allowed: list[str]) -> str:
    v = (value or "").strip()
    if v in allowed:
        return v
    cand = difflib.get_close_matches(v.lower(), [a.lower() for a in allowed], n=1, cutoff=0.75)
    if cand:
        idx = [a.lower() for a in allowed].index(cand[0])
        return allowed[idx]
    return "Non spécifié" if "Non spécifié" in allowed else allowed[-1]

_BRAND_REGEX = re.compile(
    r"\b(c4|control4|core[135]|ea-?[135]|unifi|usg|udm|uxg|sonos|hikvision|caseta|homeworks|lutron|somfy|qsc|apc|polycom|myq|helix|apple\s*tv|nas)\b",
    flags=re.IGNORECASE,
)
_BRAND_CANON = {
    "control4": "Control4", "c4": "Control4", "core1": "Control4", "core3": "Control4", "core5": "Control4",
    "ea1": "Control4", "ea-1": "Control4", "ea3": "Control4", "ea-3": "Control4", "ea5": "Control4", "ea-5": "Control4",
    "unifi": "Unifi", "usg": "Unifi", "udm": "Unifi", "uxg": "Unifi",
    "sonos": "Sonos", "hikvision": "Hikvision", "caseta": "Lutron", "homeworks": "Lutron", "lutron": "Lutron",
    "somfy": "Somfy", "qsc": "QSC", "apc": "APC", "polycom": "Polycom",
    "myq": "MyQ", "helix": "Helix", "apple tv": "Apple TV", "nas": "NAS",
}

def _explicit_brand(text: str) -> str | None:
    m = _BRAND_REGEX.search(text or "")
    if not m:
        return None
    key = m.group(1).lower().replace(" ", "")
    key = "apple tv" if key.startswith("apple") else key
    return _BRAND_CANON.get(key)

def _keyword_score(text: str, table: dict[str, dict[str, str]]) -> dict[str, dict[str, int]]:
    t = (text or "").lower()
    scores = {"type_probleme": {}, "categorie": {}, "systeme": {}}
    for field in ["type_probleme", "categorie", "systeme"]:
        for k, v in table[field].items():
            if k in t:
                scores[field][v] = scores[field].get(v, 0) + 1
    return scores

def _prompt_json(text: str) -> str:
    return f"""
Tu es un classificateur SAV. Réponds UNIQUEMENT par un JSON valide, sans texte autour.

Contraintes:
- "type_probleme" ∈ {TAXO_FR["type_probleme"]}
- "categorie" ∈ {TAXO_FR["categorie"]}
- "systeme" ∈ {TAXO_FR["systeme"]} (si aucune marque explicite, mets "Non spécifié")
- "systeme_suggere" ∈ {TAXO_FR["systeme"]} ou null (si "systeme" = "Non spécifié", propose la plus probable; sinon null)
- "confiance_systeme" ∈ [0.0, 1.0]
- "justification_systeme": phrase courte (≤12 mots)

Maintenant, classe ce texte:
\"\"\"{text}\"\"\"
""".strip()

def classify_service_call_gpt(issue_text: str) -> dict:
    text = (issue_text or "").strip()
    if not text:
        return {
            "type_probleme": "Autre",
            "categorie": "Autres",
            "systeme": "Non spécifié",
            "systeme_suggere": None,
            "confiance_systeme": 0.0,
            "justification_systeme": "",
        }

    key = _cache_key_for_text(text)
    if key in CLASS_CACHE:
        return CLASS_CACHE[key]

    explicit = _explicit_brand(text)
    explicit_system = explicit if explicit in TAXO_FR["systeme"] else None

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Tu rends uniquement un JSON valide respectant strictement les contraintes."},
                {"role": "user", "content": _prompt_json(text)},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Le JSON précédent était invalide. Renvoie un JSON strict, sans autre texte."},
                {"role": "user", "content": _prompt_json(text)},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")

    tp = _snap(data.get("type_probleme", ""), TAXO_FR["type_probleme"])
    cat = _snap(data.get("categorie", ""), TAXO_FR["categorie"])

    if explicit_system:
        sys_off = explicit_system
        sys_sugg = None
        conf = max(float(data.get("confiance_systeme", 0.0) or 0.0), 0.9)
        just = "marque explicite"
    else:
        sys_off = _snap(data.get("systeme", "Non spécifié"), TAXO_FR["systeme"])
        if sys_off != "Non spécifié":
            sys_sugg = None
            conf = float(data.get("confiance_systeme", 0.0) or 0.0)
            just = (data.get("justification_systeme") or "")[:120]
        else:
            s_sugg = _snap(data.get("systeme_suggere") or "", TAXO_FR["systeme"])
            sys_sugg = None if s_sugg == "Non spécifié" else s_sugg
            conf = float(data.get("confiance_systeme", 0.0) or 0.0)
            just = (data.get("justification_systeme") or "")[:120]

    scores = _keyword_score(text, KEYWORDS_FR)
    if scores["type_probleme"]:
        tp = max(scores["type_probleme"], key=scores["type_probleme"].get)
    if scores["categorie"]:
        cat = max(scores["categorie"], key=scores["categorie"].get)
    if not explicit_system and sys_off == "Non spécifié" and scores["systeme"]:
        best_sys = max(scores["systeme"], key=scores["systeme"].get)
        if best_sys in TAXO_FR["systeme"]:
            sys_sugg = best_sys
            conf = max(conf, 0.7)
            if not just:
                just = "mot-clé système détecté"

    result = {
        "type_probleme": tp,
        "categorie": cat,
        "systeme": sys_off,
        "systeme_suggere": sys_sugg,
        "confiance_systeme": round(float(conf), 2),
        "justification_systeme": just,
    }

    CLASS_CACHE[key] = result
    _save_cache(CLASS_CACHE)

    return result

# -----------------------
# Pipeline avec cache de rapport
# -----------------------
def classify_all(df: pd.DataFrame, file_hash: str) -> pd.DataFrame:
    if file_hash in REPORT_CACHE:
        cached_data = REPORT_CACHE[file_hash]
        st.success("✅ Rapport chargé depuis le cache (0 appel API)")
        return pd.DataFrame(cached_data)
    
    rows = []
    progress_bar = st.progress(0)
    total = len(df)
    
    for idx, txt in enumerate(df["issue_text"].fillna("")):
        rows.append(classify_service_call_gpt(txt))
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    
    result_df = pd.DataFrame(rows)
    
    REPORT_CACHE[file_hash] = result_df.to_dict('records')
    _save_report_cache(REPORT_CACHE)
    
    return result_df

# -----------------------
# Interface Streamlit
# -----------------------
with st.sidebar:
    st.header("📥 Données")
    uploaded = st.file_uploader("Téléverser les données en format JSON", type=["json"])
    show_heatmap = st.toggle("Heatmap Type × Catégorie", True)

    st.divider()
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Cache textes"):
            CLASS_CACHE.clear()
            _save_cache(CLASS_CACHE)
            st.success("✓")
    
    with col_b:
        if st.button("🗑️ Cache rapports"):
            REPORT_CACHE.clear()
            _save_report_cache(REPORT_CACHE)
            st.success("✓")
    
    st.caption(f"📊 {len(CLASS_CACHE)} textes en cache")
    st.caption(f"📁 {len(REPORT_CACHE)} rapports en cache")

if not uploaded:
    st.info("➡️ Charge un fichier JSON récupéré depuis l'API de D-Tools.")
    st.stop()

file_content = uploaded.getvalue()
file_hash = _compute_file_hash(file_content)
uploaded.seek(0)

df_raw = json_to_df(uploaded)
if df_raw.empty:
    st.warning("Aucune donnée trouvée.")
    st.stop()

st.info("⏳ Traitement en cours...")
df_cls = classify_all(df_raw, file_hash)
df = pd.concat([df_raw, df_cls], axis=1)

# -----------------------
# KPI & Graphiques
# -----------------------
total_tickets = len(df)
completed_df = df[df["stateName"].astype(str).str.lower() == "completed"]
avg_cost = float(completed_df["price"].mean()) if not completed_df.empty else 0.0

completed_with_dates = completed_df[completed_df["closure_days"].notna() & (completed_df["closure_days"] >= 0)]
avg_closure_days = float(completed_with_dates["closure_days"].mean()) if not completed_with_dates.empty else 0.0

hors_ligne_pct = 100 * df["type_probleme"].eq("Hors ligne").mean() if total_tickets else 0.0
control4_pct = 100 * df["systeme"].eq("Control4").mean() if total_tickets else 0.0
unifi_pct = 100 * df["systeme"].eq("Unifi").mean() if total_tickets else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Tickets totaux", total_tickets)
c2.metric("Hors ligne", f"{hors_ligne_pct:.1f}%")
c3.metric("Control4", f"{control4_pct:.1f}%")
c4.metric("Unifi", f"{unifi_pct:.1f}%")
c5.metric("Coût moyen", f"{avg_cost:,.0f} $")
c6.metric("Délai clôture", f"{avg_closure_days:.1f}j")

st.divider()

if not df.empty:
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names="systeme", title="Répartition des systèmes")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        cat_count = (
            df["categorie"]
            .value_counts(dropna=False)
            .rename_axis("categorie")
            .reset_index(name="count")
        )
        fig2 = px.bar(cat_count, x="categorie", y="count", title="Répartition par catégorie", text="count")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    
    trend = df[df["month_label"] != "Sans date"].groupby("month_label").size().reset_index(name="tickets")
    trend = trend.sort_values("month_label")
    fig_trend = px.line(trend, x="month_label", y="tickets", markers=True, title="Tendance mensuelle")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.divider()
    closure_data = df[df["closure_days"].notna() & (df["closure_days"] >= 0)]
    if not closure_data.empty:
        fig_closure = px.histogram(
            closure_data, 
            x="closure_days", 
            nbins=20,
            title="Distribution des délais de clôture (jours)",
            labels={"closure_days": "Jours", "count": "Tickets"}
        )

        fig_closure.update_traces(textposition='outside')
        fig_closure.update_layout(
            xaxis_tickangle=-45,
            bargap=0.2,  # Espace entre les barres (20%)
            showlegend=False
        )
        st.plotly_chart(fig_closure, use_container_width=True)

    if show_heatmap:
        st.divider()
        cross = pd.crosstab(df["type_probleme"], df["categorie"])
        st.plotly_chart(px.imshow(cross, text_auto=True, title="Heatmap — Type × Catégorie"), use_container_width=True)

# -----------------------
# Table + export
# -----------------------
cols = [
    "sc_number", "client", "createdOn", "paymentBillingDate", "closure_days", "price", "issue_text",
    "type_probleme", "categorie", "systeme", "systeme_suggere",
    "confiance_systeme", "justification_systeme"
]
for c in cols:
    if c not in df.columns:
        df[c] = None

st.dataframe(df[cols], use_container_width=True, height=480)

csv = df[cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger CSV", csv, file_name="rapport_kpi.csv", mime="text/csv")