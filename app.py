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


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="SAV KPI Dashboard V6", page_icon="📊", layout="wide")
st.title("📊 Appels de service — KPI Dashboard")
st.caption("KPI clés, comparatifs mensuels, tendances 12 mois, top clients et distribution des cycles")

openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None

CACHE_PATH = Path("classification_cache.json")
REPORT_CACHE_PATH = Path("report_cache.json")

DEFAULT_INTERNAL_CLIENTS = [
    "Câble&Son Télécom",
    "Cable & Son",
    "Bureau C&S",
    "RMA C&S",
    "Reserver Services",
]


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.92rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS STYLE PLOTLY
# =========================================================
def style_bar_chart(fig, money=False):
    if money:
        fig.update_traces(texttemplate="%{y:,.0f}$", textposition="outside", cliponaxis=False)
    else:
        fig.update_traces(texttemplate="%{y}", textposition="outside", cliponaxis=False)

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        legend_title_text="",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def style_line_chart(fig):
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text="",
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
    )
    return fig


def rename_traces(fig, mapping: dict):
    fig.for_each_trace(lambda t: t.update(name=mapping.get(t.name, t.name)))
    return fig


# =========================================================
# CACHE
# =========================================================
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
    if "REPORT_CACHE" in st.session_state and isinstance(st.session_state.REPORT_CACHE, dict):
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


def _cache_key_for_text(text: str) -> str:
    norm = (text or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


CLASS_CACHE = _load_cache()
REPORT_CACHE = _load_report_cache()


# =========================================================
# TAXONOMIE
# =========================================================
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
        "Apple TV", "Helix", "Sonos", "QSC", "MyQ", "Polycom", "NAS",
        "Générique", "Non spécifié"
    ],
}

KEYWORDS_FR = {
    "type_probleme": {
        "offline": "Hors ligne",
        "hors ligne": "Hors ligne",
        "pas d'internet": "Hors ligne",
        "no internet": "Hors ligne",
        "pas de son": "Pas de son",
        "no sound": "Pas de son",
        "pas d'image": "Pas d'image",
        "no image": "Pas d'image",
        "ne fonctionne pas": "Pas de contrôle",
        "ne marche pas": "Pas de contrôle",
        "contrôle": "Pas de contrôle",
        "programmation": "Programmation",
        "ajustement": "Programmation",
        "installer": "Installation/Remplacement",
        "installation": "Installation/Remplacement",
        "remplacer": "Installation/Remplacement",
        "batterie": "Mécanique/Batterie",
        "défectueux": "Mécanique/Batterie",
        "defectueux": "Mécanique/Batterie",
        "application": "Logiciel/App",
        "login": "Logiciel/App",
        "mot de passe": "Logiciel/App",
    },
    "categorie": {
        "audio": "Audio",
        "son": "Audio",
        "subwoofer": "Audio",
        "ampli": "Audio",
        "qsc": "Audio",
        "sonos": "Audio",
        "vidéo": "Vidéo",
        "video": "Vidéo",
        "tv": "Vidéo",
        "hdmi": "Vidéo",
        "splitter": "Vidéo",
        "apple tv": "Vidéo",
        "helix": "Vidéo",
        "polycom": "Vidéo",
        "éclairage": "Éclairage",
        "eclairage": "Éclairage",
        "lumière": "Éclairage",
        "scene": "Éclairage",
        "scène": "Éclairage",
        "réseau": "Réseau",
        "reseau": "Réseau",
        "internet": "Réseau",
        "wifi": "Réseau",
        "unifi": "Réseau",
        "usg": "Réseau",
        "uxg": "Réseau",
        "sécurité": "Sécurité",
        "securite": "Sécurité",
        "caméra": "Sécurité",
        "camera": "Sécurité",
        "hikvision": "Sécurité",
        "alarme": "Sécurité",
        "cdvi": "Sécurité",
        "myq": "Sécurité",
        "clare": "Sécurité",
        "store": "Stores",
        "toile": "Stores",
        "lutron": "Stores",
        "somfy": "Stores",
        "control4": "Système de contrôle",
        "domotique": "Système de contrôle",
    },
    "systeme": {
        "control4": "Control4",
        "c4": "Control4",
        "unifi": "Unifi",
        "usg": "Unifi",
        "udm": "Unifi",
        "uxg": "Unifi",
        "lutron": "Lutron",
        "somfy": "Somfy",
        "hikvision": "Hikvision",
        "clare": "Clare",
        "cdvi": "CDVI",
        "apple tv": "Apple TV",
        "helix": "Helix",
        "sonos": "Sonos",
        "qsc": "QSC",
        "myq": "MyQ",
        "polycom": "Polycom",
        "nas": "NAS",
    },
}

_BRAND_REGEX = re.compile(
    r"\b(c4|control4|core[135]|ea-?[135]|unifi|usg|udm|uxg|sonos|hikvision|caseta|homeworks|lutron|somfy|qsc|apc|polycom|myq|helix|apple\s*tv|nas)\b",
    flags=re.IGNORECASE,
)

_BRAND_CANON = {
    "control4": "Control4",
    "c4": "Control4",
    "core1": "Control4",
    "core3": "Control4",
    "core5": "Control4",
    "ea1": "Control4",
    "ea-1": "Control4",
    "ea3": "Control4",
    "ea-3": "Control4",
    "ea5": "Control4",
    "ea-5": "Control4",
    "unifi": "Unifi",
    "usg": "Unifi",
    "udm": "Unifi",
    "uxg": "Unifi",
    "sonos": "Sonos",
    "hikvision": "Hikvision",
    "caseta": "Lutron",
    "homeworks": "Lutron",
    "lutron": "Lutron",
    "somfy": "Somfy",
    "qsc": "QSC",
    "apc": "APC",
    "polycom": "Polycom",
    "myq": "MyQ",
    "helix": "Helix",
    "apple tv": "Apple TV",
    "nas": "NAS",
}


# =========================================================
# UTILS
# =========================================================
def clean_html(text):
    if not isinstance(text, str):
        return ""
    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


def _snap(value: str, allowed: list[str]) -> str:
    v = (value or "").strip()
    if v in allowed:
        return v
    candidates = difflib.get_close_matches(v.lower(), [a.lower() for a in allowed], n=1, cutoff=0.75)
    if candidates:
        idx = [a.lower() for a in allowed].index(candidates[0])
        return allowed[idx]
    return "Non spécifié" if "Non spécifié" in allowed else allowed[-1]


def _explicit_brand(text: str):
    match = _BRAND_REGEX.search(text or "")
    if not match:
        return None
    key = match.group(1).lower().replace(" ", "")
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
- "systeme_suggere" ∈ {TAXO_FR["systeme"]} ou null
- "confiance_systeme" ∈ [0.0, 1.0]
- "justification_systeme": phrase courte (≤12 mots)

Classe ce texte :
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
            "justification_systeme": ""
        }

    cache_key = _cache_key_for_text(text)
    if cache_key in CLASS_CACHE:
        return CLASS_CACHE[cache_key]

    explicit_system = _explicit_brand(text)
    data = {}

    if client is not None:
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
            data = {}

    type_probleme = _snap(data.get("type_probleme", ""), TAXO_FR["type_probleme"])
    categorie = _snap(data.get("categorie", ""), TAXO_FR["categorie"])

    if explicit_system:
        systeme = explicit_system
        systeme_suggere = None
        confiance = 0.95
        justification = "marque explicite"
    else:
        systeme = _snap(data.get("systeme", "Non spécifié"), TAXO_FR["systeme"])
        suggested = data.get("systeme_suggere")
        systeme_suggere = _snap(suggested, TAXO_FR["systeme"]) if suggested else None
        if systeme_suggere == "Non spécifié":
            systeme_suggere = None
        confiance = float(data.get("confiance_systeme", 0.0) or 0.0)
        justification = (data.get("justification_systeme") or "")[:120]

    scores = _keyword_score(text, KEYWORDS_FR)

    if scores["type_probleme"]:
        type_probleme = max(scores["type_probleme"], key=scores["type_probleme"].get)

    if scores["categorie"]:
        categorie = max(scores["categorie"], key=scores["categorie"].get)

    if not explicit_system and scores["systeme"] and systeme == "Non spécifié":
        best_sys = max(scores["systeme"], key=scores["systeme"].get)
        systeme_suggere = best_sys
        confiance = max(confiance, 0.70)
        if not justification:
            justification = "mot-clé système détecté"

    result = {
        "type_probleme": type_probleme,
        "categorie": categorie,
        "systeme": systeme,
        "systeme_suggere": systeme_suggere,
        "confiance_systeme": round(float(confiance), 2),
        "justification_systeme": justification,
    }

    CLASS_CACHE[cache_key] = result
    _save_cache(CLASS_CACHE)
    return result


def classify_all(df: pd.DataFrame, file_hash: str) -> pd.DataFrame:
    if file_hash in REPORT_CACHE:
        return pd.DataFrame(REPORT_CACHE[file_hash])

    rows = []
    progress_bar = st.progress(0)
    total = len(df)

    for idx, txt in enumerate(df["issue_text"].fillna("")):
        rows.append(classify_service_call_gpt(txt))
        progress_bar.progress((idx + 1) / total)

    progress_bar.empty()

    result_df = pd.DataFrame(rows)
    REPORT_CACHE[file_hash] = result_df.to_dict("records")
    _save_report_cache(REPORT_CACHE)
    return result_df


def json_to_df(file_like):
    data = json.load(file_like)
    rows = data.get("serviceCalls", data) if isinstance(data, dict) else data
    df = pd.DataFrame(rows)

    expected_cols = [
        "number", "client", "issueReported", "price", "createdOn",
        "stateName", "paymentBillingDate", "resourceIds", "status",
        "serviceContract", "serviceContractNumber", "project", "projectNumber"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df["issue_text"] = df["issueReported"].apply(clean_html)
    df["createdOn"] = pd.to_datetime(df["createdOn"], errors="coerce")
    df["paymentBillingDate"] = pd.to_datetime(df["paymentBillingDate"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    df.rename(columns={"number": "sc_number"}, inplace=True)
    month_period = df["createdOn"].dt.to_period("M")
    df["month_label"] = month_period.astype(str)
    df.loc[month_period.isna(), "month_label"] = "Sans date"

    df["year_month"] = df["createdOn"].dt.to_period("M")
    df["is_free"] = df["price"].eq(0)
    df["is_completed"] = df["stateName"].astype(str).str.lower().eq("completed")
    df["is_canceled"] = df["stateName"].astype(str).str.lower().eq("canceled")

    df["closure_days"] = None
    valid_mask = df["paymentBillingDate"].notna() & df["createdOn"].notna()
    df.loc[valid_mask, "closure_days"] = (
        df.loc[valid_mask, "paymentBillingDate"] - df.loc[valid_mask, "createdOn"]
    ).dt.days
    df["closure_days"] = pd.to_numeric(df["closure_days"], errors="coerce")

    return df


def enriched_csv_to_df(file_like):
    df = pd.read_csv(file_like)

    for c in ["createdOn", "paymentBillingDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in ["price", "closure_days", "confiance_systeme"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["is_free", "is_completed", "is_canceled"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    if "resourceIds" in df.columns:
        def parse_resource_ids(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            try:
                parsed = json.loads(x)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        df["resourceIds"] = df["resourceIds"].apply(parse_resource_ids)

    if "createdOn" in df.columns:
        df["year_month"] = df["createdOn"].dt.to_period("M")
        df["month_label"] = df["year_month"].astype(str)
        df.loc[df["year_month"].isna(), "month_label"] = "Sans date"

    return df


def months_covered_from_periods(period_series):
    clean = period_series.dropna().sort_values().unique()
    return len(clean)


def build_monthly_core(df_scope):
    monthly = (
        df_scope[df_scope["year_month"].notna()]
        .groupby("year_month")
        .agg(
            sc_count=("sc_number", "count"),
            avg_closure_days=("closure_days", "mean"),
        )
        .reset_index()
        .sort_values("year_month")
    )

    if monthly.empty:
        return monthly

    monthly["year_month_str"] = monthly["year_month"].astype(str)
    monthly["sc_count_roll12"] = monthly["sc_count"].rolling(12, min_periods=1).mean()
    monthly["avg_closure_days_roll12"] = monthly["avg_closure_days"].rolling(12, min_periods=1).mean()
    return monthly


def safe_pct_change(current, previous):
    if previous in [0, None] or pd.isna(previous):
        return None
    return (current - previous) / previous * 100


def fmt_delta_pct(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.1f}%"


def fmt_abs_change(value, metric):
    if value is None or pd.isna(value):
        return ""
    if metric == "avg_closure_days":
        return f"{value:+.1f} j"
    return f"{value:+,.0f}"


def build_delta_text(current, previous, metric):
    pct = safe_pct_change(current, previous)
    abs_delta = None if previous is None or pd.isna(previous) else current - previous
    pct_txt = fmt_delta_pct(pct)
    abs_txt = fmt_abs_change(abs_delta, metric)

    if pct_txt == "n/a" and abs_txt == "":
        return "n/a"
    if pct_txt == "n/a":
        return abs_txt
    if abs_txt == "":
        return pct_txt
    return f"{pct_txt} ({abs_txt})"


def compare_point_to_prev(monthly_df, idx, metric):
    current = monthly_df.loc[idx, metric]
    if idx < 1:
        return current, None, None
    previous = monthly_df.loc[idx - 1, metric]
    return current, previous, build_delta_text(current, previous, metric)


def compare_point_to_ly(monthly_df, idx, metric):
    current = monthly_df.loc[idx, metric]
    if idx < 12:
        return current, None, None
    previous = monthly_df.loc[idx - 12, metric]
    return current, previous, build_delta_text(current, previous, metric)


def compare_point_to_avg12(monthly_df, idx, metric):
    current = monthly_df.loc[idx, metric]
    if idx < 12:
        return current, None, None
    previous = monthly_df.loc[idx - 12:idx - 1, metric].mean()
    return current, previous, build_delta_text(current, previous, metric)


def build_metric_display(current, delta, metric):
    if current is None or pd.isna(current):
        return "n/a", "n/a"
    if metric == "avg_closure_days":
        cur = f"{current:.1f} j"
    elif metric == "avg_sc_per_month":
        cur = f"{current:,.1f}"
    else:
        cur = f"{current:,.0f}"
    return cur, delta if delta else "n/a"


def build_trend_view(monthly_all, start_period, end_period):
    if monthly_all.empty:
        return monthly_all.copy()
    view = monthly_all[(monthly_all["year_month"] >= start_period) & (monthly_all["year_month"] <= end_period)].copy()
    return view


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📥 Source de données")

    input_mode = st.radio(
        "Mode d'entrée",
        ["JSON brut (classification possible)", "CSV enrichi (sans GPT)"]
    )

    raw_file = None
    enriched_file = None

    if input_mode == "JSON brut (classification possible)":
        raw_file = st.file_uploader("Téléverser les données JSON", type=["json"], key="main_json")
    else:
        enriched_file = st.file_uploader("Téléverser le CSV enrichi", type=["csv"], key="enriched_csv")

    st.divider()
    st.subheader("🏢 Clients internes")

    internal_clients_text = st.text_area(
        "Liste clients internes (1 ligne = 1 client)",
        value="\n".join(DEFAULT_INTERNAL_CLIENTS),
        height=130
    )

    internal_clients = [x.strip() for x in internal_clients_text.splitlines() if x.strip()]
    exclude_internal_from_gratuity = st.toggle(
        "Exclure clients internes du périmètre gratuité",
        value=True
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Cache textes"):
            CLASS_CACHE.clear()
            _save_cache(CLASS_CACHE)
            st.success("Cache textes vidé")

    with col_b:
        if st.button("🗑️ Cache rapports"):
            REPORT_CACHE.clear()
            _save_report_cache(REPORT_CACHE)
            st.success("Cache rapports vidé")

    st.caption(f"📊 {len(CLASS_CACHE)} textes en cache")
    st.caption(f"📁 {len(REPORT_CACHE)} rapports en cache")


# =========================================================
# LOAD DATA
# =========================================================
if input_mode == "JSON brut (classification possible)":
    if not raw_file:
        st.info("➡️ Charge un fichier JSON brut.")
        st.stop()

    file_content = raw_file.getvalue()
    file_hash = _compute_file_hash(file_content)
    raw_file.seek(0)

    df_raw = json_to_df(raw_file)
    if df_raw.empty:
        st.warning("Aucune donnée trouvée.")
        st.stop()

    status_box = st.empty()
    status_box.info("⏳ Enrichissement en cours...")

    df_cls = classify_all(df_raw, file_hash)
    df = pd.concat([df_raw.reset_index(drop=True), df_cls.reset_index(drop=True)], axis=1)

    status_box.success("✅ Chargement et enrichissement terminés")
else:
    if not enriched_file:
        st.info("➡️ Charge un CSV enrichi.")
        st.stop()

    df = enriched_csv_to_df(enriched_file)
    if df.empty:
        st.warning("Aucune donnée dans le CSV enrichi.")
        st.stop()

    st.success("✅ CSV enrichi chargé")


# =========================================================
# FILTERS
# =========================================================
with st.expander("🔎 Filtres", expanded=False):
    f1, f2, f3, f4 = st.columns(4)

    all_clients = sorted([x for x in df["client"].dropna().unique().tolist()]) if "client" in df.columns else []
    all_systems = sorted([x for x in df["systeme"].dropna().unique().tolist()]) if "systeme" in df.columns else []
    all_categories = sorted([x for x in df["categorie"].dropna().unique().tolist()]) if "categorie" in df.columns else []
    all_months = sorted([x for x in df["month_label"].dropna().unique().tolist() if x != "Sans date"]) if "month_label" in df.columns else []

    selected_clients = f1.multiselect("Client(s)", all_clients)
    selected_systems = f2.multiselect("Système(s)", all_systems)
    selected_categories = f3.multiselect("Catégorie(s)", all_categories)

    period_start = None
    period_end = None
    if all_months:
        default_start_idx = max(0, len(all_months) - 12)
        period_start = f4.selectbox("Mois début", all_months, index=default_start_idx, key="period_start")
        period_end = f4.selectbox("Mois fin", all_months, index=len(all_months) - 1, key="period_end")

    f5, f6 = st.columns(2)
    free_filter = f5.selectbox("Facturation", ["Tous", "Facturables", "Gratuits"])
    state_filter = f6.selectbox("État", ["Tous", "Complétés", "Annulés", "Autres"])

df_base = df.copy()

if selected_clients:
    df_base = df_base[df_base["client"].isin(selected_clients)]
if selected_systems:
    df_base = df_base[df_base["systeme"].isin(selected_systems)]
if selected_categories:
    df_base = df_base[df_base["categorie"].isin(selected_categories)]

if free_filter == "Facturables":
    df_base = df_base[df_base["price"] > 0]
elif free_filter == "Gratuits":
    df_base = df_base[df_base["price"] == 0]

if state_filter == "Complétés":
    df_base = df_base[df_base["is_completed"]]
elif state_filter == "Annulés":
    df_base = df_base[df_base["is_canceled"]]
elif state_filter == "Autres":
    df_base = df_base[~df_base["is_completed"] & ~df_base["is_canceled"]]

if df_base.empty:
    st.warning("Aucune donnée après application des filtres.")
    st.stop()

df_f = df_base.copy()

if period_start and period_end:
    start_p = pd.Period(period_start, freq="M")
    end_p = pd.Period(period_end, freq="M")
    if start_p > end_p:
        start_p, end_p = end_p, start_p

    df_f = df_f[df_f["year_month"].notna()]
    df_f = df_f[(df_f["year_month"] >= start_p) & (df_f["year_month"] <= end_p)]

if df_f.empty:
    st.warning("Aucune donnée sur la période sélectionnée.")
    st.stop()


# =========================================================
# CONTEXTE / DATASETS
# =========================================================
months_covered = months_covered_from_periods(df_f["year_month"])
period_min = df_f["year_month"].dropna().min()
period_max = df_f["year_month"].dropna().max()
period_label = f"{period_min} → {period_max}" if pd.notna(period_min) and pd.notna(period_max) else "Période non définie"

total_tickets = len(df_f)
closure_valid_df = df_f[df_f["closure_days"].notna() & (df_f["closure_days"] >= 0)]

avg_closure_days = float(closure_valid_df["closure_days"].mean()) if not closure_valid_df.empty else 0.0
avg_sc_per_month = total_tickets / months_covered if months_covered > 0 else 0.0

client_summary = (
    df_f.groupby("client", dropna=False)
    .agg(sc_count=("sc_number", "count"))
    .reset_index()
    .sort_values(["sc_count", "client"], ascending=[False, True])
)

monthly_metrics_all = build_monthly_core(df_base)
monthly_metrics = build_trend_view(monthly_metrics_all, start_p, end_p) if period_start and period_end else monthly_metrics_all.copy()

selected_month_view = None
selected_idx_all = None

if not monthly_metrics.empty and not monthly_metrics_all.empty:
    selected_month_view = monthly_metrics["year_month_str"].iloc[-1]
    selected_row_all = monthly_metrics_all[monthly_metrics_all["year_month_str"] == selected_month_view]
    if not selected_row_all.empty:
        selected_idx_all = selected_row_all.index[0]


# =========================================================
# CONTEXTE DE PÉRIODE
# =========================================================
st.subheader("🗓️ Période des services")
pc1, pc2 = st.columns(2)
pc1.metric("Période analysée", period_label)
pc2.metric("Mois couverts", f"{months_covered}")


# =========================================================
# BLOC 1 : VUE PÉRIODE
# =========================================================
st.subheader("📌 Vue période analysée")
st.caption("Indicateurs clés cumulés sur la période filtrée.")

c1, c2, c3 = st.columns(3)
c1.metric("Appels de service totaux", f"{total_tickets:,}")
c2.metric("Moyenne des interventions / mois", f"{avg_sc_per_month:,.1f}")
c3.metric("Cycle moyen des interventions", f"{avg_closure_days:.1f} j")


# =========================================================
# BLOC 2 : DERNIER MOIS DE LA PÉRIODE
# =========================================================
st.subheader("📅 Dernier mois de la période")
st.caption("Comparaison automatique du dernier mois visible vs mois précédent, vs même mois l’an passé et vs moyenne 12 mois précédents.")

if selected_idx_all is None:
    st.info("Pas assez d'historique pour le bloc mensuel.")
else:
    st.markdown(
        f"""
        <div style='font-size:26px;font-weight:600;margin-top:4px;margin-bottom:10px;'>
        Mois de référence : {selected_month_view}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**Vs mois précédent**")
    m1, m2 = st.columns(2)
    cur, delta = build_metric_display(*compare_point_to_prev(monthly_metrics_all, selected_idx_all, "sc_count")[::2], "sc_count")
    m1.metric("Appels de service", cur, delta)
    cur, delta = build_metric_display(*compare_point_to_prev(monthly_metrics_all, selected_idx_all, "avg_closure_days")[::2], "avg_closure_days")
    m2.metric("Cycle moyen", cur, delta)

    st.markdown("**Vs même mois l’an passé**")
    y1, y2 = st.columns(2)
    cur, delta = build_metric_display(*compare_point_to_ly(monthly_metrics_all, selected_idx_all, "sc_count")[::2], "sc_count")
    y1.metric("Appels de service", cur, delta)
    cur, delta = build_metric_display(*compare_point_to_ly(monthly_metrics_all, selected_idx_all, "avg_closure_days")[::2], "avg_closure_days")
    y2.metric("Cycle moyen", cur, delta)

    st.markdown("**Vs moyenne 12 mois précédents**")
    a1, a2 = st.columns(2)
    cur, delta = build_metric_display(*compare_point_to_avg12(monthly_metrics_all, selected_idx_all, "sc_count")[::2], "sc_count")
    a1.metric("Appels de service", cur, delta)
    cur, delta = build_metric_display(*compare_point_to_avg12(monthly_metrics_all, selected_idx_all, "avg_closure_days")[::2], "avg_closure_days")
    a2.metric("Cycle moyen", cur, delta)

    st.caption("Les comparaisons utilisent l’historique complet filtré métier, même si la période affichée est limitée.")


# =========================================================
# TENDANCES
# =========================================================
st.divider()
st.subheader("📈 Tendances mensuelles")
st.caption("Barres = valeur mensuelle réelle. Ligne = moyenne roulante 12 mois du mois correspondant. Les écarts affichent la différence en valeur absolue et en %.")

fig_roll_sc = None
fig_roll_delay = None

if not monthly_metrics.empty:
    r1, r2 = st.columns(2)

    # -----------------------------
    # Appels de service
    # -----------------------------
    sc_chart = monthly_metrics.copy()
    sc_chart = sc_chart[sc_chart["sc_count"].notna() & sc_chart["sc_count_roll12"].notna()].copy()

    if not sc_chart.empty:
        sc_chart["month_date"] = sc_chart["year_month"].dt.to_timestamp()
        sc_chart["month_label"] = sc_chart["month_date"].dt.strftime("%b-%y")
        sc_chart["gap_abs"] = sc_chart["sc_count"] - sc_chart["sc_count_roll12"]
        sc_chart["gap_pct"] = (sc_chart["gap_abs"] / sc_chart["sc_count_roll12"] * 100).round(1)
        sc_chart["bar_text"] = (
            sc_chart["sc_count"].round(0).astype(int).astype(str)
            + "<br>"
            + sc_chart["gap_abs"].round(0).astype(int).map(lambda x: f"{x:+d}")
            + " | "
            + sc_chart["gap_pct"].map(lambda x: f"{x:+.1f}%")
        )

        fig_roll_sc = px.bar(
            sc_chart,
            x="month_date",
            y="sc_count",
            text="bar_text",
            title="Appels de service mensuels vs moyenne 12 mois",
            labels={
                "month_date": "Mois",
                "sc_count": "Appels de service",
            },
        )

        fig_roll_sc.add_scatter(
            x=sc_chart["month_date"],
            y=sc_chart["sc_count_roll12"],
            mode="lines+markers",
            name="Moyenne 12 mois",
            hovertemplate=(
                "<b>%{x|%b-%y}</b><br>"
                "Moyenne 12 mois: %{y:.1f}"
                "<extra></extra>"
            ),
        )

        fig_roll_sc.update_traces(
            selector=dict(type="bar"),
            name="Appels mensuels",
            textposition="outside",
            cliponaxis=False,
            width=20 * 24 * 60 * 60 * 1000,  # ~20 jours
            hovertemplate=(
                "<b>%{x|%b-%y}</b><br>"
                "Appels mensuels: %{y:.0f}<br>"
                "Écart vs moy. 12 mois: %{customdata[0]:+.0f}<br>"
                "Écart %: %{customdata[1]:+.1f}%"
                "<extra></extra>"
            ),
            customdata=sc_chart[["gap_abs", "gap_pct"]].to_numpy(),
        )

        fig_roll_sc.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            legend_title_text="",
            margin=dict(l=10, r=10, t=55, b=10),
            hovermode="x unified",
            bargap=0.55,
            xaxis=dict(
                tickmode="array",
                tickvals=sc_chart["month_date"],
                ticktext=sc_chart["month_label"],
            ),
        )
        r1.plotly_chart(fig_roll_sc, use_container_width=True)

    # -----------------------------
    # Cycle moyen
    # -----------------------------
    delay_chart = monthly_metrics.copy()
    delay_chart = delay_chart[
        delay_chart["avg_closure_days"].notna() &
        delay_chart["avg_closure_days_roll12"].notna()
    ].copy()

    if not delay_chart.empty:
        delay_chart["month_date"] = delay_chart["year_month"].dt.to_timestamp()
        delay_chart["month_label"] = delay_chart["month_date"].dt.strftime("%b-%y")
        delay_chart["gap_abs"] = delay_chart["avg_closure_days"] - delay_chart["avg_closure_days_roll12"]
        delay_chart["gap_pct"] = (delay_chart["gap_abs"] / delay_chart["avg_closure_days_roll12"] * 100).round(1)
        delay_chart["bar_text"] = (
            delay_chart["avg_closure_days"].round(1).map(lambda x: f"{x:.1f} j")
            + "<br>"
            + delay_chart["gap_abs"].round(1).map(lambda x: f"{x:+.1f} j")
            + " | "
            + delay_chart["gap_pct"].map(lambda x: f"{x:+.1f}%")
        )

        fig_roll_delay = px.bar(
            delay_chart,
            x="month_date",
            y="avg_closure_days",
            text="bar_text",
            title="Cycle moyen mensuel vs moyenne 12 mois",
            labels={
                "month_date": "Mois",
                "avg_closure_days": "Jours",
            },
        )

        fig_roll_delay.add_scatter(
            x=delay_chart["month_date"],
            y=delay_chart["avg_closure_days_roll12"],
            mode="lines+markers",
            name="Moyenne 12 mois",
            hovertemplate=(
                "<b>%{x|%b-%y}</b><br>"
                "Moyenne 12 mois: %{y:.1f} j"
                "<extra></extra>"
            ),
        )

        fig_roll_delay.update_traces(
            selector=dict(type="bar"),
            name="Cycle moyen mensuel",
            textposition="outside",
            cliponaxis=False,
            width=20 * 24 * 60 * 60 * 1000,  # ~20 jours
            hovertemplate=(
                "<b>%{x|%b-%y}</b><br>"
                "Cycle moyen mensuel: %{y:.1f} j<br>"
                "Écart vs moy. 12 mois: %{customdata[0]:+.1f} j<br>"
                "Écart %: %{customdata[1]:+.1f}%"
                "<extra></extra>"
            ),
            customdata=delay_chart[["gap_abs", "gap_pct"]].to_numpy(),
        )

        fig_roll_delay.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            legend_title_text="",
            margin=dict(l=10, r=10, t=55, b=10),
            hovermode="x unified",
            bargap=0.55,
            xaxis=dict(
                tickmode="array",
                tickvals=delay_chart["month_date"],
                ticktext=delay_chart["month_label"],
            ),
        )
        r2.plotly_chart(fig_roll_delay, use_container_width=True)

# =========================================================
# TOP CLIENTS
# =========================================================
st.divider()
st.subheader("👥 Top clients en nombre d'appels")

if not client_summary.empty:
    fig_top_clients = px.bar(
        client_summary.head(10),
        x="client",
        y="sc_count",
        title="Top 10 clients — nombre d'appels de service",
        labels={
            "client": "Client",
            "sc_count": "Nombre d'appels"
        },
        text="sc_count"
    )
    fig_top_clients = style_bar_chart(fig_top_clients, money=False)
    fig_top_clients.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_top_clients, use_container_width=True)
else:
    st.info("Aucun client à afficher.")


# =========================================================
# DISTRIBUTION DES CYCLES
# =========================================================
st.divider()
st.subheader("⏱️ Distribution des cycles")
st.caption("Chaque barre représente un nombre de jours de cycle exact et le nombre de tickets associés.")

if not closure_valid_df.empty:
    cycle_dist = (
        closure_valid_df["closure_days"]
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cycle_dist.columns = ["closure_days", "tickets"]

    fig_closure = px.bar(
        cycle_dist,
        x="closure_days",
        y="tickets",
        text="tickets",
        title="Distribution des cycles (jours)",
        labels={
            "closure_days": "Cycle (jours)",
            "tickets": "Nombre de tickets"
        }
    )
    fig_closure.update_traces(textposition="outside", cliponaxis=False)
    fig_closure.update_layout(
        showlegend=False,
        bargap=0.15,
        xaxis=dict(type="category"),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig_closure, use_container_width=True)
else:
    st.info("Aucune donnée de cycle disponible sur la période.")