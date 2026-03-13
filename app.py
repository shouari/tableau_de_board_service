import json
import re
import difflib
import hashlib
from pathlib import Path
from io import BytesIO
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
import openai


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="SAV KPI Dashboard V5.3", page_icon="📊", layout="wide")
st.title("📊 Service Calls — KPI Dashboard V5.3")
st.caption("Vue période + vue mois sélectionné, comparaisons avancées, anomalies, drill-down")

openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None

CACHE_PATH = Path("classification_cache.json")
REPORT_CACHE_PATH = Path("report_cache.json")

CYCLE_DAY_TO_TECH_HOURS = 0.3
MIN_SC_FOR_ALERT = 3
FREE_RATE_ALERT_HIGH = 50.0
FREE_RATE_ALERT_MEDIUM = 30.0

DEFAULT_INTERNAL_CLIENTS = [
    "Câble&Son Télécom",
    "Cable & Son",
    "Bureau C&S",
    "RMA C&S",
    "Reserver Services",
]


# =========================================================
# HELPERS STYLE PLOTLY
# =========================================================
def style_bar_chart(fig, money=False):
    if money:
        fig.update_traces(
            texttemplate="%{y:,.0f}$",
            textposition="outside",
            cliponaxis=False
        )
    else:
        fig.update_traces(
            texttemplate="%{y}",
            textposition="outside",
            cliponaxis=False
        )

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )
    return fig


def style_line_chart(fig):
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text=""
    )
    return fig


def style_pie_chart(fig):
    fig.update_layout(legend_title_text="")
    fig.update_traces(textinfo="percent+label")
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


def _explicit_brand(text: str) -> str | None:
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

    df["year"] = df["createdOn"].dt.year
    df["month_num"] = df["createdOn"].dt.month
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

    issue_lower = df["issue_text"].fillna("").str.lower()
    contract_lower = df["serviceContract"].fillna("").str.lower()

    df["free_reason_hint"] = None
    df.loc[issue_lower.str.contains("garantie", na=False), "free_reason_hint"] = "Garantie"
    df.loc[contract_lower.str.contains("garantie", na=False), "free_reason_hint"] = "Garantie"

    df.loc[
        issue_lower.str.contains(
            "sans frais|non facturable|inclus dans|plan de maintenance|plan de service|appel sans frais|déplacement sans frais",
            na=False
        ),
        "free_reason_hint"
    ] = "Contrat / sans frais"

    df.loc[contract_lower.str.contains("plan", na=False), "free_reason_hint"] = "Contrat / sans frais"

    df.loc[
        issue_lower.str.contains("notre erreur|satisfaction client|oubli du dernier technicien|rabais 100%", na=False),
        "free_reason_hint"
    ] = "Geste commercial / correction"

    df.loc[df["is_canceled"], "free_reason_hint"] = "Annulé"

    return df


def enriched_csv_to_df(file_like):
    df = pd.read_csv(file_like)

    datetime_cols = ["createdOn", "paymentBillingDate"]
    for c in datetime_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    numeric_cols = ["price", "closure_days", "confiance_systeme"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bool_cols = ["is_free", "is_completed", "is_canceled"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
            df[c] = df[c].fillna(False)

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


def build_monthly_overall(df_scope):
    monthly = (
        df_scope[df_scope["year_month"].notna()]
        .groupby("year_month")
        .agg(
            sc_count=("sc_number", "count"),
            revenue=("price", "sum"),
            canceled_sc=("is_canceled", "sum"),
            avg_price=("price", "mean"),
            avg_closure_days=("closure_days", "mean"),
            hors_ligne_sc=("type_probleme", lambda s: (s == "Hors ligne").sum()),
            control4_sc=("systeme", lambda s: (s == "Control4").sum()),
            unifi_sc=("systeme", lambda s: (s == "Unifi").sum()),
        )
        .reset_index()
        .sort_values("year_month")
    )

    if monthly.empty:
        return monthly

    monthly["cancel_rate_pct"] = (monthly["canceled_sc"] / monthly["sc_count"] * 100).round(2)
    monthly["hors_ligne_pct"] = (monthly["hors_ligne_sc"] / monthly["sc_count"] * 100).round(2)
    monthly["control4_pct"] = (monthly["control4_sc"] / monthly["sc_count"] * 100).round(2)
    monthly["unifi_pct"] = (monthly["unifi_sc"] / monthly["sc_count"] * 100).round(2)
    monthly["year_month_str"] = monthly["year_month"].astype(str)
    return monthly


def build_monthly_gratuity(df_scope):
    monthly = (
        df_scope[df_scope["year_month"].notna()]
        .groupby("year_month")
        .agg(
            free_sc=("is_free", "sum"),
            free_rate_pct=("is_free", lambda s: s.mean() * 100),
            avg_free_cycle=("closure_days", "mean"),
        )
        .reset_index()
        .sort_values("year_month")
    )

    if monthly.empty:
        return monthly

    monthly["avg_free_cycle"] = monthly["avg_free_cycle"].fillna(0)
    monthly["estimated_free_hours"] = monthly["avg_free_cycle"] * CYCLE_DAY_TO_TECH_HOURS
    monthly["estimated_hidden_cost"] = monthly["free_sc"] * monthly["estimated_free_hours"] * hourly_rate
    monthly["year_month_str"] = monthly["year_month"].astype(str)
    return monthly


def add_change_columns(monthly, metric_cols):
    if monthly.empty:
        return monthly

    monthly = monthly.copy()
    for col in metric_cols:
        monthly[f"{col}_mom_pct"] = monthly[col].pct_change() * 100
        monthly[f"{col}_yoy_pct"] = monthly[col].pct_change(12) * 100
        monthly[f"{col}_roll12"] = monthly[col].rolling(12, min_periods=1).mean()
    return monthly


def merge_monthly_tables(overall, gratuity):
    if overall.empty and gratuity.empty:
        return pd.DataFrame()

    if overall.empty:
        monthly = gratuity.copy()
    elif gratuity.empty:
        monthly = overall.copy()
    else:
        monthly = pd.merge(overall, gratuity, on=["year_month", "year_month_str"], how="outer")

    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    return monthly


def safe_pct_change(current, previous):
    if previous in [0, None] or pd.isna(previous):
        return None
    return (current - previous) / previous * 100


def fmt_delta_pct(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.1f}%"


def compare_selected_period(monthly_df, idx, metric, mode):
    current = monthly_df.loc[idx, metric]

    if mode == "vs mois précédent":
        if idx < 1:
            return current, None
        previous = monthly_df.loc[idx - 1, metric]
        return current, safe_pct_change(current, previous)

    elif mode == "vs même mois année précédente":
        if idx < 12:
            return current, None
        previous = monthly_df.loc[idx - 12, metric]
        return current, safe_pct_change(current, previous)

    elif mode == "vs moyenne 3 mois précédents":
        if idx < 3:
            return current, None
        previous = monthly_df.loc[idx - 3:idx - 1, metric].mean()
        return current, safe_pct_change(current, previous)

    elif mode == "vs moyenne 6 mois précédents":
        if idx < 6:
            return current, None
        previous = monthly_df.loc[idx - 6:idx - 1, metric].mean()
        return current, safe_pct_change(current, previous)

    elif mode == "vs moyenne 12 mois précédents":
        if idx < 12:
            return current, None
        previous = monthly_df.loc[idx - 12:idx - 1, metric].mean()
        return current, safe_pct_change(current, previous)

    elif mode == "3 mois glissants vs 3 mois précédents":
        if idx < 5:
            return None, None
        current_3 = monthly_df.loc[idx - 2:idx, metric].sum()
        previous_3 = monthly_df.loc[idx - 5:idx - 3, metric].sum()
        return current_3, safe_pct_change(current_3, previous_3)

    elif mode == "6 mois glissants vs 6 mois précédents":
        if idx < 11:
            return None, None
        current_6 = monthly_df.loc[idx - 5:idx, metric].sum()
        previous_6 = monthly_df.loc[idx - 11:idx - 6, metric].sum()
        return current_6, safe_pct_change(current_6, previous_6)

    elif mode == "12 mois glissants vs 12 mois précédents":
        if idx < 23:
            return None, None
        current_12 = monthly_df.loc[idx - 11:idx, metric].sum()
        previous_12 = monthly_df.loc[idx - 23:idx - 12, metric].sum()
        return current_12, safe_pct_change(current_12, previous_12)

    return current, None


def build_metric_display(current, delta, metric):
    if current is None or pd.isna(current):
        return "n/a", "n/a"

    if metric in ["revenue", "estimated_hidden_cost", "avg_price"]:
        cur = f"{current:,.0f} $"
    elif metric in ["free_rate_pct", "hors_ligne_pct", "control4_pct", "unifi_pct", "cancel_rate_pct"]:
        cur = f"{current:.1f}%"
    elif metric == "avg_closure_days":
        cur = f"{current:.1f} j"
    else:
        cur = f"{current:,.0f}"

    return cur, fmt_delta_pct(delta)


def compare_last_12_vs_prev_12(monthly_df, metric):
    if len(monthly_df) < 24:
        return None
    last_12 = monthly_df[metric].tail(12).sum()
    prev_12 = monthly_df[metric].iloc[-24:-12].sum()
    return safe_pct_change(last_12, prev_12)


def detect_monthly_anomalies(monthly_df):
    if monthly_df.empty or len(monthly_df) < 3:
        return pd.DataFrame(columns=["month", "metric", "value", "delta_pct", "severity", "message"])

    rows = []
    checks = [
        ("sc_count_mom_pct", "SC", "volume d'appels", "sc_count"),
        ("revenue_mom_pct", "Revenu", "revenu mensuel", "revenue"),
        ("free_rate_pct_mom_pct", "Taux gratuit", "taux de gratuité", "free_rate_pct"),
        ("estimated_hidden_cost_mom_pct", "Coût caché", "coût caché estimé", "estimated_hidden_cost"),
        ("hors_ligne_pct_mom_pct", "Hors ligne", "part des tickets hors ligne", "hors_ligne_pct"),
    ]

    for _, row in monthly_df.iterrows():
        month = str(row["year_month"])
        for delta_col, label, desc, base_col in checks:
            delta = row.get(delta_col)
            if pd.isna(delta):
                continue

            abs_delta = abs(delta)
            if abs_delta >= 50:
                sev = "Critique"
            elif abs_delta >= 30:
                sev = "Élevée"
            elif abs_delta >= 20:
                sev = "Modérée"
            else:
                continue

            direction = "hausse" if delta > 0 else "baisse"
            rows.append({
                "month": month,
                "metric": label,
                "value": row.get(base_col),
                "delta_pct": round(delta, 1),
                "severity": sev,
                "message": f"{month} : {direction} de {abs_delta:.1f}% sur {desc}"
            })

    anomalies = pd.DataFrame(rows)
    if not anomalies.empty:
        severity_order = {"Critique": 3, "Élevée": 2, "Modérée": 1}
        anomalies["sev_rank"] = anomalies["severity"].map(severity_order)
        anomalies = anomalies.sort_values(["sev_rank", "delta_pct"], ascending=[False, False]).drop(columns=["sev_rank"])
    return anomalies


def markdown_summary_to_html(summary_markdown: str) -> str:
    lines = [line.strip() for line in summary_markdown.splitlines() if line.strip()]
    html_parts = []
    in_list = False

    for line in lines:
        if line.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h2>{escape(line[4:])}</h2>")
        elif line.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{escape(line[2:])}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{escape(line)}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "".join(html_parts)


def to_excel_bytes(dataframes: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df_sheet in dataframes.items():
            safe_name = str(sheet_name)[:31]
            df_sheet.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            for idx, col_name in enumerate(df_sheet.columns, start=1):
                max_len = len(str(col_name))
                if not df_sheet.empty:
                    sample = df_sheet[col_name].fillna("").astype(str).head(300).tolist()
                    max_len = max(max_len, max(len(v) for v in sample) if sample else max_len)
                ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = min(max_len + 2, 40)
    output.seek(0)
    return output.getvalue()


def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})


def build_html_report(summary_markdown, kpis, figures, title="Rapport SAV KPI"):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_html = markdown_summary_to_html(summary_markdown)

    kpi_cards = "".join([
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape(label)}</div>
            <div class="kpi-value">{escape(value)}</div>
            <div class="kpi-sub">{escape(sub)}</div>
        </div>
        """
        for label, value, sub in kpis
    ])

    charts_html = "".join([
        f"""
        <section class="chart-card">
            <div class="chart-header">
                <h3>{escape(section_title)}</h3>
            </div>
            <div class="chart-body">
                {fig_to_html(fig)}
            </div>
        </section>
        """
        for section_title, fig in figures
        if fig is not None
    ])

    html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>{escape(title)}</title>
    <style>
        :root {{
            --bg: #f3f6fb;
            --panel: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #e5e7eb;
            --shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
            --radius: 18px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            background: linear-gradient(180deg, #eff6ff 0%, var(--bg) 180px);
            color: var(--text);
        }}
        .page {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 28px;
        }}
        .hero {{
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #3b82f6 100%);
            color: white;
            border-radius: 24px;
            padding: 28px 30px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }}
        .hero h1 {{ margin: 0 0 8px 0; font-size: 34px; line-height: 1.1; }}
        .hero p {{ margin: 0; color: rgba(255,255,255,0.90); font-size: 15px; }}
        .summary-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 22px 24px;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
        }}
        .summary-card h2 {{ margin: 0 0 12px 0; font-size: 22px; }}
        .summary-card ul {{ margin: 0; padding-left: 22px; }}
        .summary-card li {{ margin-bottom: 10px; line-height: 1.5; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            box-shadow: var(--shadow);
        }}
        .kpi-label {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.04em; }}
        .kpi-value {{ font-size: 30px; font-weight: 700; line-height: 1.1; margin-bottom: 8px; }}
        .kpi-sub {{ font-size: 13px; color: var(--muted); }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 20px;
        }}
        .chart-card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
        }}
        .chart-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        }}
        .chart-header h3 {{ margin: 0; font-size: 18px; }}
        .chart-body {{ padding: 10px 10px 2px 10px; }}
        .footer {{ margin-top: 26px; text-align: center; color: var(--muted); font-size: 13px; }}
        @media (max-width: 1100px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 700px) {{
            .page {{ padding: 16px; }}
            .hero {{ padding: 22px 20px; }}
            .hero h1 {{ font-size: 28px; }}
            .kpi-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>{escape(title)}</h1>
            <p>Rapport statique exporté le {escape(generated_at)}</p>
        </section>

        <section class="summary-card">
            {summary_html}
        </section>

        <section class="kpi-grid">
            {kpi_cards}
        </section>

        <section class="charts-grid">
            {charts_html}
        </section>

        <div class="footer">
            Rapport généré automatiquement depuis le dashboard SAV KPI
        </div>
    </div>
</body>
</html>
"""
    return html.encode("utf-8")


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

    tech_map_file = st.file_uploader(
        "Téléverser le mapping techniciens JSON (optionnel)",
        type=["json"],
        key="tech_map_json"
    )

    TECH_MAP = {}
    if tech_map_file is not None:
        try:
            TECH_MAP = json.load(tech_map_file)
            if isinstance(TECH_MAP, list):
                converted = {}
                for item in TECH_MAP:
                    if isinstance(item, dict) and "id" in item and "name" in item:
                        name = re.sub(r"^\s*\d+\s*-\s*", "", str(item["name"])).strip()
                        name = re.sub(r"^\s*\d+\-\s*", "", name).strip()
                        converted[int(item["id"])] = name
                TECH_MAP = converted
            else:
                TECH_MAP = {int(k): v for k, v in TECH_MAP.items()}
            st.success("✅ Mapping techniciens chargé")
        except Exception as e:
            st.error(f"Erreur lecture mapping techniciens : {e}")

    st.divider()
    st.subheader("⚙️ Paramètres")

    hourly_rate = st.slider(
        "Coût horaire technicien ($)",
        min_value=50,
        max_value=250,
        value=100,
        step=5
    )

    st.caption(f"Conversion retenue : 1 jour de cycle = {CYCLE_DAY_TO_TECH_HOURS} h technicien")

    show_heatmap = st.toggle("Heatmap Type × Catégorie", True)
    show_client_bubble = st.toggle("Bubble chart clients", True)
    show_data_table = st.toggle("Afficher table détaillée", True)
    show_exec_summary = st.toggle("Afficher résumé exécutif", True)

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
    st.subheader("🚨 Détection clients à forte gratuité")
    st.write(f"- Volume minimum : **{MIN_SC_FOR_ALERT} SC**")
    st.write(f"- Alerte critique : **≥ {FREE_RATE_ALERT_HIGH:.0f}%**")
    st.write(f"- À surveiller : **≥ {FREE_RATE_ALERT_MEDIUM:.0f}%**")

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
    data_origin = "json_brut"

else:
    if not enriched_file:
        st.info("➡️ Charge un CSV enrichi.")
        st.stop()

    df = enriched_csv_to_df(enriched_file)
    if df.empty:
        st.warning("Aucune donnée dans le CSV enrichi.")
        st.stop()

    st.success("✅ CSV enrichi chargé")
    data_origin = "csv_enrichi"


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
        period_start = f4.selectbox("Mois début", all_months, index=0, key="period_start")
        period_end = f4.selectbox("Mois fin", all_months, index=len(all_months) - 1, key="period_end")

    f5, f6 = st.columns(2)
    free_filter = f5.selectbox("Facturation", ["Tous", "Facturables", "Gratuits"])
    state_filter = f6.selectbox("État", ["Tous", "Complétés", "Annulés", "Autres"])

df_f = df.copy()

if selected_clients:
    df_f = df_f[df_f["client"].isin(selected_clients)]
if selected_systems:
    df_f = df_f[df_f["systeme"].isin(selected_systems)]
if selected_categories:
    df_f = df_f[df_f["categorie"].isin(selected_categories)]

if period_start and period_end:
    start_p = pd.Period(period_start, freq="M")
    end_p = pd.Period(period_end, freq="M")
    if start_p > end_p:
        start_p, end_p = end_p, start_p
    df_f = df_f[df_f["year_month"].notna()]
    df_f = df_f[(df_f["year_month"] >= start_p) & (df_f["year_month"] <= end_p)]

if free_filter == "Facturables":
    df_f = df_f[df_f["price"] > 0]
elif free_filter == "Gratuits":
    df_f = df_f[df_f["price"] == 0]

if state_filter == "Complétés":
    df_f = df_f[df_f["is_completed"]]
elif state_filter == "Annulés":
    df_f = df_f[df_f["is_canceled"]]
elif state_filter == "Autres":
    df_f = df_f[~df_f["is_completed"] & ~df_f["is_canceled"]]

if df_f.empty:
    st.warning("Aucune donnée après application des filtres.")
    st.stop()


# =========================================================
# PERIOD CONTEXT
# =========================================================
months_covered = months_covered_from_periods(df_f["year_month"])
period_min = df_f["year_month"].dropna().min()
period_max = df_f["year_month"].dropna().max()
period_label = f"{period_min} → {period_max}" if pd.notna(period_min) and pd.notna(period_max) else "Période non définie"


# =========================================================
# PREP DATASETS
# =========================================================
total_tickets = len(df_f)
completed_df = df_f[df_f["is_completed"]]
closure_valid_df = df_f[df_f["closure_days"].notna() & (df_f["closure_days"] >= 0)]

df_gratuity_scope = df_f.copy()
if exclude_internal_from_gratuity:
    df_gratuity_scope = df_gratuity_scope[~df_gratuity_scope["client"].isin(internal_clients)]

free_df = df_gratuity_scope[df_gratuity_scope["is_free"]]
free_with_closure = free_df[free_df["closure_days"].notna() & (free_df["closure_days"] >= 0)]

total_revenue = float(df_f["price"].sum())
avg_cost_all = float(df_f["price"].mean()) if total_tickets else 0.0
avg_closure_days = float(closure_valid_df["closure_days"].mean()) if not closure_valid_df.empty else 0.0
avg_free_closure_days = float(free_with_closure["closure_days"].mean()) if not free_with_closure.empty else 0.0

estimated_avg_free_hours = avg_free_closure_days * CYCLE_DAY_TO_TECH_HOURS
free_hidden_cost = len(free_df) * estimated_avg_free_hours * hourly_rate
hidden_cost_vs_revenue = (free_hidden_cost / total_revenue * 100) if total_revenue > 0 else 0.0

hors_ligne_pct = 100 * df_f["type_probleme"].eq("Hors ligne").mean() if total_tickets else 0.0
control4_pct = 100 * df_f["systeme"].eq("Control4").mean() if total_tickets else 0.0
unifi_pct = 100 * df_f["systeme"].eq("Unifi").mean() if total_tickets else 0.0
free_rate = 100 * free_df["is_free"].mean() if len(df_gratuity_scope) > 0 else 0.0

avg_sc_per_month = total_tickets / months_covered if months_covered > 0 else 0.0
avg_revenue_per_month = total_revenue / months_covered if months_covered > 0 else 0.0
avg_free_hidden_cost_per_month = free_hidden_cost / months_covered if months_covered > 0 else 0.0

df_exp = df_f.copy()
df_exp["resourceIds"] = df_exp["resourceIds"].apply(lambda x: x if isinstance(x, list) else [])
df_exp = df_exp.explode("resourceIds")
df_exp = df_exp[df_exp["resourceIds"].notna()]

if not df_exp.empty:
    tech_summary = (
        df_exp.groupby("resourceIds")
        .agg(
            sc_count=("sc_number", "count"),
            revenue=("price", "sum"),
            free_sc=("is_free", "sum"),
            avg_price=("price", "mean"),
        )
        .reset_index()
        .rename(columns={"resourceIds": "tech_id"})
        .sort_values(["sc_count", "revenue"], ascending=[False, False])
    )
    tech_summary["technicien"] = tech_summary["tech_id"].map(TECH_MAP).fillna(tech_summary["tech_id"].astype(str))
else:
    tech_summary = pd.DataFrame(columns=["tech_id", "technicien", "sc_count", "revenue", "free_sc", "avg_price"])

client_summary = (
    df_f.groupby("client", dropna=False)
    .agg(
        sc_count=("sc_number", "count"),
        revenue=("price", "sum"),
        free_sc=("is_free", "sum"),
        avg_price=("price", "mean"),
    )
    .reset_index()
)
client_summary["free_rate_pct"] = (client_summary["free_sc"] / client_summary["sc_count"] * 100).round(1)
client_summary["estimated_hidden_cost"] = (client_summary["free_sc"] * estimated_avg_free_hours * hourly_rate).round(2)
client_summary["is_internal_client"] = client_summary["client"].isin(internal_clients)
client_summary["free_risk_level"] = client_summary.apply(
    lambda row: "Volume faible" if row["sc_count"] < MIN_SC_FOR_ALERT else (
        "Critique" if row["free_rate_pct"] >= FREE_RATE_ALERT_HIGH else (
            "À surveiller" if row["free_rate_pct"] >= FREE_RATE_ALERT_MEDIUM else "Normal"
        )
    ),
    axis=1
)
client_summary = client_summary.sort_values(["sc_count", "revenue"], ascending=[False, False])

client_summary_for_risk = client_summary.copy()
if exclude_internal_from_gratuity:
    client_summary_for_risk = client_summary_for_risk[~client_summary_for_risk["is_internal_client"]]

free_risk_clients = client_summary_for_risk[
    (client_summary_for_risk["sc_count"] >= MIN_SC_FOR_ALERT) &
    (client_summary_for_risk["free_rate_pct"] >= FREE_RATE_ALERT_MEDIUM)
].copy()
free_risk_clients = free_risk_clients.sort_values(
    ["free_rate_pct", "free_sc", "sc_count"],
    ascending=[False, False, False]
)

problem_summary = (
    df_f.groupby("type_probleme", dropna=False)
    .agg(sc_count=("sc_number", "count"), revenue=("price", "sum"))
    .reset_index()
    .sort_values("sc_count", ascending=False)
)

system_summary = (
    df_f.groupby("systeme", dropna=False)
    .agg(sc_count=("sc_number", "count"), revenue=("price", "sum"))
    .reset_index()
    .sort_values("sc_count", ascending=False)
)

reason_summary = (
    free_df["free_reason_hint"]
    .fillna("Non précisé")
    .value_counts()
    .rename_axis("raison")
    .reset_index(name="count")
)
if not reason_summary.empty:
    reason_summary["part_pct"] = (reason_summary["count"] / reason_summary["count"].sum() * 100).round(1)

monthly_overall = build_monthly_overall(df_f)
monthly_gratuity = build_monthly_gratuity(df_gratuity_scope)
monthly_metrics = merge_monthly_tables(monthly_overall, monthly_gratuity)

metric_cols = [
    "sc_count", "revenue", "avg_price", "avg_closure_days",
    "hors_ligne_pct", "control4_pct", "unifi_pct", "cancel_rate_pct",
    "free_sc", "free_rate_pct", "estimated_hidden_cost"
]
monthly_metrics = add_change_columns(monthly_metrics, [c for c in metric_cols if c in monthly_metrics.columns])

anomalies_df = detect_monthly_anomalies(monthly_metrics)

monthly_sc = (
    df_f[df_f["month_label"] != "Sans date"]
    .groupby("month_label")
    .size()
    .reset_index(name="tickets")
    .sort_values("month_label")
)

monthly_revenue = (
    df_f[df_f["month_label"] != "Sans date"]
    .groupby("month_label")["price"]
    .sum()
    .reset_index(name="revenue")
    .sort_values("month_label")
)

monthly_free = (
    df_gratuity_scope[df_gratuity_scope["month_label"] != "Sans date"]
    .groupby("month_label")["is_free"]
    .mean()
    .mul(100)
    .reset_index(name="free_rate_pct")
    .sort_values("month_label")
)

top_problem_name = problem_summary.iloc[0]["type_probleme"] if not problem_summary.empty else "N/A"
top_problem_count = int(problem_summary.iloc[0]["sc_count"]) if not problem_summary.empty else 0
top_system_name = system_summary.iloc[0]["systeme"] if not system_summary.empty else "N/A"
top_system_count = int(system_summary.iloc[0]["sc_count"]) if not system_summary.empty else 0
top_client_name = client_summary.iloc[0]["client"] if not client_summary.empty else "N/A"
top_client_count = int(client_summary.iloc[0]["sc_count"]) if not client_summary.empty else 0
free_risk_count = len(free_risk_clients)
top_free_risk_client = free_risk_clients.iloc[0]["client"] if not free_risk_clients.empty else "N/A"
top_free_risk_rate = float(free_risk_clients.iloc[0]["free_rate_pct"]) if not free_risk_clients.empty else 0.0

latest_mom_sc = monthly_metrics.iloc[-1]["sc_count_mom_pct"] if len(monthly_metrics) and "sc_count_mom_pct" in monthly_metrics.columns else None
latest_mom_revenue = monthly_metrics.iloc[-1]["revenue_mom_pct"] if len(monthly_metrics) and "revenue_mom_pct" in monthly_metrics.columns else None
latest_yoy_sc = monthly_metrics.iloc[-1]["sc_count_yoy_pct"] if len(monthly_metrics) and "sc_count_yoy_pct" in monthly_metrics.columns else None
latest_yoy_revenue = monthly_metrics.iloc[-1]["revenue_yoy_pct"] if len(monthly_metrics) and "revenue_yoy_pct" in monthly_metrics.columns else None
last_12_vs_prev_12_sc = compare_last_12_vs_prev_12(monthly_metrics, "sc_count") if "sc_count" in monthly_metrics.columns else None
last_12_vs_prev_12_revenue = compare_last_12_vs_prev_12(monthly_metrics, "revenue") if "revenue" in monthly_metrics.columns else None

# =========================================================
# RÉSUMÉ EXÉCUTIF
# =========================================================
summary_markdown = f"""
### Résumé exécutif
- **Période analysée : {period_label}** (**{months_covered} mois** couverts).
- **{total_tickets:,} tickets** analysés pour un **revenu total de {total_revenue:,.0f} $**.
- Les appels **gratuits représentent {free_rate:.1f}%** du volume sur le périmètre gratuité retenu.
- Le **coût caché estimé** du SAV gratuit est de **{free_hidden_cost:,.0f} $**, soit **{hidden_cost_vs_revenue:.1f}% du revenu service analysé**.
- Le problème le plus fréquent est **{top_problem_name}** avec **{top_problem_count:,} tickets**.
- Le système le plus présent est **{top_system_name}** avec **{top_system_count:,} tickets**.
- Le client le plus demandeur est **{top_client_name}** avec **{top_client_count:,} tickets**.
- Le **cycle moyen** des tickets est de **{avg_closure_days:.1f} jours**.
- Les tickets **hors ligne** représentent **{hors_ligne_pct:.1f}%** du volume.
- Évolution récente : **SC {fmt_delta_pct(latest_mom_sc)}** et **revenu {fmt_delta_pct(latest_mom_revenue)}** vs mois précédent.
- Vs même mois l’an dernier : **SC {fmt_delta_pct(latest_yoy_sc)}** et **revenu {fmt_delta_pct(latest_yoy_revenue)}**.
- Comparatif **12 mois glissants vs 12 mois précédents** : **SC {fmt_delta_pct(last_12_vs_prev_12_sc)}** et **revenu {fmt_delta_pct(last_12_vs_prev_12_revenue)}**.
- Les **clients internes sont {'exclus' if exclude_internal_from_gratuity else 'inclus'}** du périmètre gratuité et du calcul du coût caché.
""".strip()

if show_exec_summary:
    st.markdown(summary_markdown)

# =========================================================
# CONTEXTE DE PÉRIODE
# =========================================================
st.subheader("🗓️ Contexte de période")

pc1, pc2 = st.columns(2)
pc1.metric("Période analysée", period_label)
pc2.metric("Mois couverts", f"{months_covered}")

# =========================================================
# BLOC 1 : VUE PÉRIODE
# =========================================================
st.subheader("📌 Bloc 1 — Vue période analysée")
st.caption("Indicateurs cumulés sur la période filtrée.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Tickets totaux", f"{total_tickets:,}")
c2.metric("Revenu total", f"{total_revenue:,.0f} $")
c3.metric("SC gratuits", f"{len(free_df):,}", f"{free_rate:.1f}%")
c4.metric("Coût moyen / SC", f"{avg_cost_all:,.0f} $")
c5.metric("Cycle moyen", f"{avg_closure_days:.1f} j")
c6.metric("Coût caché total", f"{free_hidden_cost:,.0f} $", f"{hidden_cost_vs_revenue:.1f}% du revenu")

c7, c8, c9, c10, c11, c12 = st.columns(6)
c7.metric("SC moyens / mois", f"{avg_sc_per_month:,.1f}")
c8.metric("Revenu moyen / mois", f"{avg_revenue_per_month:,.0f} $")
c9.metric("Coût caché moyen / mois", f"{avg_free_hidden_cost_per_month:,.0f} $")
c10.metric("Hors ligne", f"{hors_ligne_pct:.1f}%")
c11.metric("Clients à risque", f"{free_risk_count:,}")
c12.metric("Clients internes exclus", "Oui" if exclude_internal_from_gratuity else "Non")

# =========================================================
# BLOC 2 : VUE MOIS SÉLECTIONNÉ
# =========================================================
st.subheader("📅 Bloc 2 — Vue mois sélectionné")
st.caption("Indicateurs du mois choisi, avec comparaison.")

selected_month_view = None
selected_idx = None
comparison_mode = None

if monthly_metrics.empty:
    st.info("Pas assez de données mensuelles pour afficher le bloc mensuel.")
else:
    cm1, cm2 = st.columns(2)
    available_months = monthly_metrics["year_month_str"].tolist()

    selected_month_view = cm1.selectbox(
        "Mois sélectionné",
        options=available_months,
        index=len(available_months) - 1
    )

    comparison_mode = cm2.selectbox(
        "Mode de comparaison",
        options=[
            "vs mois précédent",
            "vs même mois année précédente",
            "vs moyenne 3 mois précédents",
            "vs moyenne 6 mois précédents",
            "vs moyenne 12 mois précédents",
            "3 mois glissants vs 3 mois précédents",
            "6 mois glissants vs 6 mois précédents",
            "12 mois glissants vs 12 mois précédents",
        ]
    )

    selected_row = monthly_metrics[monthly_metrics["year_month_str"] == selected_month_view]
    if not selected_row.empty:
        selected_idx = selected_row.index[0]

if monthly_metrics.empty or selected_idx is None:
    st.info("Pas assez de données pour la vue mois.")
else:
    metrics_to_show = [
        ("sc_count", "SC"),
        ("revenue", "Revenu"),
        ("free_rate_pct", "Taux gratuit"),
        ("estimated_hidden_cost", "Coût caché"),
        ("avg_price", "Revenu moyen / SC"),
        ("avg_closure_days", "Cycle moyen"),
        ("hors_ligne_pct", "Part hors ligne"),
        ("cancel_rate_pct", "Taux annulé"),
    ]

    values = {}
    for key, _ in metrics_to_show:
        if key in monthly_metrics.columns:
            values[key] = compare_selected_period(monthly_metrics, selected_idx, key, comparison_mode)
        else:
            values[key] = (None, None)

    m1, m2, m3, m4 = st.columns(4)
    cur, delta = build_metric_display(*values["sc_count"], "sc_count")
    m1.metric("SC du mois", cur, delta)
    cur, delta = build_metric_display(*values["revenue"], "revenue")
    m2.metric("Revenu du mois", cur, delta)
    cur, delta = build_metric_display(*values["free_rate_pct"], "free_rate_pct")
    m3.metric("Taux gratuit du mois", cur, delta)
    cur, delta = build_metric_display(*values["estimated_hidden_cost"], "estimated_hidden_cost")
    m4.metric("Coût caché du mois", cur, delta)

    m5, m6, m7, m8 = st.columns(4)
    cur, delta = build_metric_display(*values["avg_price"], "avg_price")
    m5.metric("Revenu moyen / SC", cur, delta)
    cur, delta = build_metric_display(*values["avg_closure_days"], "avg_closure_days")
    m6.metric("Cycle moyen", cur, delta)
    cur, delta = build_metric_display(*values["hors_ligne_pct"], "hors_ligne_pct")
    m7.metric("Part hors ligne", cur, delta)
    cur, delta = build_metric_display(*values["cancel_rate_pct"], "cancel_rate_pct")
    m8.metric("Taux annulé", cur, delta)

    st.caption(f"Référence : {selected_month_view} • {comparison_mode}")

# =========================================================
# GRAPHIQUES DE TENDANCE
# =========================================================
st.divider()
st.subheader("📈 Tendances mensuelles")

fig_roll_sc = None
fig_roll_rev = None
fig_roll_free = None
fig_roll_hidden = None

if not monthly_metrics.empty:
    r1, r2 = st.columns(2)

    fig_roll_sc = px.line(
        monthly_metrics,
        x="year_month_str",
        y=["sc_count", "sc_count_roll12"],
        title="Services mensuels vs tendance 12 mois",
        labels={
            "year_month_str": "Mois",
            "value": "Nombre de services",
            "variable": "Série",
        },
        markers=True
    )
    fig_roll_sc = rename_traces(fig_roll_sc, {
        "sc_count": "Services mensuels",
        "sc_count_roll12": "Tendance 12 mois",
    })
    fig_roll_sc = style_line_chart(fig_roll_sc)
    r1.plotly_chart(fig_roll_sc, use_container_width=True)

    fig_roll_rev = px.line(
        monthly_metrics,
        x="year_month_str",
        y=["revenue", "revenue_roll12"],
        title="Revenu mensuel vs tendance 12 mois",
        labels={
            "year_month_str": "Mois",
            "value": "Revenu ($)",
            "variable": "Série",
        },
        markers=True
    )
    fig_roll_rev = rename_traces(fig_roll_rev, {
        "revenue": "Revenu mensuel",
        "revenue_roll12": "Tendance 12 mois",
    })
    fig_roll_rev = style_line_chart(fig_roll_rev)
    r2.plotly_chart(fig_roll_rev, use_container_width=True)

    r3, r4 = st.columns(2)

    if "free_rate_pct" in monthly_metrics.columns:
        fig_roll_free = px.line(
            monthly_metrics,
            x="year_month_str",
            y=["free_rate_pct", "free_rate_pct_roll12"],
            title="Taux de gratuité vs tendance 12 mois",
            labels={
                "year_month_str": "Mois",
                "value": "Taux de gratuité (%)",
                "variable": "Série",
            },
            markers=True
        )
        fig_roll_free = rename_traces(fig_roll_free, {
            "free_rate_pct": "Taux gratuit mensuel",
            "free_rate_pct_roll12": "Tendance 12 mois",
        })
        fig_roll_free = style_line_chart(fig_roll_free)
        r3.plotly_chart(fig_roll_free, use_container_width=True)

    if "estimated_hidden_cost" in monthly_metrics.columns:
        fig_roll_hidden = px.line(
            monthly_metrics,
            x="year_month_str",
            y=["estimated_hidden_cost", "estimated_hidden_cost_roll12"],
            title="Coût caché mensuel vs tendance 12 mois",
            labels={
                "year_month_str": "Mois",
                "value": "Coût caché ($)",
                "variable": "Série",
            },
            markers=True
        )
        fig_roll_hidden = rename_traces(fig_roll_hidden, {
            "estimated_hidden_cost": "Coût caché mensuel",
            "estimated_hidden_cost_roll12": "Tendance 12 mois",
        })
        fig_roll_hidden = style_line_chart(fig_roll_hidden)
        r4.plotly_chart(fig_roll_hidden, use_container_width=True)

# =========================================================
# ANOMALIES
# =========================================================
st.divider()
st.subheader("⚠️ Détection automatique des anomalies mensuelles")

if anomalies_df.empty:
    st.success("Aucune anomalie mensuelle significative détectée selon les seuils actuels.")
    fig_anomaly = None
else:
    st.dataframe(anomalies_df.head(20), use_container_width=True, height=280)

    anomaly_counts = (
        anomalies_df["metric"]
        .value_counts()
        .rename_axis("metric")
        .reset_index(name="count")
    )
    fig_anomaly = px.bar(
        anomaly_counts,
        x="metric",
        y="count",
        title="Nombre d'anomalies détectées par métrique",
        labels={
            "metric": "Métrique",
            "count": "Nombre d'anomalies"
        },
        text="count"
    )
    fig_anomaly = style_bar_chart(fig_anomaly, money=False)
    st.plotly_chart(fig_anomaly, use_container_width=True)

# =========================================================
# CHARTS
# =========================================================
st.divider()

fig_systems = px.pie(
    df_f,
    names="systeme",
    title="Répartition des systèmes",
    labels={"systeme": "Système"}
)
fig_systems = style_pie_chart(fig_systems)

cat_count = (
    df_f["categorie"]
    .value_counts(dropna=False)
    .rename_axis("categorie")
    .reset_index(name="count")
)
fig_categories = px.bar(
    cat_count,
    x="categorie",
    y="count",
    title="Répartition par catégorie",
    labels={
        "categorie": "Catégorie",
        "count": "Nombre de services"
    },
    text="count"
)
fig_categories = style_bar_chart(fig_categories, money=False)

fig_monthly_sc = px.line(
    monthly_sc,
    x="month_label",
    y="tickets",
    title="Services par mois",
    labels={
        "month_label": "Mois",
        "tickets": "Nombre de services"
    },
    markers=True
)
fig_monthly_sc = rename_traces(fig_monthly_sc, {
    "tickets": "Services"
})
fig_monthly_sc = style_line_chart(fig_monthly_sc)

fig_monthly_revenue = px.bar(
    monthly_revenue,
    x="month_label",
    y="revenue",
    title="Revenu par mois",
    labels={
        "month_label": "Mois",
        "revenue": "Revenu ($)"
    }
)
fig_monthly_revenue = style_bar_chart(fig_monthly_revenue, money=True)

fig_top_problems = px.bar(
    problem_summary.head(10),
    x="type_probleme",
    y="sc_count",
    title="Top problèmes",
    labels={
        "type_probleme": "Type de problème",
        "sc_count": "Nombre de services"
    },
    text="sc_count"
)
fig_top_problems = style_bar_chart(fig_top_problems, money=False)

fig_top_clients = px.bar(
    client_summary.head(10),
    x="client",
    y="sc_count",
    title="Top clients — nombre de services",
    labels={
        "client": "Client",
        "sc_count": "Nombre de services"
    },
    text="sc_count"
)
fig_top_clients = style_bar_chart(fig_top_clients, money=False)
fig_top_clients.update_layout(xaxis_tickangle=-35)

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.plotly_chart(fig_systems, use_container_width=True)
with row1_col2:
    st.plotly_chart(fig_categories, use_container_width=True)

st.divider()
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.plotly_chart(fig_monthly_sc, use_container_width=True)
with row2_col2:
    st.plotly_chart(fig_monthly_revenue, use_container_width=True)

st.divider()
row3_col1, row3_col2 = st.columns(2)
with row3_col1:
    st.plotly_chart(fig_top_problems, use_container_width=True)
with row3_col2:
    st.plotly_chart(fig_top_clients, use_container_width=True)

st.divider()
row4_col1, row4_col2 = st.columns(2)
with row4_col1:
    fig_clients_rev = px.bar(
        client_summary.sort_values("revenue", ascending=False).head(10),
        x="client",
        y="revenue",
        title="Top clients — revenu",
        labels={
            "client": "Client",
            "revenue": "Revenu ($)"
        }
    )
    fig_clients_rev = style_bar_chart(fig_clients_rev, money=True)
    fig_clients_rev.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_clients_rev, use_container_width=True)

with row4_col2:
    if not tech_summary.empty:
        fig_tech = px.bar(
            tech_summary.head(10),
            x="technicien",
            y="sc_count",
            title="Techniciens — nombre d'interventions",
            labels={
                "technicien": "Technicien",
                "sc_count": "Nombre d'interventions"
            },
            text="sc_count"
        )
        fig_tech = style_bar_chart(fig_tech, money=False)
        st.plotly_chart(fig_tech, use_container_width=True)
    else:
        st.info("Aucune donnée technicien exploitable.")
        fig_tech = None

st.divider()
row5_col1, row5_col2 = st.columns(2)
with row5_col1:
    if not tech_summary.empty:
        fig_tech_rev = px.bar(
            tech_summary.sort_values("revenue", ascending=False).head(10),
            x="technicien",
            y="revenue",
            title="Techniciens — revenu généré",
            labels={
                "technicien": "Technicien",
                "revenue": "Revenu ($)"
            }
        )
        fig_tech_rev = style_bar_chart(fig_tech_rev, money=True)
        st.plotly_chart(fig_tech_rev, use_container_width=True)
    else:
        st.info("Aucune donnée technicien exploitable.")
        fig_tech_rev = None

with row5_col2:
    fig_free = px.line(
        monthly_free,
        x="month_label",
        y="free_rate_pct",
        title="Taux gratuit mensuel",
        labels={
            "month_label": "Mois",
            "free_rate_pct": "Taux gratuit (%)"
        },
        markers=True
    )
    fig_free = rename_traces(fig_free, {
        "free_rate_pct": "Taux gratuit"
    })
    fig_free = style_line_chart(fig_free)
    st.plotly_chart(fig_free, use_container_width=True)

# =========================================================
# COÛT CACHÉ
# =========================================================
st.divider()
st.subheader("💸 Coût caché estimé du SAV gratuit")

cc1, cc2, cc3, cc4, cc5 = st.columns(5)
cc1.metric("Nombre de SC gratuits", f"{len(free_df):,}")
cc2.metric("Cycle moyen SC gratuits", f"{avg_free_closure_days:.1f} j")
cc3.metric("Heures estimées / SC", f"{estimated_avg_free_hours:.2f} h")
cc4.metric("Coût horaire", f"{hourly_rate:,.0f} $")
cc5.metric("Coût caché total", f"{free_hidden_cost:,.0f} $")

st.caption(
    "Le coût caché estimé du SAV gratuit suit le même périmètre que l'analyse de gratuité "
    f"({'clients internes exclus' if exclude_internal_from_gratuity else 'clients internes inclus'})."
)

cost_by_reason = reason_summary.copy()
if not cost_by_reason.empty:
    cost_by_reason["estimated_hidden_cost"] = cost_by_reason["count"] * estimated_avg_free_hours * hourly_rate
    fig_cost_reason = px.bar(
        cost_by_reason,
        x="raison",
        y="estimated_hidden_cost",
        hover_data=["count", "part_pct"],
        title="Répartition estimée du coût caché des SC gratuits par raison",
        labels={
            "raison": "Raison",
            "estimated_hidden_cost": "Coût caché estimé ($)",
            "count": "Nombre de SC",
            "part_pct": "Part (%)"
        }
    )
    fig_cost_reason = style_bar_chart(fig_cost_reason, money=True)
    st.plotly_chart(fig_cost_reason, use_container_width=True)
else:
    fig_cost_reason = None

# =========================================================
# CLIENTS À RISQUE
# =========================================================
st.divider()
st.subheader("🚨 Détection automatique des clients à forte gratuité")

risk_k1, risk_k2, risk_k3, risk_k4 = st.columns(4)
risk_k1.metric("Clients à risque", f"{free_risk_count}")
risk_k2.metric("Plus fort taux gratuité", f"{top_free_risk_rate:.1f}%" if free_risk_count > 0 else "0.0%")
risk_k3.metric("Client le plus exposé", f"{top_free_risk_client}" if free_risk_count > 0 else "N/A")
risk_k4.metric("Clients internes exclus", "Oui" if exclude_internal_from_gratuity else "Non")

st.caption(
    f"Détection basée sur au moins {MIN_SC_FOR_ALERT} SC par client et un taux de gratuité ≥ {FREE_RATE_ALERT_MEDIUM:.0f}%."
)

fig_risk_clients = None
if not free_risk_clients.empty:
    fig_risk_clients = px.bar(
        free_risk_clients.head(15),
        x="client",
        y="free_rate_pct",
        color="free_risk_level",
        hover_data=["sc_count", "free_sc", "revenue", "estimated_hidden_cost", "is_internal_client"],
        title="Clients à forte gratuité — taux de gratuité",
        labels={
            "client": "Client",
            "free_rate_pct": "Taux de gratuité (%)",
            "free_risk_level": "Niveau de risque",
            "sc_count": "Nombre de SC",
            "free_sc": "SC gratuits",
            "revenue": "Revenu",
            "estimated_hidden_cost": "Coût caché estimé",
            "is_internal_client": "Client interne"
        },
        text="free_rate_pct"
    )
    fig_risk_clients.update_traces(texttemplate="%{y:.1f}%", textposition="outside", cliponaxis=False)
    fig_risk_clients.update_layout(xaxis_title=None, yaxis_title=None, xaxis_tickangle=-35, legend_title_text="")
    st.plotly_chart(fig_risk_clients, use_container_width=True)

    st.dataframe(
        free_risk_clients[
            [
                "client", "sc_count", "free_sc", "free_rate_pct",
                "revenue", "estimated_hidden_cost", "free_risk_level",
                "is_internal_client"
            ]
        ],
        use_container_width=True,
        height=320
    )
else:
    st.success("Aucun client à forte gratuité détecté selon les seuils actuels.")

# =========================================================
# CLOSURE / HEATMAP
# =========================================================
st.divider()
if not closure_valid_df.empty:
    fig_closure = px.histogram(
        closure_valid_df,
        x="closure_days",
        nbins=20,
        title="Distribution du cycle ticket",
        labels={
            "closure_days": "Jours",
            "count": "Nombre de tickets"
        }
    )
    fig_closure.update_layout(xaxis_tickangle=-45, bargap=0.2, showlegend=False)
    st.plotly_chart(fig_closure, use_container_width=True)
else:
    fig_closure = None

if show_heatmap:
    cross = pd.crosstab(df_f["type_probleme"], df_f["categorie"])
    fig_heatmap = px.imshow(
        cross,
        text_auto=True,
        title="Heatmap — Type de problème × Catégorie",
        labels=dict(x="Catégorie", y="Type de problème", color="Nombre")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    fig_heatmap = None

# =========================================================
# BUBBLE CLIENTS
# =========================================================
if show_client_bubble:
    st.divider()
    bubble_df = client_summary[client_summary["sc_count"] >= 2].copy()
    if not bubble_df.empty:
        fig_bubble = px.scatter(
            bubble_df,
            x="sc_count",
            y="revenue",
            size="free_sc",
            hover_name="client",
            color="free_rate_pct",
            title="Clients — volume SAV vs revenu vs gratuité",
            labels={
                "sc_count": "Nombre de services",
                "revenue": "Revenu total ($)",
                "free_rate_pct": "Taux gratuit (%)",
                "free_sc": "SC gratuits"
            }
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        fig_bubble = None
else:
    fig_bubble = None

# =========================================================
# DRILL DOWN
# =========================================================
st.divider()
st.subheader("🔎 Analyse détaillée")

analysis_mode = st.selectbox(
    "Axe d'analyse",
    ["Client", "Problème", "Système", "Technicien", "Mois"]
)

detail_df = df_f.copy()

if analysis_mode == "Client":
    options = sorted([x for x in detail_df["client"].dropna().unique().tolist()])
    selected_value = st.selectbox("Sélection", options)
    subset = detail_df[detail_df["client"] == selected_value]

elif analysis_mode == "Problème":
    options = sorted([x for x in detail_df["type_probleme"].dropna().unique().tolist()])
    selected_value = st.selectbox("Sélection", options)
    subset = detail_df[detail_df["type_probleme"] == selected_value]

elif analysis_mode == "Système":
    options = sorted([x for x in detail_df["systeme"].dropna().unique().tolist()])
    selected_value = st.selectbox("Sélection", options)
    subset = detail_df[detail_df["systeme"] == selected_value]

elif analysis_mode == "Technicien":
    df_exp2 = detail_df.copy()
    df_exp2["resourceIds"] = df_exp2["resourceIds"].apply(lambda x: x if isinstance(x, list) else [])
    df_exp2 = df_exp2.explode("resourceIds")
    df_exp2 = df_exp2[df_exp2["resourceIds"].notna()].copy()
    df_exp2["technicien"] = df_exp2["resourceIds"].map(TECH_MAP).fillna(df_exp2["resourceIds"].astype(str))
    options = sorted(df_exp2["technicien"].dropna().unique().tolist())
    selected_value = st.selectbox("Sélection", options)
    subset = df_exp2[df_exp2["technicien"] == selected_value]

else:
    options = sorted([x for x in detail_df["month_label"].dropna().unique().tolist() if x != "Sans date"])
    selected_value = st.selectbox("Sélection", options)
    subset = detail_df[detail_df["month_label"] == selected_value]

sd1, sd2, sd3, sd4 = st.columns(4)
sd1.metric("SC", f"{len(subset):,}")
sd2.metric("Revenu", f"{subset['price'].sum():,.0f} $")
sd3.metric("Gratuits", f"{int(subset['is_free'].sum()):,}")
sd4.metric("Taux gratuit", f"{subset['is_free'].mean()*100:.1f}%" if len(subset) else "0.0%")

detail_cols = [
    "sc_number", "client", "createdOn", "paymentBillingDate", "closure_days", "price",
    "is_free", "free_reason_hint", "issue_text",
    "type_probleme", "categorie", "systeme", "systeme_suggere",
    "confiance_systeme", "justification_systeme", "resourceIds"
]
for c in detail_cols:
    if c not in subset.columns:
        subset[c] = None

st.dataframe(subset[detail_cols], use_container_width=True, height=300)

# =========================================================
# MINI PIVOT
# =========================================================
st.divider()
st.subheader("🧮 Mini croisé dynamique")

pivot_index = st.selectbox("Lignes", ["client", "systeme", "categorie", "type_probleme", "month_label"], index=0)
pivot_columns = st.selectbox("Colonnes", ["month_label", "categorie", "systeme", "is_free", "stateName"], index=0)
pivot_value = st.selectbox("Mesure", ["count_sc", "sum_revenue", "avg_price"], index=0)

pivot_df = df_f.copy()

if pivot_value == "count_sc":
    pivot = pd.pivot_table(
        pivot_df, index=pivot_index, columns=pivot_columns,
        values="sc_number", aggfunc="count", fill_value=0
    )
elif pivot_value == "sum_revenue":
    pivot = pd.pivot_table(
        pivot_df, index=pivot_index, columns=pivot_columns,
        values="price", aggfunc="sum", fill_value=0
    )
else:
    pivot = pd.pivot_table(
        pivot_df, index=pivot_index, columns=pivot_columns,
        values="price", aggfunc="mean", fill_value=0
    )

st.dataframe(pivot, use_container_width=True, height=320)

# =========================================================
# TABLEAUX
# =========================================================
st.divider()
st.subheader("📋 Synthèses exploitables")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Clients", "Techniciens", "Problèmes", "Systèmes", "Clients à risque", "Mensuel", "Anomalies"]
)

with tab1:
    st.dataframe(
        client_summary[
            [
                "client", "sc_count", "revenue", "free_sc", "free_rate_pct",
                "estimated_hidden_cost", "free_risk_level", "is_internal_client", "avg_price"
            ]
        ],
        use_container_width=True,
        height=350
    )

with tab2:
    if not tech_summary.empty:
        st.dataframe(
            tech_summary[["technicien", "tech_id", "sc_count", "revenue", "free_sc", "avg_price"]],
            use_container_width=True,
            height=350
        )
    else:
        st.info("Aucune donnée technicien.")

with tab3:
    st.dataframe(problem_summary, use_container_width=True, height=350)

with tab4:
    st.dataframe(system_summary, use_container_width=True, height=350)

with tab5:
    st.dataframe(
        free_risk_clients[
            [
                "client", "sc_count", "free_sc", "free_rate_pct",
                "revenue", "estimated_hidden_cost", "free_risk_level", "is_internal_client"
            ]
        ] if not free_risk_clients.empty else free_risk_clients,
        use_container_width=True,
        height=350
    )

with tab6:
    st.dataframe(monthly_metrics, use_container_width=True, height=350)

with tab7:
    st.dataframe(anomalies_df, use_container_width=True, height=350)

# =========================================================
# DÉTAIL TICKETS
# =========================================================
st.divider()
for c in detail_cols:
    if c not in df_f.columns:
        df_f[c] = None

if show_data_table:
    st.subheader("🧾 Détail des tickets")
    st.dataframe(df_f[detail_cols], use_container_width=True, height=500)

# =========================================================
# EXPORTS
# =========================================================
st.divider()
st.subheader("⬇️ Exports")

csv_detail = df_f[detail_cols].to_csv(index=False).encode("utf-8")

df_enriched_export = df.copy()
if "resourceIds" in df_enriched_export.columns:
    df_enriched_export["resourceIds"] = df_enriched_export["resourceIds"].apply(json.dumps)
csv_enriched = df_enriched_export.to_csv(index=False).encode("utf-8")

summary_df = pd.DataFrame([{
    "periode_analysee": period_label,
    "mois_couverts": months_covered,
    "tickets_totaux": total_tickets,
    "revenu_total": total_revenue,
    "sc_moyens_par_mois": round(avg_sc_per_month, 2),
    "revenu_moyen_par_mois": round(avg_revenue_per_month, 2),
    "cout_cache_moyen_par_mois": round(avg_free_hidden_cost_per_month, 2),
    "sc_gratuits": len(free_df),
    "taux_gratuit_pct": round(free_rate, 2),
    "cout_moyen_sc": round(avg_cost_all, 2),
    "cycle_moyen_jours": round(avg_closure_days, 2),
    "cycle_moyen_sc_gratuits_jours": round(avg_free_closure_days, 2),
    "heures_estimees_par_sc_gratuit": round(estimated_avg_free_hours, 2),
    "cout_horaire": hourly_rate,
    "cout_cache_estime": round(free_hidden_cost, 2),
    "cout_cache_vs_revenu_pct": round(hidden_cost_vs_revenue, 2),
    "hors_ligne_pct": round(hors_ligne_pct, 2),
    "control4_pct": round(control4_pct, 2),
    "unifi_pct": round(unifi_pct, 2),
    "clients_a_risque_gratuite": free_risk_count,
    "top_probleme": top_problem_name,
    "top_systeme": top_system_name,
    "top_client": top_client_name,
    "top_client_risque_gratuite": top_free_risk_client,
    "top_client_risque_taux_pct": round(top_free_risk_rate, 2),
    "clients_internes_exclus_gratuite": exclude_internal_from_gratuity,
}])

excel_bytes = to_excel_bytes({
    "Résumé": summary_df,
    "Clients": client_summary[
        [
            "client", "sc_count", "revenue", "free_sc", "free_rate_pct",
            "estimated_hidden_cost", "free_risk_level", "is_internal_client", "avg_price"
        ]
    ],
    "Techniciens": tech_summary[
        ["technicien", "tech_id", "sc_count", "revenue", "free_sc", "avg_price"]
    ] if not tech_summary.empty else tech_summary,
    "Problèmes": problem_summary,
    "Systèmes": system_summary,
    "Clients à risque": free_risk_clients[
        [
            "client", "sc_count", "free_sc", "free_rate_pct",
            "revenue", "estimated_hidden_cost", "free_risk_level", "is_internal_client"
        ]
    ] if not free_risk_clients.empty else free_risk_clients,
    "Raisons gratuité": reason_summary,
    "Mensuel": monthly_metrics,
    "Anomalies": anomalies_df,
    "Mini pivot": pivot.reset_index(),
    "Détail tickets": df_f[detail_cols],
})

figures_for_html = [
    ("Répartition des systèmes", fig_systems),
    ("Répartition par catégorie", fig_categories),
    ("Services par mois", fig_monthly_sc),
    ("Revenu par mois", fig_monthly_revenue),
    ("Services vs tendance 12 mois", fig_roll_sc),
    ("Revenu vs tendance 12 mois", fig_roll_rev),
    ("Top problèmes", fig_top_problems),
    ("Top clients — nombre de services", fig_top_clients),
    ("Clients à forte gratuité", fig_risk_clients),
    ("Coût caché par raison", fig_cost_reason),
]

html_report_bytes = build_html_report(
    summary_markdown=summary_markdown,
    kpis=[
        ("Période analysée", period_label, f"{months_covered} mois"),
        ("Tickets totaux", f"{total_tickets:,}", ""),
        ("Revenu total", f"{total_revenue:,.0f} $", ""),
        ("SC moyens / mois", f"{avg_sc_per_month:,.1f}", ""),
        ("Revenu moyen / mois", f"{avg_revenue_per_month:,.0f} $", ""),
        ("Coût caché total", f"{free_hidden_cost:,.0f} $", f"{hidden_cost_vs_revenue:.1f}% du revenu"),
    ],
    figures=figures_for_html,
    title="Rapport SAV KPI"
)

e1, e2, e3, e4 = st.columns(4)

with e1:
    st.download_button(
        "⬇️ CSV détaillé filtré",
        csv_detail,
        file_name="rapport_kpi_detail.csv",
        mime="text/csv"
    )

with e2:
    st.download_button(
        "⬇️ CSV enrichi complet",
        csv_enriched,
        file_name="service_calls_enriched.csv",
        mime="text/csv"
    )

with e3:
    st.download_button(
        "⬇️ Excel multi-onglets",
        excel_bytes,
        file_name="rapport_kpi_v5_3.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with e4:
    st.download_button(
        "⬇️ Rapport HTML",
        html_report_bytes,
        file_name="rapport_kpi.html",
        mime="text/html"
    )