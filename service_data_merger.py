"""
service_data_merger.py
======================
Fusionne service_calls_last_extract.json dans service_calls_master.json.

Regles de merge :
  - Cle de deduplication : champ "number" (ex: SC-2041).
  - En cas de doublon, l'extract prend TOUJOURS la priorite sur le master.
  - Les anciens enregistrements uniquement dans le master sont conserves.
  - Les nouveaux enregistrements de l'extract sont ajoutes.
  - Aucun NaT / NaN ne doit apparaitre dans le JSON de sortie.
  - Les dates sont serializees au format ISO 8601.
"""

import json
import math

print("Starting service data merger...")

MASTER_FILE = "service_calls_master.json"
NEW_FILE    = "service_calls_last_extract.json"
OUTPUT_FILE = "service_calls_master.json"

# ─── Chargement brut (sans pandas) ───────────────────────────────────────────

def load_records(path: str) -> list:
    """Charge le fichier JSON et retourne une liste de dicts bruts."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"  Avertissement : '{path}' vide, invalide ou introuvable. Ignore.")
        return []

    if isinstance(data, dict) and "serviceCalls" in data:
        return data["serviceCalls"]
    if isinstance(data, list):
        return data
    return []


records_master = load_records(MASTER_FILE)
records_new    = load_records(NEW_FILE)

print(f"  Master : {len(records_master)} enregistrements")
print(f"  Extract: {len(records_new)} enregistrements")

if not records_new:
    print("Rien a fusionner.")
    exit(0)

# ─── Nettoyage des valeurs NaN/NaT ───────────────────────────────────────────
# json.loads traduit les "NaT" strings en str Python, mais l'ancien master avait
# aussi des NaN bare (produits par pandas). On nettoie recursivement.

def clean_value(val):
    """Convertit NaN float, 'NaT', 'NaN', 'nan' en None (-> null JSON)."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, str) and val.strip() in ("NaT", "NaN", "nan"):
        return None
    if isinstance(val, list):
        return [clean_value(v) for v in val]
    if isinstance(val, dict):
        return {k: clean_value(v) for k, v in val.items()}
    return val

def clean_record(rec: dict) -> dict:
    return {k: clean_value(v) for k, v in rec.items()}

# ─── Merge : extract prend la priorite ───────────────────────────────────────
# On construit un dict {number: record} en mettant d'abord le master,
# puis en ecrasant avec l'extract (les cles en commun sont remplacees,
# les nouvelles sont ajoutees, les anciennes non presentes dans l'extract
# sont conservees).

DEDUPE_KEY = "number"

merged: dict = {}

# 1) Charger le master (nettoyage inclus)
for rec in records_master:
    key = rec.get(DEDUPE_KEY)
    if key:
        merged[key] = clean_record(rec)

# 2) Ecraser / ajouter avec l'extract (priorite totale)
new_count      = 0
updated_count  = 0
for rec in records_new:
    key = rec.get(DEDUPE_KEY)
    if not key:
        continue
    cleaned = clean_record(rec)
    if key in merged:
        updated_count += 1
    else:
        new_count += 1
    merged[key] = cleaned

print(f"  Mis a jour  : {updated_count}")
print(f"  Nouveaux    : {new_count}")

# ─── Tri chronologique final ──────────────────────────────────────────────────

def sort_key(rec: dict) -> str:
    """Cle de tri sur createdOn ; '' si absent pour mettre en dernier."""
    v = rec.get("createdOn")
    return v if isinstance(v, str) and v else ""

final_records = sorted(merged.values(), key=sort_key)

# ─── Serialisation ───────────────────────────────────────────────────────────
# On utilise json.dump standard (pas de default=str) pour detecter
# immediatement tout type non serialisable residuel.

output = {"serviceCalls": final_records}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Master mis a jour : {len(final_records)} enregistrements -> {OUTPUT_FILE}")