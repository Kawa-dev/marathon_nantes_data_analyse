from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import os
import gpxpy
import gpxpy.gpx

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_2025 = os.path.join(BASE_DIR, 'data', 'marathon_nantes_2025_features.parquet')

# L'application cherche dans 'data', sinon dans téléchargements
DATA_2026_LOCAL = os.path.join(BASE_DIR, 'data', 'marathon_nantes_2026_features.parquet')
DATA_2026_DL = r"C:\Users\0302259T\Downloads\marathon_nantes_2026_features.parquet"
DATA_2026 = DATA_2026_LOCAL if os.path.exists(DATA_2026_LOCAL) else DATA_2026_DL

GPX_PATH = os.path.join(BASE_DIR, 'data', 'parcours.gpx')

def vitesse_to_allure(vitesse_kmh):
    if vitesse_kmh <= 0 or pd.isna(vitesse_kmh): return "--:--"
    allure_decimal = 60 / vitesse_kmh
    minutes = int(allure_decimal)
    secondes = int((allure_decimal - minutes) * 60)
    return f"{minutes:02d}:{secondes:02d}"

def safe_rank(val):
    try:
        v = int(val)
        return v if v > 0 else None
    except:
        return None

def extract_category_sex(name):
    try:
        if not isinstance(name, str): return 'SE', 'M'
        suffix = name.split('-')[-1].strip()
        parts = suffix.split()
        if len(parts) >= 2:
            return parts[-2].upper(), parts[-1].upper()
        elif len(parts) == 1:
            s = parts[0]
            return s[:-1].upper(), s[-1].upper()
    except:
        pass
    return 'SE', 'M'

def group_category(cat):
    cat = str(cat).upper().strip()
    if cat in ['CA', 'JU', 'ES', 'SE']: return 'Seniors & Jeunes (<35 ans)'
    if cat in ['M0', 'M1', 'M2', 'V1']: return 'Masters 0-2 (35-49 ans)'
    if cat in ['M3', 'M4', 'V2']: return 'Masters 3-4 (50-59 ans)'
    return 'Masters 5+ (60 ans et +)'

def load_gpx_track():
    if not os.path.exists(GPX_PATH): return []
    with open(GPX_PATH, 'r', encoding='utf-8') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    points = []
    total_dist = 0.0
    previous_point = None
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if previous_point:
                    total_dist += point.distance_2d(previous_point) / 1000.0 
                ele = point.elevation if point.elevation is not None else 0.0
                points.append({'lat': point.latitude, 'lng': point.longitude, 'dist': total_dist, 'ele': ele})
                previous_point = point
    return points

gpx_data = load_gpx_track()

def prepare_dataframe(filepath):
    if not os.path.exists(filepath): 
        return pd.DataFrame()
    df = pd.read_parquet(filepath).fillna(0)
    
    if 'Sexe' not in df.columns or 'Cat_Brute' not in df.columns:
        df[['Cat_Brute', 'Sexe']] = df['Nom'].apply(lambda x: pd.Series(extract_category_sex(x)))
        
    df['Macro_Cat'] = df['Cat_Brute'].apply(group_category)
    
    # DÉTECTION DES PUCES INCOMPLÈTES
    check_cols = ['Passage_KM10_sec', 'Passage_KM15_sec', 'Passage_KM21_sec', 'Passage_KM25_sec', 'Passage_KM30_sec', 'Passage_KM37_sec', 'Passage_KM40_sec']
    existing_cols = [c for c in check_cols if c in df.columns]
    df['is_complete'] = (df[existing_cols] > 0).all(axis=1)
    
    return df[df['Passage_ARRIVEE_sec'] > 0].copy()

def compute_base_metrics(df_finishers):
    if df_finishers.empty: return {}
    
    total_coureurs = len(df_finishers)
    vitesse_moy = df_finishers['Vitesse_kmh_ARRIVEE'].mean()
    vitesse_med = df_finishers['Vitesse_kmh_ARRIVEE'].median()
    vitesse_moy = vitesse_moy if not pd.isna(vitesse_moy) else 0
    vitesse_med = vitesse_med if not pd.isna(vitesse_med) else 0
    
    nb_femmes = len(df_finishers[df_finishers['Sexe'] == 'F'])
    pct_femmes = (nb_femmes / total_coureurs) * 100 if total_coureurs > 0 else 0
    
    is_negative_split = (df_finishers['Passage_ARRIVEE_sec'] - df_finishers['Passage_KM21_sec']) < df_finishers['Passage_KM21_sec']
    nb_neg = is_negative_split.sum()
    pct_neg = (nb_neg / total_coureurs) * 100 if total_coureurs > 0 else 0
    
    murs = (df_finishers['Derive_Allure_vs_Precedent_%_KM37'] >= 30).sum()
    pct_murs = (murs / total_coureurs) * 100 if total_coureurs > 0 else 0
    
    murs_15 = (df_finishers['Derive_Allure_vs_Precedent_%_KM37'] >= 15).sum()
    pct_murs_15 = (murs_15 / total_coureurs) * 100 if total_coureurs > 0 else 0
    
    # Vainqueur (Uniquement complets)
    df_complete = df_finishers[df_finishers['is_complete'] == True]
    if not df_complete.empty:
        if 'Classement_ARRIVEE' in df_complete.columns:
            vainqueur_idx = df_complete['Classement_ARRIVEE'].idxmin()
        else:
            vainqueur_idx = df_complete['Passage_ARRIVEE_sec'].idxmin()
        vainqueur_dos = df_complete.loc[vainqueur_idx, 'Dossard']
    else:
        vainqueur_dos = ""

    return {
        'total': total_coureurs,
        'vitesse_moy': vitesse_moy,
        'vitesse_med': vitesse_med,
        'pct_femmes': pct_femmes,
        'pct_neg': pct_neg,
        'pct_murs': pct_murs,
        'pct_murs_15': pct_murs_15,
        'vainqueur_dos': vainqueur_dos
    }

def compute_dashboard(df_finishers, year, metrics_prev_year=None):
    if df_finishers.empty: return None
    
    base_metrics = compute_base_metrics(df_finishers)
    
    deltas = {}
    if metrics_prev_year:
        deltas = {
            'total': base_metrics['total'] - metrics_prev_year['total'],
            'vitesse_moy': base_metrics['vitesse_moy'] - metrics_prev_year['vitesse_moy'],
            'pct_femmes': base_metrics['pct_femmes'] - metrics_prev_year['pct_femmes'],
            'pct_neg': base_metrics['pct_neg'] - metrics_prev_year['pct_neg'],
            'pct_murs': base_metrics['pct_murs'] - metrics_prev_year['pct_murs'],
            'pct_murs_15': base_metrics['pct_murs_15'] - metrics_prev_year['pct_murs_15']
        }

    # --- TOP 10 NEGATIVE SPLIT ---
    df_valid_split = df_finishers[(df_finishers['is_complete'] == True) & (df_finishers['Passage_KM21_sec'] > 0) & (df_finishers['Passage_ARRIVEE_sec'] > df_finishers['Passage_KM21_sec'])].copy()
    top_neg_splits = []
    if not df_valid_split.empty:
        df_valid_split['Semi1_sec'] = df_valid_split['Passage_KM21_sec']
        df_valid_split['Semi2_sec'] = df_valid_split['Passage_ARRIVEE_sec'] - df_valid_split['Passage_KM21_sec']
        df_valid_split['Vitesse_Semi1'] = 21.1 / (df_valid_split['Semi1_sec'] / 3600)
        df_valid_split['Vitesse_Semi2'] = 21.095 / (df_valid_split['Semi2_sec'] / 3600)
        df_valid_split['Gain_Vitesse_Pct'] = ((df_valid_split['Vitesse_Semi2'] - df_valid_split['Vitesse_Semi1']) / df_valid_split['Vitesse_Semi1']) * 100
        
        top_negative_splits_df = df_valid_split.sort_values(by='Gain_Vitesse_Pct', ascending=False).head(10)
        
        for _, row in top_negative_splits_df.iterrows():
            s1_h, s1_rem = divmod(row['Semi1_sec'], 3600)
            s1_m, s1_s = divmod(s1_rem, 60)
            time_s1 = f"{int(s1_h):02d}:{int(s1_m):02d}:{int(s1_s):02d}" if s1_h > 0 else f"{int(s1_m):02d}:{int(s1_s):02d}"
            
            s2_h, s2_rem = divmod(row['Semi2_sec'], 3600)
            s2_m, s2_s = divmod(s2_rem, 60)
            time_s2 = f"{int(s2_h):02d}:{int(s2_m):02d}:{int(s2_s):02d}" if s2_h > 0 else f"{int(s2_m):02d}:{int(s2_s):02d}"

            tot_h, tot_rem = divmod(row['Passage_ARRIVEE_sec'], 3600)
            tot_m, tot_s = divmod(tot_rem, 60)
            time_tot = f"{int(tot_h):02d}:{int(tot_m):02d}:{int(tot_s):02d}"
            
            top_neg_splits.append({
                'Nom': str(row['Nom']).split('-')[0].strip(),
                'Classement': safe_rank(row.get('Classement_ARRIVEE', 0)),
                'Temps_Final': time_tot,
                'Temps_S1': time_s1,
                'Temps_S2': time_s2,
                'Vitesse_S1_kmh': f"{row['Vitesse_Semi1']:.2f}",
                'Allure_S1': vitesse_to_allure(row['Vitesse_Semi1']),
                'Vitesse_S2_kmh': f"{row['Vitesse_Semi2']:.2f}",
                'Allure_S2': vitesse_to_allure(row['Vitesse_Semi2']),
                'Gain_Pct': f"{row['Gain_Vitesse_Pct']:.1f}"
            })
            
    # --- CHRONOLOGIE DES EXPLOSIONS ---
    df_comp = df_finishers[df_finishers['is_complete'] == True].copy()
    explosions = []
    if not df_comp.empty:
        v21 = df_comp['Vitesse_kmh_KM21'].replace(0, np.nan)
        v25 = df_comp['Vitesse_kmh_KM25'].replace(0, np.nan)
        v30 = df_comp['Vitesse_kmh_KM30'].replace(0, np.nan)
        v37 = df_comp['Vitesse_kmh_KM37'].replace(0, np.nan)
        v40 = df_comp['Vitesse_kmh_KM40'].replace(0, np.nan)
        
        drop_25 = ((v21 - v25) / v21) * 100
        drop_30 = ((v25 - v30) / v25) * 100
        drop_37 = ((v30 - v37) / v30) * 100
        drop_40 = ((v37 - v40) / v37) * 100
        
        explosions = [
            round((drop_25 >= 15).sum() / len(df_comp) * 100, 1),
            round((drop_30 >= 15).sum() / len(df_comp) * 100, 1),
            round((drop_37 >= 15).sum() / len(df_comp) * 100, 1),
            round((drop_40 >= 15).sum() / len(df_comp) * 100, 1)
        ]
    else:
        explosions = [0, 0, 0, 0]
    
    # --- GRAPHIQUES GLOBAUX ---
    bins_minutes = np.arange(120, 435, 15) 
    labels_tranches = [f"{int(b//60)}h{int(b%60):02d}" for b in bins_minutes[:-1]]
    df_finishers['Minutes_Totales'] = df_finishers['Passage_ARRIVEE_sec'] / 60
    df_finishers['Tranche'] = pd.cut(df_finishers['Minutes_Totales'], bins=bins_minutes, labels=labels_tranches, right=False)
    
    dist_sexe = df_finishers.dropna(subset=['Tranche']).groupby(['Tranche', 'Sexe'], observed=False).size().unstack(fill_value=0)
    hist_labels = dist_sexe.index.tolist()
    hist_hommes = dist_sexe['M'].tolist() if 'M' in dist_sexe else [0]*len(hist_labels)
    hist_femmes = dist_sexe['F'].tolist() if 'F' in dist_sexe else [0]*len(hist_labels)
    
    dist_cat = df_finishers.dropna(subset=['Tranche']).groupby(['Tranche', 'Macro_Cat'], observed=False).size().unstack(fill_value=0)
    macro_cats = ['Seniors & Jeunes (<35 ans)', 'Masters 0-2 (35-49 ans)', 'Masters 3-4 (50-59 ans)', 'Masters 5+ (60 ans et +)']
    dist_cat_data = {c: dist_cat[c].tolist() if c in dist_cat else [0]*len(hist_labels) for c in macro_cats}
    
    sexe_counts = df_finishers['Sexe'].value_counts()
    sexe_labels = sexe_counts.index.tolist()
    sexe_values = sexe_counts.tolist()
    
    cat_counts = df_finishers['Macro_Cat'].value_counts().reindex(macro_cats).fillna(0)
    cat_labels = cat_counts.index.tolist()
    cat_values = cat_counts.tolist()
    
    mur_bins = [-np.inf, -5, 5, 15, 30, np.inf]
    mur_labels_txt = ['< -5%', 'Neutre (-5% à +5%)', '+5% à +15%', '+15% à +30%', '> +30%']
    df_finishers['Mur_Tranche'] = pd.cut(df_finishers['Derive_Allure_vs_Precedent_%_KM37'], bins=mur_bins, labels=mur_labels_txt)
    mur_counts_serie = df_finishers['Mur_Tranche'].value_counts().reindex(mur_labels_txt).fillna(0)
    mur_labels = mur_counts_serie.index.tolist()
    mur_values = mur_counts_serie.tolist()

    return {
        'metrics': base_metrics,
        'deltas': deltas,
        'vitesse_moy_str': f"{base_metrics['vitesse_moy']:.2f}", 
        'allure_moy_str': vitesse_to_allure(base_metrics['vitesse_moy']),
        'vitesse_med_str': f"{base_metrics['vitesse_med']:.2f}", 
        'allure_med_str': vitesse_to_allure(base_metrics['vitesse_med']),
        'pct_neg_str': f"{base_metrics['pct_neg']:.1f}",
        'pct_murs_str': f"{base_metrics['pct_murs']:.1f}",
        'pct_murs_15_str': f"{base_metrics['pct_murs_15']:.1f}",
        'pct_femmes_str': f"{base_metrics['pct_femmes']:.1f}",
        'top_neg_splits': top_neg_splits,
        'explosions_values': explosions,
        'hist_labels': hist_labels, 
        'hist_hommes': hist_hommes, 'hist_femmes': hist_femmes,
        'dist_cat_data': dist_cat_data, 
        'sexe_labels': sexe_labels, 'sexe_values': sexe_values,
        'cat_labels': cat_labels, 'cat_values': cat_values,
        'mur_labels': mur_labels, 'mur_values': mur_values
    }

# --- CHARGEMENT ---
print("\n" + "="*50)
print("🚀 DÉMARRAGE DU DASHBOARD MARATHON")
print("="*50)
print(f"📊 Fichier 2025 : {'✅ Trouvé' if os.path.exists(DATA_2025) else '❌ Introuvable'} ({DATA_2025})")
print(f"📊 Fichier 2026 : {'✅ Trouvé' if os.path.exists(DATA_2026) else '❌ Introuvable'} ({DATA_2026})")
print("="*50 + "\n")

datasets = {
    "2025": prepare_dataframe(DATA_2025),
    "2026": prepare_dataframe(DATA_2026)
}

base_2025 = compute_base_metrics(datasets["2025"]) if not datasets["2025"].empty else None

dashboards = {
    "2025": compute_dashboard(datasets["2025"], "2025"),
    "2026": compute_dashboard(datasets["2026"], "2026", metrics_prev_year=base_2025)
}

@app.route('/')
def index():
    year = request.args.get('year', '2026')
    if year not in dashboards or dashboards[year] is None:
        fallback = '2025' if year == '2026' else '2026'
        if fallback in dashboards and dashboards[fallback] is not None:
            year = fallback
        else:
            return f"<h3>Erreur critique</h3><p>Aucun fichier de données (2025 ou 2026) n'a pu être chargé.</p>"
            
    v_2025 = str(base_2025['vainqueur_dos']) if base_2025 else ""
    return render_template('index.html', current_year=year, vainqueur_2025=v_2025, **dashboards[year])

@app.route('/api/gpx')
def get_gpx():
    return jsonify(gpx_data)

@app.route('/api/search')
def search():
    year = request.args.get('year', '2026')
    query = request.args.get('q', '').lower()
    df = datasets.get(year)
    
    if not query or df is None or df.empty: return jsonify([])
    mask_nom = df['Nom'].str.lower().str.contains(query, na=False)
    mask_dos = df['Dossard'].astype(str).str.startswith(query, na=False)
    
    results = df[mask_nom | mask_dos].head(10)[['Dossard', 'Nom', 'ARRIVEE', 'is_complete']]
    results['is_complete'] = results['is_complete'].astype(bool)
    return jsonify(results.to_dict(orient='records'))

@app.route('/api/replay/<dossard>')
def get_replay_data(dossard):
    force_year = request.args.get('force_year', None)
    year = force_year if force_year else request.args.get('year', '2026')
    
    df = datasets.get(year)
    if df is None or df.empty: return jsonify({"error": "Data unavailable"}), 404
    
    coureur = df[df['Dossard'].astype(str) == str(dossard)]
    if coureur.empty: return jsonify({"error": "Non trouvé"}), 404
    
    c = coureur.iloc[0]
    
    if not bool(c.get('is_complete', True)):
        return jsonify({"error": "Incomplete", "Nom": str(c['Nom']).split('-')[0].strip()}), 400
    
    timeline = [
        {"km": 0, "sec": 0},
        {"km": 4, "sec": float(c.get('Passage_KM4_sec', 0))},
        {"km": 10, "sec": float(c.get('Passage_KM10_sec', 0))},
        {"km": 15, "sec": float(c.get('Passage_KM15_sec', 0))},
        {"km": 21.1, "sec": float(c.get('Passage_KM21_sec', 0))},
        {"km": 25, "sec": float(c.get('Passage_KM25_sec', 0))},
        {"km": 30, "sec": float(c.get('Passage_KM30_sec', 0))},
        {"km": 37, "sec": float(c.get('Passage_KM37_sec', 0))},
        {"km": 40, "sec": float(c.get('Passage_KM40_sec', 0))},
        {"km": 42.195, "sec": float(c.get('Passage_ARRIVEE_sec', 0))}
    ]
    timeline = [pt for pt in timeline if pt['sec'] > 0 or pt['km'] == 0] 

    v_dep = c.get('Vitesse_kmh_KM4', 0)
    if v_dep == 0: v_dep = c.get('Vitesse_kmh_KM10', 0)

    vitesses = [
        float(v_dep) if float(v_dep) > 0 else None,
        float(c.get('Vitesse_kmh_KM4', 0)) if float(c.get('Vitesse_kmh_KM4', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_KM10', 0)) if float(c.get('Vitesse_kmh_KM10', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_KM15', 0)) if float(c.get('Vitesse_kmh_KM15', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_KM21', 0)) if float(c.get('Vitesse_kmh_KM21', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_KM25', 0)) if float(c.get('Vitesse_kmh_KM25', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_KM30', 0)) if float(c.get('Vitesse_kmh_KM30', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_KM37', 0)) if float(c.get('Vitesse_kmh_KM37', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_KM40', 0)) if float(c.get('Vitesse_kmh_KM40', 0)) > 0 else None,
        float(c.get('Vitesse_kmh_ARRIVEE', 0)) if float(c.get('Vitesse_kmh_ARRIVEE', 0)) > 0 else None,
    ]

    classements = [
        None, 
        safe_rank(c.get('Classement_KM4', 0)),
        safe_rank(c.get('Classement_KM10', 0)),
        safe_rank(c.get('Classement_KM15', 0)),
        safe_rank(c.get('Classement_KM21', 0)),
        safe_rank(c.get('Classement_KM25', 0)),
        safe_rank(c.get('Classement_KM30', 0)),
        safe_rank(c.get('Classement_KM37', 0)),
        safe_rank(c.get('Classement_KM40', 0)),
        safe_rank(c.get('Classement_ARRIVEE', 0))
    ]
    
    nom_propre = str(c['Nom']).split('-')[0].strip()
    display_name = f"{nom_propre} ({year})" if force_year else nom_propre

    return jsonify({
        "Dossard": str(c['Dossard']),
        "Nom": display_name,
        "Categorie": str(c.get('Cat_Brute', '')), 
        "Sexe": str(c.get('Sexe', '')),
        "Chrono": str(c['ARRIVEE']),
        "Derive": float(c.get('Derive_Allure_vs_Precedent_%_KM37', 0)),
        "Timeline": timeline,
        "Vitesses": vitesses,
        "Classements": classements,
        "Year": year
    })

@app.route('/api/mur_comparison')
def get_mur_comparison():
    if "2025" not in dashboards or dashboards["2025"] is None:
        return jsonify([])
    return jsonify({
        'mur_labels': dashboards["2025"]['mur_labels'],
        'mur_values': dashboards["2025"]['mur_values'],
        'explosions_values': dashboards["2025"]['explosions_values']
    })

if __name__ == '__main__':
    app.run(debug=True)