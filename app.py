import streamlit as st
import requests
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="EcoScan — Ecosystem Health Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0e1a12; color: #d4e8d0; }
section[data-testid="stSidebar"] { background: #0a1209; border-right: 1px solid #1e3a24; }
.main-title { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #7ecb8f; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 0.2rem; }
.main-sub { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #4a7a56; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 2rem; }
.metric-card { background: #12201a; border: 1px solid #1e3a24; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem; }
.metric-label { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #4a7a56; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.metric-value { font-size: 1.6rem; font-weight: 500; color: #7ecb8f; }
.health-healthy { color: #5ecf7a; }
.health-moderate { color: #e8b94a; }
.health-degraded { color: #e85a4f; }
.tier-badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.05em; }
.badge-healthy  { background: #1a3d24; color: #5ecf7a; border: 1px solid #2a5a36; }
.badge-moderate { background: #3d2e10; color: #e8b94a; border: 1px solid #5a4318; }
.badge-degraded { background: #3d1a18; color: #e85a4f; border: 1px solid #5a2824; }
.cell-summary { background: #12201a; border: 1px solid #1e3a24; border-left: 3px solid #7ecb8f; border-radius: 0 10px 10px 0; padding: 1rem 1.2rem; font-size: 0.88rem; line-height: 1.65; color: #b8d4b4; margin-top: 1rem; }
.ai-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #4a7a56; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; }
.coord-box { background: #12201a; border: 1px solid #1e3a24; border-radius: 8px; padding: 0.5rem 0.75rem; font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #7ecb8f; margin-bottom: 0.5rem; }
.coord-lbl { font-size: 0.65rem; color: #4a7a56; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem; }
.stButton > button { background: #1e3d28 !important; color: #7ecb8f !important; border: 1px solid #2a5a36 !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; letter-spacing: 0.05em !important; padding: 0.5rem 1.2rem !important; }
.stButton > button:hover { background: #2a5a36 !important; border-color: #7ecb8f !important; }
.stSelectbox > div > div { background: #12201a !important; border: 1px solid #1e3a24 !important; color: #d4e8d0 !important; border-radius: 8px !important; }
.stTextInput > div > div > input { background: #12201a !important; border: 1px solid #1e3a24 !important; color: #d4e8d0 !important; border-radius: 8px !important; }
.stSlider > div > div > div > div { background: #7ecb8f !important; }
hr { border-color: #1e3a24; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GBIF_BASE = "https://api.gbif.org/v1"
GRID_SIZE = 0.02

INVASIVE_SPECIES = {
    'Rhododendron ponticum', 'Gaultheria shallon', 'Prunus serotina',
    'Fallopia japonica', 'Impatiens glandulifera', 'Heracleum mantegazzianum',
    'Quercus rubra', 'Crassula helmsii',
}
THREATENED_CODES = {'VU', 'EN', 'CR'}
WOODY_FAMILIES = {
    'Pinaceae', 'Fagaceae', 'Betulaceae', 'Salicaceae',
    'Ericaceae', 'Rosaceae', 'Oleaceae', 'Aquifoliaceae'
}
HEATHLAND_GENERA = {'Calluna', 'Erica', 'Vaccinium', 'Empetrum', 'Ulex'}

FEATURE_COLS = [
    'species_richness', 'genus_richness', 'family_richness',
    'shannon_diversity', 'simpson_diversity',
    'invasive_ratio', 'threatened_ratio',
    'woody_ratio', 'heathland_score',
    'temporal_span', 'classes'
]

# Verified bounding boxes — tightened to reduce sparse edge cells
PRESET_AREAS = {
    "Cannock Chase, UK":   {"lat": "52.68,52.82", "lon": "-2.03,-1.88"},
    "New Forest, UK":      {"lat": "50.84,51.00", "lon": "-1.72,-1.48"},
    "Peak District, UK":   {"lat": "53.20,53.40", "lon": "-1.88,-1.64"},
    "Dartmoor, UK":        {"lat": "50.54,50.70", "lon": "-4.00,-3.78"},
    "Sherwood Forest, UK": {"lat": "53.18,53.30", "lon": "-1.12,-0.98"},
    "Custom area":         None,
}

TIER_COLORS = {"Healthy": "#5ecf7a", "Moderate": "#e8b94a", "Degraded": "#e85a4f"}

# ── Helper functions ──────────────────────────────────────────────────────────
def shannon(counts):
    c = np.array(counts)
    c = c[c > 0]
    if len(c) == 0:
        return 0.0
    p = c / c.sum()
    return -np.sum(p * np.log(p + 1e-12))

def simpson(counts):
    c = np.array(counts)
    c = c[c > 0]
    if len(c) == 0:
        return 0.0
    p = c / c.sum()
    return 1 - np.sum(p ** 2)

@st.cache_data(show_spinner=False)
def fetch_occurrences(lat_range, lon_range, target=3000):
    records, offset = [], 0
    while len(records) < target:
        params = {
            'decimalLatitude': lat_range, 'decimalLongitude': lon_range,
            'hasCoordinate': 'true', 'hasGeospatialIssue': 'false',
            'occurrenceStatus': 'PRESENT', 'limit': 300, 'offset': offset,
        }
        try:
            resp = requests.get(f"{GBIF_BASE}/occurrence/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            batch = data.get('results', [])
            if not batch:
                break
            records.extend(batch)
            offset += 300
            if data.get('endOfRecords', True):
                break
        except Exception as e:
            st.error(f"GBIF API error: {e}")
            break
    return records

def parse_records(raw):
    FIELDS = ['gbifID','decimalLatitude','decimalLongitude','species','genus',
              'family','order','class','kingdom','year','month',
              'basisOfRecord','iucnRedListCategory']
    df = pd.DataFrame([{f: r.get(f) for f in FIELDS} for r in raw])
    df.dropna(subset=['decimalLatitude','decimalLongitude','species'], inplace=True)
    df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'])
    df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'])
    df.drop_duplicates(subset=['gbifID'], inplace=True)
    return df

def assign_grid(df):
    df = df.copy()
    df['lat_cell'] = (df['decimalLatitude'] // GRID_SIZE) * GRID_SIZE
    df['lon_cell'] = (df['decimalLongitude'] // GRID_SIZE) * GRID_SIZE
    df['cell_id'] = df['lat_cell'].round(4).astype(str) + '_' + df['lon_cell'].round(4).astype(str)
    return df

def compute_features(group):
    sp_counts = group['species'].value_counts()
    total = len(group)
    year_vals = group['year'].dropna()
    return pd.Series({
        'lat_cell': group['lat_cell'].iloc[0],
        'lon_cell': group['lon_cell'].iloc[0],
        'record_count': total,
        'species_richness': group['species'].nunique(),
        'genus_richness': group['genus'].nunique(),
        'family_richness': group['family'].nunique(),
        'shannon_diversity': shannon(sp_counts.values),
        'simpson_diversity': simpson(sp_counts.values),
        'invasive_ratio': group['species'].isin(INVASIVE_SPECIES).sum() / total,
        'threatened_ratio': group['iucnRedListCategory'].isin(THREATENED_CODES).sum() / total,
        'temporal_span': (year_vals.max() - year_vals.min()) if len(year_vals) > 1 else 0,
        'woody_ratio': group['family'].isin(WOODY_FAMILIES).sum() / total,
        'heathland_score': group['genus'].isin(HEATHLAND_GENERA).sum() / total,
        'kingdoms': group['kingdom'].nunique(),
        'classes': group['class'].nunique(),
    })

def run_model(features, min_records=5):
    feat = features[features['record_count'] >= min_records].copy()
    if len(feat) < 4:
        st.warning("Too few cells — try lowering the minimum records threshold.")
        return feat
    X_scaled = StandardScaler().fit_transform(feat[FEATURE_COLS].fillna(0))
    iso = IsolationForest(contamination=0.15, random_state=42)
    feat['health_proxy'] = iso.fit(X_scaled).score_samples(X_scaled)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    feat['cluster'] = km.fit_predict(X_scaled)
    cluster_means = feat.groupby('cluster')['health_proxy'].mean()
    label_map = {c: i for i, c in enumerate(cluster_means.sort_values().index)}
    feat['health_tier'] = feat['cluster'].map(label_map).map({0:'Degraded',1:'Moderate',2:'Healthy'})
    return feat

def get_ai_summary(row, api_key):
    prompt = f"""You are an ecologist analysing GBIF species occurrence data for a grid cell in a UK nature reserve.
Data: Health tier: {row['health_tier']}, Species richness: {int(row['species_richness'])}, Shannon H′: {row['shannon_diversity']:.2f}, Invasive ratio: {row['invasive_ratio']:.3f}, Threatened ratio: {row['threatened_ratio']:.3f}, Woody ratio: {row.get('woody_ratio',0):.3f}, Heathland score: {row.get('heathland_score',0):.3f}, Records: {int(row['record_count'])}.
Write 2-3 sentences interpreting this cell ecologically. Note what habitat type the scores suggest and any concerning signals. Be specific and scientific but readable."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30
        )
        return resp.json()['content'][0]['text']
    except Exception as e:
        return f"AI summary unavailable: {e}"

def build_map(feat):
    centre_lat = feat['lat_cell'].mean() + GRID_SIZE / 2
    centre_lon = feat['lon_cell'].mean() + GRID_SIZE / 2
    fmap = folium.Map(location=[centre_lat, centre_lon], zoom_start=12, tiles='CartoDB dark_matter')
    for _, row in feat.iterrows():
        color = TIER_COLORS.get(row['health_tier'], '#888888')
        popup_html = f"""<div style="font-family:monospace;font-size:12px;min-width:190px;background:#0e1a12;color:#d4e8d0;padding:10px;border-radius:6px;">
            <div style="font-size:13px;font-weight:bold;color:{color};margin-bottom:6px;">{row['health_tier']}</div>
            <div><b>Species richness:</b> {int(row['species_richness'])}</div>
            <div><b>Shannon H′:</b> {row['shannon_diversity']:.2f}</div>
            <div><b>Invasive ratio:</b> {row['invasive_ratio']:.3f}</div>
            <div><b>Records:</b> {int(row['record_count'])}</div>
            <div style="margin-top:5px;color:#4a7a56;font-size:10px;">{row['lat_cell']:.3f}°N {row['lon_cell']:.3f}°E</div>
        </div>"""
        folium.Rectangle(
            bounds=[[row['lat_cell'], row['lon_cell']],
                    [row['lat_cell'] + GRID_SIZE, row['lon_cell'] + GRID_SIZE]],
            color=color, fill=True, fill_color=color, fill_opacity=0.5, weight=1,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['health_tier']} — {int(row['species_richness'])} spp."
        ).add_to(fmap)
    return fmap

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [('results', None), ('ai_summaries', {}), ('last_preset', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">EcoScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Ecosystem Health Explorer</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Study area**")

    preset = st.selectbox("Preset locations", list(PRESET_AREAS.keys()))

    # Clear cached results when area changes
    if st.session_state.last_preset != preset:
        st.session_state.results = None
        st.session_state.ai_summaries = {}
        st.session_state.last_preset = preset

    if PRESET_AREAS[preset] is not None:
        lat_range = PRESET_AREAS[preset]["lat"]
        lon_range = PRESET_AREAS[preset]["lon"]
        st.markdown(f'<div class="coord-lbl">Latitude range</div><div class="coord-box">{lat_range}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="coord-lbl">Longitude range</div><div class="coord-box">{lon_range}</div>', unsafe_allow_html=True)
    else:
        lat_range = st.text_input("Latitude range", placeholder="e.g. 52.68,52.82")
        lon_range = st.text_input("Longitude range", placeholder="e.g. -2.03,-1.88")

    st.markdown("---")
    st.markdown("**Analysis settings**")
    target_records = st.slider("Max records to fetch", 500, 5000, 3000, step=500)
    min_records = st.slider("Min records per cell", 3, 20, 5)

    st.markdown("---")
    st.markdown("**AI summaries (optional)**")
    api_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
    st.caption("Adds AI-written ecological interpretation per cell.")

    st.markdown("---")
    run_btn = st.button("Run analysis", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
if not run_btn and st.session_state.results is None:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
        <div style="font-family:'DM Serif Display',serif;font-size:3.5rem;color:#7ecb8f;margin-bottom:1rem;">🌿</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#7ecb8f;margin-bottom:0.5rem;">Select an area and run the analysis</div>
        <div style="color:#4a7a56;font-size:0.9rem;max-width:420px;margin:0 auto;line-height:1.7;">
            Queries GBIF for species occurrence records, engineers ecological features,
            and classifies grid cells into health tiers using ML.
        </div>
    </div>""", unsafe_allow_html=True)

if run_btn:
    if not lat_range or not lon_range:
        st.error("Please enter a latitude and longitude range.")
    else:
        bar = st.progress(0, text="Fetching records from GBIF...")
        raw = fetch_occurrences(lat_range, lon_range, target=target_records)
        if not raw:
            st.error("No records returned. Check your bounding box.")
            bar.empty()
        else:
            bar.progress(40, text=f"Parsing {len(raw)} records...")
            df = parse_records(raw)
            df = assign_grid(df)
            bar.progress(65, text="Engineering ecological features...")
            features = df.groupby('cell_id').apply(compute_features).reset_index()
            bar.progress(80, text="Running ML model...")
            results = run_model(features, min_records=min_records)
            st.session_state.results = results
            st.session_state.ai_summaries = {}

            if api_key and len(results) > 0:
                for i, (_, row) in enumerate(results.iterrows()):
                    cid = row['cell_id']
                    if cid not in st.session_state.ai_summaries:
                        st.session_state.ai_summaries[cid] = get_ai_summary(row, api_key)
                    bar.progress(80 + int(20 * (i+1) / len(results)),
                                 text=f"AI summaries {i+1}/{len(results)}...")

            bar.progress(100, text="Complete.")
            bar.empty()

if st.session_state.results is not None:
    results = st.session_state.results
    tc = results['health_tier'].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Grid cells</div><div class="metric-value">{len(results)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Healthy</div><div class="metric-value health-healthy">{tc.get("Healthy",0)}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Moderate</div><div class="metric-value health-moderate">{tc.get("Moderate",0)}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Degraded</div><div class="metric-value health-degraded">{tc.get("Degraded",0)}</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Shannon H′</div><div class="metric-value">{results["shannon_diversity"].mean():.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    map_col, detail_col = st.columns([3, 2])

    with map_col:
        fmap = build_map(results)
        map_data = st_folium(fmap, width=None, height=480, returned_objects=["last_object_clicked"])

    with detail_col:
        st.markdown("**Cell detail**")
        st.caption("Click a cell on the map to inspect it")
        clicked = None
        if map_data and map_data.get("last_object_clicked"):
            click = map_data["last_object_clicked"]
            clat, clng = click.get("lat"), click.get("lng")
            if clat and clng:
                results['_dist'] = ((results['lat_cell'] + GRID_SIZE/2 - clat)**2 +
                                    (results['lon_cell'] + GRID_SIZE/2 - clng)**2)
                clicked = results.loc[results['_dist'].idxmin()]

        if clicked is not None:
            tier = clicked['health_tier']
            st.markdown(f'<span class="tier-badge badge-{tier.lower()}">{tier}</span>', unsafe_allow_html=True)
            st.markdown(f"**{clicked['lat_cell']:.3f}°N, {clicked['lon_cell']:.3f}°E**")
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("Species richness", int(clicked['species_richness']))
                st.metric("Shannon H′", f"{clicked['shannon_diversity']:.2f}")
                st.metric("Invasive ratio", f"{clicked['invasive_ratio']:.3f}")
            with mc2:
                st.metric("Records", int(clicked['record_count']))
                st.metric("Simpson D", f"{clicked['simpson_diversity']:.2f}")
                st.metric("Threatened ratio", f"{clicked['threatened_ratio']:.3f}")

            cid = clicked['cell_id']
            if cid in st.session_state.ai_summaries:
                st.markdown(f'<div class="cell-summary"><div class="ai-label">AI ecological interpretation</div>{st.session_state.ai_summaries[cid]}</div>', unsafe_allow_html=True)
            elif api_key:
                if st.button("Generate AI summary for this cell"):
                    with st.spinner("Generating..."):
                        summary = get_ai_summary(clicked, api_key)
                        st.session_state.ai_summaries[cid] = summary
                        st.rerun()
        else:
            st.markdown('<div style="color:#4a7a56;font-size:0.85rem;padding:2rem 0;line-height:1.7;">Click any coloured grid cell on the map to see its full breakdown.</div>', unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("Full results table"):
        display_cols = [c for c in ['cell_id','health_tier','record_count','species_richness',
            'shannon_diversity','simpson_diversity','invasive_ratio',
            'threatened_ratio','woody_ratio','heathland_score'] if c in results.columns]
        st.dataframe(results[display_cols].round(3), use_container_width=True, hide_index=True)

    st.download_button(
        "Download results CSV",
        data=results.round(4).to_csv(index=False),
        file_name="ecosystem_health_results.csv",
        mime="text/csv"
    )
