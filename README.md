# EcoScan — Ecosystem Biodiversity Explorer

A Streamlit web app that queries the GBIF API for species occurrence records, 
engineers ecological features per grid cell, and classifies spatial biodiversity 
patterns using unsupervised machine learning. Optionally generates AI-written 
ecological interpretations per cell via the Claude API.

Built as part of a research project investigating whether species occurrence 
patterns alone can characterise ecosystem structure and biodiversity gradients 
across a defined study area.

## Features

- Query any area worldwide using a bounding box or preset UK study areas
- Computes species richness, Shannon H′, Simpson diversity, invasive species 
  ratio, threatened species ratio, woody plant ratio, and heathland indicator score
- Unsupervised ML pipeline — Isolation Forest anomaly detection + K-Means clustering
- Interactive Folium map with per-cell popup details
- Optional AI-generated ecological interpretation per cell (Claude API)
- Export results as CSV

## Preset study areas

- Cannock Chase AONB, UK
- New Forest, UK  
- Peak District, UK
- Dartmoor, UK
- Sherwood Forest, UK
- Custom bounding box (any region worldwide)

## Quick start
```bash
git clone https://github.com/YOUR_USERNAME/ecoscan
cd ecoscan
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)

Push to GitHub, then deploy on [Streamlit Cloud](https://share.streamlit.io) 
for a free public URL — no server required.

## AI summaries

Add an [Anthropic API key](https://console.anthropic.com) in the sidebar to 
generate plain-English ecological interpretation for each grid cell.

## Data source

Species occurrence data from the [Global Biodiversity Information Facility (GBIF)](https://www.gbif.org). 
No API key required for read-only queries.

## Caveats

Health tier classifications are relative, unsupervised clusters based on 
occurrence patterns only — not absolute assessments of ecological condition. 
GBIF data contains observation bias toward roads and well-visited sites. 
See report for full methodological discussion.

## Tech stack

- Python, Streamlit, Folium, scikit-learn, pandas
- GBIF Occurrence API
- Anthropic Claude API (optional)
```

---

**GitHub About box** (the short description in the right sidebar of your repo — Settings → About):
```
Biodiversity pattern analysis from GBIF occurrence data — ML clustering + AI ecological interpretation. Built with Streamlit.