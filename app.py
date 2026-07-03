import streamlit as st
import pandas as pd
import os
from src.features.aggregation import SpatialAggregator
from src.visualization.plots import SpatialVisualizer, PlayerProfiler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Football-Ops | Analytics", page_icon="⚽", layout="wide")

# --- DATA LOADER ---
# The @st.cache_data decorator keeps the 120k+ row matrix in RAM 
# so the UI doesn't reload it every time a user clicks a button.
@st.cache_data
def load_data():
    file_path = "data_store/valued_actions.parquet"
    if not os.path.exists(file_path):
        return None
    return pd.read_parquet(file_path)

df = load_data()

if df is None:
    st.error("🚨 Data Lake Empty: Please run 'python run_scaleup.py' to generate 'valued_actions.parquet'.")
    st.stop()

# --- UI HEADER ---
st.title("⚽ Football-Ops: Spatiotemporal Analytics Engine")
st.markdown("Interactive Expected Threat (xT) and VAEP Scouting Dashboard built on StatsBomb Open Data.")
st.divider()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Scouting Filters")

# Extract unique teams dynamically
teams = sorted(df['team'].dropna().unique())
selected_team = st.sidebar.selectbox("Target Team", ["All Teams"] + list(teams))

# Filter dataset based on selection
if selected_team != "All Teams":
    df_filtered = df[df['team'] == selected_team]
else:
    df_filtered = df.copy()

min_actions = st.sidebar.slider("Minimum Actions to Qualify", min_value=10, max_value=500, value=150)

# --- SECTION 1: SQUAD RECRUITMENT MATRIX ---
st.header("📋 Tactical Value Leaderboard")
st.markdown("Rankings derived from mathematically isolated on-ball impact (VAEP).")

profiler = PlayerProfiler()
# Generate report using our backend class
scouting_report = profiler.generate_scouting_report(df_filtered, min_actions=min_actions)

# Render interactive dataframe
st.dataframe(
    scouting_report, 
    use_container_width=True,
    hide_index=True
)

st.divider()

# --- SECTION 2: SPATIAL THREAT DEEP DIVE ---
st.header("🗺️ Dynamic Player Spatial Threat Map")
st.markdown("Select a player from the filtered recruitment matrix to render their specific xT generation zones.")

# Get available players from the current filtered report
available_players = sorted(scouting_report['player'].unique()) if not scouting_report.empty else []

if available_players:
    selected_player = st.selectbox("Select Target Player", available_players)
    
    if st.button("Generate Spatial Heatmap"):
        with st.spinner(f"Aggregating 105x68 spatial threat matrix for {selected_player}..."):
            
            # Isolate the player's actions
            player_df = df_filtered[df_filtered['player'] == selected_player]
            
            # Run Backend Aggregation
            aggregator = SpatialAggregator(x_bins=12, y_bins=8)
            threat_matrix = aggregator.generate_threat_grid(player_df)
            
            # Run Backend Visualization
            visualizer = SpatialVisualizer()
            visualizer.plot_threat_heatmap(threat_matrix)
            
            # Display the generated PNG asset in the Streamlit UI
            st.image("outputs/spatial_xt_heatmap.png", caption=f"Total Offensive VAEP (xT) Created: {selected_player}")
else:
    st.info("⚠️ No players found matching the current minimum action thresholds.")