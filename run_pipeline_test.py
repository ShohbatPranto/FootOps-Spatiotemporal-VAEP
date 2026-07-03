import pandas as pd
import numpy as np
from src.data.ingestion import EventIngestor
from src.data.spadl import SPADLConverter
from src.features.engineering import SpatialFeatureEngineer

def main():
    print("🚀 Starting End-to-End Pipeline Verification Test...")
    
    # 1. Fetch a valid match ID dynamically from World Cup 2018 (Comp: 43, Season: 3)
    print("📥 Connecting to StatsBomb Open Data Server...")
    match_ids = EventIngestor.fetch_competition_matches(comp_id=43, season_id=3)
    
    if not match_ids:
        print("❌ Error: No matches retrieved. Verify your internet connection.")
        return
        
    test_match_id = match_ids[0]
    print(f"✅ Connection successful. Target Match Selected: ID {test_match_id}")
    
    # 2. Extract Raw Stream
    print("⚡ Extracting raw event streams...")
    df_raw = EventIngestor.fetch_match_events(test_match_id)
    print(f"✅ Extracted {len(df_raw)} raw action-based events.")
    
    # 3. Transform to SPADL Matrix
    print("📐 Converting to standardized 105x68 SPADL Schema matrix...")
    converter = SPADLConverter()
    df_spadl = converter.transform(df_raw)
    print(f"✅ SPADL Conversion Complete. Matrix Dimensions: {df_spadl.shape}")
    
    # 4. Run Vector Calculations
    print("🧮 Engineering spatial and geometric features...")
    engineer = SpatialFeatureEngineer()
    df_features = engineer.transform(df_spadl)
    print(f"✅ Spatial Feature Matrix Built. Matrix Dimensions: {df_features.shape}")
    
    # 5. Pipeline Integrity Summary
    print("\n📊 --- PIPELINE MATRIX AUDIT ---")
    sample_cols = [
        'type', 'result_success', 'start_x', 'start_y', 
        'start_dist_to_goal', 'start_angle_to_goal', 'progression_dist'
    ]
    print(df_features[sample_cols].head(10))
    
    print("\n🎉 ALL SYSTEMS OPERATIONAL: The pipeline data flow is structurally sound!")

if __name__ == "__main__":
    main()