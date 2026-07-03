import os
import pandas as pd
from pathlib import Path
from typing import List

from src.data.ingestion import EventIngestor
from src.data.spadl import SPADLConverter
from src.features.engineering import SpatialFeatureEngineer

class LocalFeatureStore:
    """
    Enterprise-grade Feature Cache manager.
    Handles local disk I/O to prevent network bottlenecks, 
    serializing ML-ready matrices into compressed Parquet partitions.
    """
    
    def __init__(self, base_dir: str = "data_store"):
        # Create the local data lake structure
        self.feature_dir = Path(base_dir) / "features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        
        self.converter = SPADLConverter()
        self.engineer = SpatialFeatureEngineer()

    def build_competition_features(self, comp_id: int, season_id: int) -> pd.DataFrame:
        """
        Compiles an entire competition season, processes all spatial features,
        and saves the matrix to a single partitioned Parquet file.
        """
        file_path = self.feature_dir / f"spadl_features_c{comp_id}_s{season_id}.parquet"
        
        # 1. Cache Hit: Load from disk instantly
        if file_path.exists():
            print(f"⚡ CACHE HIT: Loading compiled matrix from {file_path}")
            return pd.read_parquet(file_path)
            
        # 2. Cache Miss: Build the dataset
        print(f"🌊 CACHE MISS: Flooding Data Lake for Comp {comp_id} | Season {season_id}...")
        match_ids = EventIngestor.fetch_competition_matches(comp_id, season_id)
        
        all_features = []
        total_matches = len(match_ids)
        
        for idx, match_id in enumerate(match_ids, 1):
            try:
                # Extract, Transform, Engineer
                df_raw = EventIngestor.fetch_match_events(match_id)
                if len(df_raw) == 0: continue
                
                df_spadl = self.converter.transform(df_raw)
                df_features = self.engineer.transform(df_spadl)
                
                all_features.append(df_features)
                
                if idx % 10 == 0 or idx == total_matches:
                    print(f"   Processed {idx}/{total_matches} matches...")
                    
            except Exception as e:
                print(f"⚠️ Skipped match {match_id} due to extraction error: {e}")

        if not all_features:
            raise ValueError("No matches were successfully processed.")

        # 3. Concatenate and Serialize
        print("💾 Serializing 105x68 spatial matrix to Parquet format...")
        df_final = pd.concat(all_features, ignore_index=True)
        
        # Ensure all datatypes are strictly compatible with PyArrow
        df_final.to_parquet(file_path, index=False, engine='pyarrow')
        
        return df_final