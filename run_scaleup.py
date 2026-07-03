import pandas as pd
from src.data.feature_store import LocalFeatureStore
from src.models.train import VAEPDatasetBuilder, VAEPModelTrainer
from src.models.valuation import VAEPValuator
from src.features.aggregation import SpatialAggregator
from src.visualization.plots import SpatialVisualizer, PlayerProfiler
import warnings

# Suppress StatsBomb warnings
warnings.filterwarnings("ignore", message="credentials were not supplied")

def main():
    print("🏟️ Initiating Large-Scale Football-Ops Pipeline...")
    
    # 1. Data Lake & Feature Store (Comp 43, Season 3 = World Cup 2018)
    store = LocalFeatureStore()
    df_features = store.build_competition_features(comp_id=43, season_id=3)
    
    print(f"\n📊 Feature Store Loaded: {df_features.shape[0]} total actions.")
    
    # 2. Generate Target Labels
    builder = VAEPDatasetBuilder(lookahead_actions=10)
    y_scores, y_concedes = builder.generate_targets(df_features)
    
    print(f"   Goals Scored in dataset:   {y_scores.sum()}")
    print(f"   Goals Conceded in dataset: {y_concedes.sum()}")
    
    # 3. Train the Models
    trainer = VAEPModelTrainer()
    _ = trainer.train(df_features, y_scores, y_concedes, test_ratio=0.2)
    
    # 4. Action Valuation
    print("\n💰 Calculating VAEP Values for 100k+ actions...")
    p_scores, p_concedes = trainer.predict_probabilities(df_features)
    df_valued = VAEPValuator.calculate_value(df_features, p_scores, p_concedes)
    
    #Save the fully valued matrix to the Data Lake
    df_valued.to_parquet('data_store/valued_actions.parquet', index=False)
    
    # 5. Visualization & Aggregation
    print("\n📈 Generating Threat Matrices & Scouting Reports...")
    aggregator = SpatialAggregator()
    threat_matrix = aggregator.generate_threat_grid(df_valued)
    
    visualizer = SpatialVisualizer()
    visualizer.plot_threat_heatmap(threat_matrix)
    
    profiler = PlayerProfiler()
    # Increased minimum actions to 150 because we are using an entire tournament now
    scouting_report = profiler.generate_scouting_report(df_valued, min_actions=150)
    
    print("\n📋 WORLD CUP 2018: Top 10 Most Valuable Players (VAEP)")
    print(scouting_report.head(10).to_string(index=False))

if __name__ == "__main__":
    main()