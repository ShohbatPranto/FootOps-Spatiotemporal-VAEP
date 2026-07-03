from src.models.valuation import VAEPValuator
from src.features.aggregation import SpatialAggregator
import pandas as pd
from src.data.ingestion import EventIngestor
from src.models.train import VAEPDatasetBuilder, VAEPModelTrainer
from src.visualization.plots import SpatialVisualizer, PlayerProfiler

def main():
    print("🏟️ Starting Spatiotemporal ML Training Pipeline...")
    
    # 1. Fetch World Cup 2018 Match IDs
    print("📡 Querying StatsBomb for World Cup 2018 Match IDs...")
    match_ids = EventIngestor.fetch_competition_matches(comp_id=43, season_id=3)
    
    if not match_ids:
        print("❌ Error: No matches found. Verify connection.")
        return
        
    # To keep execution time fast during initial runs, we will train on 15 matches.
    # You can easily increase this to train on all 64 World Cup matches.
    num_train_matches = min(15, len(match_ids))
    target_match_ids = match_ids[:num_train_matches]
    
    print(f"✅ Selected {num_train_matches} matches for temporal training and validation.")
    
    # 2. Compile and Engineer features
    builder = VAEPDatasetBuilder(lookahead_actions=10)
    df_features, y_scores, y_concedes = builder.build_dataset(target_match_ids)
    
    print(f"\n📊 Consolidated Dataset Details:")
    print(f"   Total Actions extracted: {df_features.shape[0]}")
    print(f"   Features Shape:          {df_features.shape}")
    print(f"   Goals Scored in dataset: {y_scores.sum()}")
    print(f"   Goals Conceded in dataset: {y_concedes.sum()}")
    
    # 3. Train Dual Classifiers
    trainer = VAEPModelTrainer()
    metrics = trainer.train(df_features, y_scores, y_concedes, test_ratio=0.2)
    
    # 4. Display Feature Importance
    print("\n🌲 Model A (P_scores) Top Feature Importances:")
    importance_df = pd.DataFrame({
        'Feature': trainer.feature_cols,
        'Importance': trainer.model_scores.feature_importance(importance_type='gain')
    }).sort_values(by='Importance', ascending=False)
    
    for rank, row in enumerate(importance_df.itertuples(), 1):
        print(f"   {rank}. {row.Feature:<25} : {row.Importance:.2f}")

    print("\n🎉 PHASE 3 REPLICABILITY COMPLETE: Models compiled successfully!")

    # --- PHASE 4: VALUATION & AGGREGATION ---
    print("\n💰 PHASE 4: Calculating VAEP Values for all actions...")
    
    # Generate probabilities for the entire dataset
    p_scores, p_concedes = trainer.predict_probabilities(df_features)
    
    # Calculate VAEP
    df_valued = VAEPValuator.calculate_value(df_features, p_scores, p_concedes)
    print(f"✅ Valuation Complete. Matrix Shape: {df_valued.shape}")
    
    # Display the top 5 most valuable non-shot actions in the dataset
    print("\n🌟 Top 5 Most Valuable Actions (Pass/Carry):")
    non_shots = df_valued[df_valued['type'] != 'Shot'].nlargest(5, 'vaep_value')
    cols_to_show = ['player', 'type', 'vaep_value', 'offensive_value', 'start_x', 'end_x']
    print(non_shots[cols_to_show].to_string(index=False))

    # Calculate 2D Spatial Threat Grid
    print("\n🗺️ Generating 2D Spatial Threat Matrix (12x8 Grid)...")
    aggregator = SpatialAggregator(x_bins=12, y_bins=8)
    threat_matrix = aggregator.generate_threat_grid(df_valued)
    print(f"✅ Grid Complete. Zones active: {len(threat_matrix)}/96")

    print("\n🎉 PHASE 4 COMPLETE: Pipeline is mathematically generating spatial currency!")

    # --- PHASE 5: VISUALIZATION & SCOUTING ---
    print("\n📈 PHASE 5: Generating Visualizations and Scouting Reports...")
    
    # 1. Render the Spatial Pitch Heatmap
    visualizer = SpatialVisualizer()
    visualizer.plot_threat_heatmap(threat_matrix)
    
    # 2. Generate the Top Player Recruitment Profiles
    profiler = PlayerProfiler()
    scouting_report = profiler.generate_scouting_report(df_valued, min_actions=50)
    
    print("\n📋 SCOUTING DEPARTMENT: Top 10 Most Valuable Players (VAEP)")
    print(scouting_report.head(10).to_string(index=False))
    
    print("\n🏆 PROJECT COMPLETE: Your production-grade Expected Threat & VAEP pipeline is fully operational!")

if __name__ == "__main__":
    main()