import pandas as pd
import numpy as np

class SpatialAggregator:
    """
    Discretizes the pitch into a 2D grid to calculate aggregated spatial value.
    """
    
    def __init__(self, x_bins: int = 12, y_bins: int = 8):
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        
    def generate_threat_grid(self, df_valued: pd.DataFrame) -> pd.DataFrame:
        """
        Maps actions to a 2D spatial grid and sums the total offensive value 
        created in each zone (Expected Threat matrix).
        """
        df = df_valued.copy()
        
        # 1. Discretize spatial coordinates into bins
        x_edges = np.linspace(0, self.pitch_length, self.x_bins + 1)
        y_edges = np.linspace(0, self.pitch_width, self.y_bins + 1)
        
        df['zone_x'] = pd.cut(df['start_x'], bins=x_edges, labels=False, include_lowest=True)
        df['zone_y'] = pd.cut(df['start_y'], bins=y_edges, labels=False, include_lowest=True)
        
        # 2. Aggregate offensive value by zone
        # We focus primarily on successful passes and carries for spatial xT
        mask_progression = (df['type'].isin(['Pass', 'Carry'])) & (df['result_success'] == 1)
        
        threat_grid = df[mask_progression].groupby(['zone_x', 'zone_y'])['offensive_value'].sum().reset_index()
        threat_grid = threat_grid.rename(columns={'offensive_value': 'total_threat_created'})
        
        return threat_grid