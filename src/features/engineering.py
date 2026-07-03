import pandas as pd
import numpy as np

class SpatialFeatureEngineer:
    """
    Engineers mathematical and temporal features from the base SPADL schema.
    """
    
    def __init__(self):
        # SPADL target goal coordinates (attacking right)
        self.goal_x = 105.0
        self.goal_y = 34.0

    def _calculate_distance_to_goal(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Calculates Euclidean distance to the center of the opponent's goal."""
        return np.sqrt((self.goal_x - x)**2 + (self.goal_y - y)**2)

    def _calculate_angle_to_goal(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """Calculates the absolute angle (in degrees) to the center of the goal."""
        dx = np.where((self.goal_x - x) == 0, 1e-6, self.goal_x - x)
        dy = self.goal_y - y
        angle_rad = np.arctan(np.abs(dy / dx))
        return angle_rad * (180.0 / np.pi)

    def _calculate_time_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the time elapsed since the previous action in the match."""
        # Force string timestamps into pandas timedelta objects
        if not pd.api.types.is_timedelta64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_timedelta(df['timestamp'])
            
        # Convert period and timestamp to absolute seconds in the match
        df['absolute_time'] = (df['period'] - 1) * 2700 + df['timestamp'].dt.total_seconds()
        
        # Calculate delta (grouped by match to avoid bleeding across different games)
        df['time_delta'] = df.groupby('match_id')['absolute_time'].diff().fillna(0.0)
        
        # Cap unreasonable time deltas (e.g., long injury breaks) to 60 seconds
        df['time_delta'] = df['time_delta'].clip(upper=60.0)
        
        return df

    def transform(self, df_spadl: pd.DataFrame) -> pd.DataFrame:
        """Executes the feature engineering pipeline."""
        df = df_spadl.copy()
        
        # 1. Spatial Features (Start Location)
        df['start_dist_to_goal'] = self._calculate_distance_to_goal(df['start_x'], df['start_y'])
        df['start_angle_to_goal'] = self._calculate_angle_to_goal(df['start_x'], df['start_y'])
        
        # 2. Spatial Features (End Location)
        df['end_dist_to_goal'] = self._calculate_distance_to_goal(df['end_x'], df['end_y'])
        df['end_angle_to_goal'] = self._calculate_angle_to_goal(df['end_x'], df['end_y'])
        
        # 3. Action context
        df['progression_dist'] = df['start_dist_to_goal'] - df['end_dist_to_goal']
        
        # 4. Temporal Features
        if 'timestamp' in df.columns:
             df = self._calculate_time_deltas(df)
        else:
             df['time_delta'] = 0.0 # Safe fallback
        
        return df