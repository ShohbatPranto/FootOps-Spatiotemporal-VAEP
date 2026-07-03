import pandas as pd
import numpy as np

class VAEPValuator:
    """
    Computes the offensive, defensive, and total VAEP value for every action
    by calculating the change in probabilities between game states.
    """
    
    @staticmethod
    def calculate_value(df: pd.DataFrame, p_scores: np.ndarray, p_concedes: np.ndarray) -> pd.DataFrame:
        """
        Injects probabilities into the dataframe and calculates delta values.
        """
        df = df.copy()
        
        # 1. Assign probabilities
        df['P_scores'] = p_scores
        df['P_concedes'] = p_concedes
        
        # 2. Shift probabilities to get the previous state (i-1)
        # Group by match and period to avoid shifting across halftime or different games
        df['prev_P_scores'] = df.groupby(['match_id', 'period'])['P_scores'].shift(1).fillna(0)
        df['prev_P_concedes'] = df.groupby(['match_id', 'period'])['P_concedes'].shift(1).fillna(0)
        df['prev_team'] = df.groupby(['match_id', 'period'])['team'].shift(1)
        
        # 3. Handle Possession Shifts
        # If the team changes, the previous team's scoring prob is our conceding prob
        same_team = df['team'] == df['prev_team']
        
        df['adjusted_prev_P_scores'] = np.where(same_team, df['prev_P_scores'], df['prev_P_concedes'])
        df['adjusted_prev_P_concedes'] = np.where(same_team, df['prev_P_concedes'], df['prev_P_scores'])
        
        # For the very first action of a half, previous probabilities are 0
        is_first_action = df['prev_team'].isna()
        df.loc[is_first_action, ['adjusted_prev_P_scores', 'adjusted_prev_P_concedes']] = 0.0
        
        # 4. Calculate Deltas (Value)
        df['offensive_value'] = df['P_scores'] - df['adjusted_prev_P_scores']
        df['defensive_value'] = -(df['P_concedes'] - df['adjusted_prev_P_concedes'])
        
        df['vaep_value'] = df['offensive_value'] + df['defensive_value']
        
        # Clean up temporary calculation columns
        cols_to_drop = ['prev_P_scores', 'prev_P_concedes', 'prev_team', 'adjusted_prev_P_scores', 'adjusted_prev_P_concedes']
        return df.drop(columns=cols_to_drop)