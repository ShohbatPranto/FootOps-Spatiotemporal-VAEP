import pandas as pd
import numpy as np
from src.config import PitchDimensions

class SPADLConverter:
    """Transforms raw vendor event data into the SPADL schema."""
    
    def __init__(self):
        self.dims = PitchDimensions()
        # Conversion ratios
        self.x_ratio = self.dims.SPADL_LENGTH / self.dims.SB_LENGTH
        self.y_ratio = self.dims.SPADL_WIDTH / self.dims.SB_WIDTH

    def _normalize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts StatsBomb 120x80 to SPADL 105x68.
        Note: StatsBomb Y=0 is top. SPADL Y=0 is bottom. We invert Y.
        """
        # 1. Extract starting coordinates safely
        start_x = []
        start_y = []
        
        # Ensure 'location' column exists
        if 'location' not in df.columns:
            df['location'] = np.nan
            
        for loc in df['location']:
            if isinstance(loc, list) and len(loc) >= 2:
                start_x.append(loc[0])
                start_y.append(loc[1])
            else:
                start_x.append(np.nan)
                start_y.append(np.nan)
                
        df['raw_start_x'] = start_x
        df['raw_start_y'] = start_y

        # Ensure all potential ending location columns exist
        for col in ['pass_end_location', 'carry_end_location', 'shot_end_location']:
            if col not in df.columns:
                df[col] = np.nan

        # 2. Coalesce ending coordinates, falling back to start coordinates if none found
        raw_end_x = []
        raw_end_y = []
        
        zip_cols = zip(
            df['raw_start_x'], df['raw_start_y'],
            df['pass_end_location'], df['carry_end_location'], df['shot_end_location']
        )
        
        for sx, sy, p_end, c_end, s_end in zip_cols:
            found = False
            for end_loc in [p_end, c_end, s_end]:
                if isinstance(end_loc, list) and len(end_loc) >= 2:
                    raw_end_x.append(end_loc[0])
                    raw_end_y.append(end_loc[1])
                    found = True
                    break
            if not found:
                raw_end_x.append(sx)
                raw_end_y.append(sy)

        df['raw_end_x'] = raw_end_x
        df['raw_end_y'] = raw_end_y

        # Normalize to 105x68 SPADL coords
        df['start_x'] = df['raw_start_x'] * self.x_ratio
        df['start_y'] = self.dims.SPADL_WIDTH - (df['raw_start_y'] * self.y_ratio)
        df['end_x'] = df['raw_end_x'] * self.x_ratio
        df['end_y'] = self.dims.SPADL_WIDTH - (df['raw_end_y'] * self.y_ratio)

        # Drop temporary raw extraction columns
        df = df.drop(columns=['raw_start_x', 'raw_start_y', 'raw_end_x', 'raw_end_y'])
        return df

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Executes the full pipeline to generate the SPADL matrix."""
        df = df_raw.copy()
        
        # 1. Normalize Coordinates
        df = self._normalize_coordinates(df)
        
        # 2. Extract standardized results (1 = Success, 0 = Fail)
        # In StatsBomb, a missing 'pass_outcome' means the pass was completed.
        df['result_success'] = np.where(
            (df['type'] == 'Pass') & (df['pass_outcome'].isna()), 1,
            np.where((df['type'] == 'Shot') & (df['shot_outcome'] == 'Goal'), 1, 
            np.where((df['type'] == 'Carry'), 1, 0))
        )
        
        # 3. Select strictly the SPADL core columns
        spadl_cols = [
            'match_id', 'id', 'period', 'timestamp', 'minute', 'second',
            'team', 'player', 'type', 'result_success', 
            'start_x', 'start_y', 'end_x', 'end_y'
        ]
        
        # Handle missing columns gracefully before returning
        available_cols = [col for col in spadl_cols if col in df.columns]
        return df[available_cols].sort_values(by=['period', 'timestamp']).reset_index(drop=True)