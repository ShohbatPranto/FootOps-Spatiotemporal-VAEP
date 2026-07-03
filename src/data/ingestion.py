import pandas as pd
from statsbombpy import sb
from typing import List
from src.config import ACTION_EVENTS  # Clean import at the top

class EventIngestor:
    """Handles the extraction of raw event data from providers."""
    
    @staticmethod
    def fetch_match_events(match_id: int) -> pd.DataFrame:
        """Fetches and performs initial filtering of raw StatsBomb events."""
        try:
            df_events = sb.events(match_id=match_id)
            action_mask = df_events['type'].isin(ACTION_EVENTS)
            return df_events[action_mask].copy()
        except Exception as e:
            raise RuntimeError(f"Failed to ingest match {match_id}: {str(e)}")
            
    @staticmethod
    def fetch_competition_matches(comp_id: int, season_id: int) -> List[int]:
        """Retrieves all match IDs for a specific competition and season."""
        df_matches = sb.matches(competition_id=comp_id, season_id=season_id)
        return df_matches['match_id'].tolist()