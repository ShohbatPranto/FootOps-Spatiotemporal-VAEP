import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os

class SpatialVisualizer:
    """Handles the rendering of high-end spatial pitch visualizations."""
    
    def __init__(self):
        # Configure a modern, dark-themed analytics pitch
        self.pitch = Pitch(
            pitch_type='custom', 
            pitch_length=105.0, 
            pitch_width=68.0, 
            pitch_color='#121212', 
            line_color='#4A4A4A',
            linewidth=1.5
        )
        # Ensure output directory exists
        os.makedirs('outputs', exist_ok=True)

    def plot_threat_heatmap(self, threat_matrix: pd.DataFrame, x_bins: int = 12, y_bins: int = 8):
        """Renders the 2D Expected Threat (xT) grid as a spatial heatmap."""
        fig, ax = self.pitch.draw(figsize=(12, 8))
        fig.set_facecolor('#121212')
        
        # Reconstruct the 2D NumPy array from our Pandas aggregation
        grid = np.zeros((y_bins, x_bins))
        for _, row in threat_matrix.iterrows():
            grid[int(row['zone_y']), int(row['zone_x'])] = row['total_threat_created']
            
        # Overlay the heatmap onto the SPADL pitch coordinates
        # origin='lower' ensures (0,0) is at the bottom left (SPADL standard)
        heatmap = ax.imshow(
            grid, 
            extent=(0, 105.0, 0, 68.0), 
            cmap='magma', 
            alpha=0.75, 
            origin='lower',
            aspect='auto'
        )
        
        # Styling and labels
        cbar = fig.colorbar(heatmap, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Total Offensive VAEP Created', color='white', size=12)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        
        plt.title("Spatial Expected Threat (xT) Generation Zones", color='white', fontsize=16, pad=20)
        
        output_path = 'outputs/spatial_xt_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#121212')
        print(f"📸 Saved Spatial Heatmap to: {output_path}")
        plt.close()


class PlayerProfiler:
    """Generates aggregate scouting metrics based on VAEP outputs."""
    
    @staticmethod
    def generate_scouting_report(df_valued: pd.DataFrame, min_actions: int = 30) -> pd.DataFrame:
        """
        Aggregates offensive and defensive VAEP to rank players.
        Filters out players with too few actions to prevent small-sample anomalies.
        """
        # Filter for intentional on-ball actions
        core_actions = ['Pass', 'Carry', 'Dribble', 'Shot', 'Clearance', 'Interception']
        mask = df_valued['type'].isin(core_actions)
        
        report = df_valued[mask].groupby('player').agg(
            total_actions=('id', 'count'),
            total_vaep=('vaep_value', 'sum'),
            offensive_vaep=('offensive_value', 'sum'),
            defensive_vaep=('defensive_value', 'sum')
        ).reset_index()
        
        # Filter by minimum actions and sort by highest total value
        report = report[report['total_actions'] >= min_actions]
        report = report.sort_values('total_vaep', ascending=False).reset_index(drop=True)
        
        # Round for clean reporting
        report[['total_vaep', 'offensive_vaep', 'defensive_vaep']] = report[['total_vaep', 'offensive_vaep', 'defensive_vaep']].round(4)
        
        return report