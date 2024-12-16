import json
import glob
from typing import Dict, List
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config.cfg_cpt import cfg_cpt

# this is only for visual, only remember to use consistent value
SCALE_PROFIT = 5
SCALE_HOLD = 5

@dataclass
class IndicatorStats:
    profit_mean: float
    profit_std: float
    profit_median: float
    hold_mean: float  
    hold_std: float
    hold_median: float
    indicator: str
    params: dict
    year: int

class IndicatorAnalyzer:
    def __init__(self):
        # Color scheme for different indicators
        self.colors = {
            'chandelier': '#FF6B6B',   # red tone
            'parabolic_sar': '#FFD166', # warm yellow
            'supertrend': '#118AB2'    # deeper blue
        }
        
        # Acronyms for parameter names
        self.param_acronyms = {
            'length': 'L',
            'atr_period': 'AP',
            'mult': 'M',
            'acceleration': 'ACC',
            'max_acceleration': 'MAX',
            'initial_acceleration': 'IACC',
            'atr_len': 'AL',
            'factor': 'F',
            'lookback': 'LB'
        }

    def get_param_text(self, params: dict) -> str:
        """Convert parameters to short form text"""
        return ','.join(f"{self.param_acronyms.get(k, k)}:{v}" for k, v in params.items())

    def load_data(self, data_dir: str) -> List[IndicatorStats]:
        """Load and process all JSON files in the specified directory"""
        stats_list = []
        json_files = glob.glob(f"{data_dir}/*.json")
        
        for file_path in json_files:
            split_names = Path(file_path).stem.split('_')
            year = int(split_names[0])
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for exchange, exchange_data in data.items():
                for indicator, indicator_configs in exchange_data.items():
                    for config_id, results in indicator_configs.items():
                        stats_list.append(IndicatorStats(
                            profit_mean=(results['long_profits'][0] + results['short_profits'][0]) * 0.5,
                            profit_std=(results['long_profits'][1] + results['short_profits'][1]) * 0.5/SCALE_PROFIT,
                            profit_median=(results['long_profits'][4] + results['short_profits'][4]) * 0.5,
                            hold_mean=(results['long_holds'][0] + results['short_holds'][0]) * 0.5 / 3600,
                            hold_std=(results['long_holds'][1] + results['short_holds'][1]) * 0.5 / 3600/SCALE_HOLD,
                            hold_median=(results['long_holds'][4] + results['short_holds'][4]) * 0.5 / 3600,
                            indicator=indicator,
                            params=results['params'],
                            year=year
                        ))
        return stats_list

    def create_ellipse_points(self, center_x: float, center_y: float, 
                            width: float, height: float, n_points: int = 100) -> tuple:
        t = np.linspace(0, 2*np.pi, n_points)
        return center_x + width * np.cos(t), center_y + height * np.sin(t)

    def cluster_points(self, stats_list: List[IndicatorStats], eps=0.3, min_samples=2):
        """Cluster points based on profit and hold time medians, separately for each indicator"""
        # Get unique indicators
        indicators = set(stat.indicator for stat in stats_list)
        
        # Initialize labels array with -1 (noise)
        all_labels = np.array([-1] * len(stats_list))
        current_label = 0
        
        # Process each indicator separately
        for indicator in indicators:
            # Get indices and points for this indicator
            indicator_indices = [i for i, stat in enumerate(stats_list) if stat.indicator == indicator]
            if not indicator_indices:
                continue
                
            indicator_points = np.array([[stats_list[i].profit_median, stats_list[i].hold_median] 
                                       for i in indicator_indices])
            
            # Scale points for this indicator
            scaler = StandardScaler()
            points_scaled = scaler.fit_transform(indicator_points)
            
            # Perform clustering for this indicator
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_scaled)
            
            # Update labels, offsetting by current_label to keep them unique across indicators
            valid_labels = clustering.labels_[clustering.labels_ != -1]
            if len(valid_labels) > 0:
                clustering.labels_[clustering.labels_ != -1] += current_label
                current_label = max(clustering.labels_) + 1
            
            # Assign labels to main array
            all_labels[indicator_indices] = clustering.labels_
        
        return all_labels, None  # Return None for scaler as it's not used anymore
    
    def get_cluster_info(self, stats_list: List[IndicatorStats], cluster_labels: np.ndarray, 
                        cluster_id: int) -> str:
        """Get summary info for a cluster"""
        cluster_stats = [s for s, l in zip(stats_list, cluster_labels) if l == cluster_id]
        
        # Ensure all stats in cluster are from the same indicator
        indicators = set(stat.indicator for stat in cluster_stats)
        if len(indicators) > 1:
            print(f"Warning: Cluster {cluster_id} contains multiple indicators: {indicators}")
            return "Error: Mixed indicators"
        
        # Get unique parameter combinations
        param_ranges = {}
        for stat in cluster_stats:
            for param, value in stat.params.items():
                if param not in param_ranges:
                    param_ranges[param] = set()
                param_ranges[param].add(value)
        
        # Create summary text
        param_text = []
        for param, values in param_ranges.items():
            acronym = self.param_acronyms.get(param, param)
            if len(values) == 1:
                param_text.append(f"{acronym}:{next(iter(values))}")
            else:
                param_text.append(f"{acronym}:{min(values)}-{max(values)}")
        
        return f"n={len(cluster_stats)}\n" + "\n".join(param_text)

    def plot_performance(self, stats_list: List[IndicatorStats], output_path: str = ''):
        fig = go.Figure()

        # Cluster points
        cluster_labels, scaler = self.cluster_points(stats_list, eps=0.3, min_samples=2)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"Number of clusters: {n_clusters}")

        # Add grouped legend entries
        # First group: Point types (Mean/Median)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='diamond', size=15, color='gray'),
            name='Mean',
            showlegend=True,
            legendgroup='point_types',
            legendgrouptitle_text="Point Types"
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='circle', size=15, color='gray'),
            name='Median',
            showlegend=True,
            legendgroup='point_types'
        ))

        # Second group: Indicators
        for indicator, color in self.colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color=color, size=15),
                name=indicator,
                showlegend=True,
                legendgroup='indicators',
                legendgrouptitle_text="Indicators"
            ))

        # Plot ellipses and points
        for stat in stats_list:
            base_color = self.colors[stat.indicator]

            # Add ellipse
            x, y = self.create_ellipse_points(
                stat.profit_mean, stat.hold_mean,
                stat.profit_std, stat.hold_std
            )
            fig.add_trace(go.Scatter(
                x=x, y=y,
                fill="toself",
                fillcolor=f'rgba{tuple(list(int(base_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
                line=dict(color=base_color, width=0.5),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Add line connecting mean and median
            fig.add_trace(go.Scatter(
                x=[stat.profit_mean, stat.profit_median],
                y=[stat.hold_mean, stat.hold_median],
                mode='lines',
                line=dict(color=base_color, width=0.5, dash='dot'),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Add mean point
            fig.add_trace(go.Scatter(
                x=[stat.profit_mean],
                y=[stat.hold_mean],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=15,
                    color=base_color,
                    opacity=0.7,
                    line=dict(color=base_color, width=1)
                ),
                showlegend=False,
                hovertext=f"Mean<br>Params: {stat.params}<br>Profit: {stat.profit_mean:.2f}%<br>Hold: {stat.hold_mean:.2f}h",
                hoverinfo="text"
            ))

            # Add median point
            fig.add_trace(go.Scatter(
                x=[stat.profit_median],
                y=[stat.hold_median],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color=base_color,
                    opacity=0.7,
                    line=dict(color=base_color, width=1)
                ),
                showlegend=False,
                hovertext=f"Median<br>Params: {stat.params}<br>Profit: {stat.profit_median:.2f}%<br>Hold: {stat.hold_median:.2f}h",
                hoverinfo="text"
            ))

        # Add cluster labels
        for i, cluster_id in enumerate(range(n_clusters)):
            mask = cluster_labels == cluster_id
            if not any(mask):
                continue
            
            cluster_points = [(s.profit_median, s.hold_median) for s, m in zip(stats_list, mask) if m]
            cluster_center = np.mean(cluster_points, axis=0)
    
            indicator = next(s.indicator for s, m in zip(stats_list, mask) if m)
            base_color = self.colors[indicator]
    
            # Calculate offset direction based on cluster position
            # This creates a more distributed annotation layout
            angle = (i * 90) % 360  # Rotate around 4 quadrants
            offset_x = 40 * np.cos(np.radians(angle))
            offset_y = 40 * np.sin(np.radians(angle))
    
            cluster_info = self.get_cluster_info(stats_list, cluster_labels, cluster_id)
            fig.add_annotation(
                x=cluster_center[0],
                y=cluster_center[1],
                text=cluster_info,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=base_color,
                font=dict(size=12, color=base_color),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=base_color,
                borderwidth=1,
                borderpad=4,
                ax=offset_x,
                ay=offset_y,
                xref='x',
                yref='y',
                # Add annotation positioning properties
                xanchor='center',    # Center the annotation box
                yanchor='middle',    # Center the annotation box
                standoff=15         # Minimum distance between arrow and box
            )

        # Update layout
        fig.update_layout(
            title="Take-Profit/Stop-Loss Method Analysis",
            xaxis_title="Profit (%)",
            yaxis_title="Hold Time (hours)",
            hovermode='closest',
            dragmode='select',  # Enable selection/dragging
            modebar_add=['drawopenpath', 'eraseshape'],  # Add drawing tools
            plot_bgcolor='white',
            width=1200,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='LightGray',
                borderwidth=1,
                font=dict(size=12)  # Increased legend font size
            ),
            font=dict(size=12)  # Increased general font size
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True)

        if output_path:
            fig.write_html(output_path)
        fig.show()

# def main():
#     analyzer = IndicatorAnalyzer()
#     
#     # Load data from all JSON files in the directory
#     stats_list = analyzer.load_data("./stats")
#     
#     # Create visualization
#     analyzer.plot_performance(stats_list, "indicator_performance.html")
# 
# if __name__ == "__main__":
#     main()

class long_short_analyzer:
    def __init__(self):
        pass
    
    def init(self):
        self.long_switchs = 0
        self.short_switchs = 0
        self.long_holds = []
        self.short_holds = []
        self.long_profits = []
        self.short_profits = []
        self.entry_price = None
        self.entry_ts = None
    
    def update(self, long_switch:bool, short_switch:bool, bar_arrays:Dict[str, np.ndarray], i:int):
        long_pos = 1-cfg_cpt.FEE/2
        short_pos = 1+cfg_cpt.FEE/2
        
        switch = False
        if long_switch:
            switch = True
            self.long_switchs += 1
        elif short_switch:
            switch = True
            self.short_switchs += 1
        if switch:
            if self.entry_price and self.entry_ts:
                if long_switch:
                    self.short_profits.append((self.entry_price*long_pos - bar_arrays['close'][i]*short_pos)/self.entry_price*100)
                    self.short_holds.append(bar_arrays['timestamp'][i] - self.entry_ts)
                elif short_switch:
                    self.long_profits.append((bar_arrays['close'][i]*long_pos - self.entry_price*short_pos)/self.entry_price*100)
                    self.long_holds.append(bar_arrays['timestamp'][i] - self.entry_ts)
            self.entry_price = bar_arrays['close'][i]
            self.entry_ts = bar_arrays['timestamp'][i]
            
    def get_stats(self, params):
        long_profits = self.distribution_info(self.long_profits)
        short_profits = self.distribution_info(self.short_profits)
        long_holds = self.distribution_info(self.long_holds)
        short_holds = self.distribution_info(self.short_holds)
        if long_profits and short_profits:
            print(f"Avg long profit({params}):{long_profits[0]:03.2f}/{short_profits[0]:03.2f}")
        return [self.long_switchs,self.short_switchs,long_profits,short_profits,long_holds,short_holds]
        
    @staticmethod 
    def distribution_info(data:List[float]):
        if len(data) == 0:
            print('Error getting long short results...')
            return None
        # Calculate min, max, mean, median, and standard deviations
        min_val       = round(np.min(data)           , 2)
        max_val       = round(np.max(data)           , 2)
        mean_val      = round(np.mean(data)          , 2)
        median_val    = round(np.median(data)        , 2)
        stddev        = round(np.std(data)           , 2)
        percentile_5  = round(np.percentile(data, 5) , 2)
        percentile_95 = round(np.percentile(data, 95), 2)
        return [mean_val,stddev,min_val,percentile_5,median_val,percentile_95,max_val,]