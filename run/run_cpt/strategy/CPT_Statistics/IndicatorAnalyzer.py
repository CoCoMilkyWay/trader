import json
import glob
import warnings
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from config.cfg_cpt import cfg_cpt

# this is only for visual, only remember to use consistent value
SCALE_PROFIT = 5
SCALE_HOLD = 5

@dataclass
class IndicatorLSStats: # longshort
    # longshort analysis
    profit_mean: float
    profit_std: float
    profit_median: float
    hold_mean: float
    hold_std: float
    hold_median: float
    profit_shape: str
    hold_shape: str
    indicator: str
    params: dict
    code: str
    year: int
    month: int
    
@dataclass
class IndicatorBiStats: # bi
    # bi analysis
    delta_mean: float
    delta_std: float
    delta_median: float
    period_mean: float
    period_std: float
    period_median: float
    delta_shape: str
    period_shape: str
    indicator: str
    lv: str
    code: str
    year: int
    month: int

@dataclass
class IndicatorVwmaStats: # vwma
    # vwma analysis
    max_dev_long_mean: float
    max_dev_long_std: float
    max_dev_long_median: float
    max_dev_short_mean: float
    max_dev_short_std: float
    max_dev_short_median: float
    period_long_mean: float
    period_long_std: float
    period_long_median: float
    period_short_mean: float
    period_short_std: float
    period_short_median: float
    max_dev_shape: str
    period_shape: str
    indicator: str
    params: dict
    code: str
    year: int
    month: int

class IndicatorAnalyzer:
    def __init__(self):
        # Color scheme for different indicators
        self.colors = {
            'chandelier': '#FF6B6B',    # red tone
            'chandekroll': '#EF476F',   # Pinkish-red tone
            'parabolic_sar': '#FFD166', # warm yellow
            'supertrend': '#118AB2',    # deeper blue
            'lorentzian': '#06D6A0',    # Green tone
        }
        
        # Acronyms for parameter names
        self.param_acronyms = {
            'length': 'L',
            'atr_period': 'AP',
            'mult': 'M',
            'atr_length': 'AL',
            'atr_coef': 'AC',
            'stop_len': 'SL',
            'acceleration': 'ACC',
            'max_acceleration': 'MAX',
            'initial_acceleration': 'IACC',
            'atr_len': 'AL',
            'factor': 'F',
            'kernel_lookback': 'KLB',
        }

    def get_param_text(self, params: dict) -> str:
        """Convert parameters to short form text"""
        return ','.join(f"{self.param_acronyms.get(k, k)}:{v}" for k, v in params.items())

    def load_data(self, data_dir: str) -> Dict:
        """Load and process all JSON files in the specified directory"""
        ind_fourier = []
        ind_bi = []
        ind_vwma = []
        ind_longshort = []
        json_files = glob.glob(f"{data_dir}/*.json")
        
        for file_path in json_files:
            split_names = Path(file_path).stem.split('_')
            year = int(split_names[0])
            month = int(split_names[1])
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for code, code_data in data.items():
                for indicator, indicator_data in code_data.items():
                    if indicator == 'fourier':
                        ind_fourier.append(indicator_data)
                        ind_fourier[-1]['year'] = year
                        ind_fourier[-1]['month'] = month
                    elif indicator == 'bi':
                        for bi_lv, results in indicator_data.items():
                            ind_bi.append(IndicatorBiStats(
                                delta_mean   = results['delta'][0],
                                delta_std    = results['delta'][1]/SCALE_PROFIT,
                                delta_median = results['delta'][4],
                                period_mean  = results['period'][0],
                                period_std   = results['period'][1]/SCALE_PROFIT,
                                period_median= results['period'][4],
                                delta_shape  = results['delta_shape'],
                                period_shape = results['period_shape'],
                                indicator    = indicator,
                                lv           = results['lv'],
                                code         = code,
                                year         = year,
                                month        = month,
                                ))
                    elif indicator == 'vwma':
                        for config_id, results in indicator_data.items():
                            ind_vwma.append(IndicatorVwmaStats(
                                max_dev_long_mean   = results['long_max_dev'][0],
                                max_dev_long_std    = results['long_max_dev'][1]/SCALE_PROFIT,
                                max_dev_long_median = results['long_max_dev'][4],
                                max_dev_short_mean  = results['short_max_dev'][0],
                                max_dev_short_std   = results['short_max_dev'][1]/SCALE_PROFIT,
                                max_dev_short_median= results['short_max_dev'][4],
                                period_long_mean    = results['long_period'][0],
                                period_long_std     = results['long_period'][1]/SCALE_PROFIT,
                                period_long_median  = results['long_period'][4],
                                period_short_mean   = results['short_period'][0],
                                period_short_std    = results['short_period'][1]/SCALE_PROFIT,
                                period_short_median = results['short_period'][4],
                                max_dev_shape       = results['max_dev_shape'],
                                period_shape        = results['period_shape'],
                                indicator           = indicator,
                                params              = results['params'],
                                code                = code,
                                year                = year,
                                month               = month,
                                ))
                    else:
                        if indicator == 'lorentzian':
                            # TODO
                            continue
                        for config_id, results in indicator_data.items():
                            ind_longshort.append(IndicatorLSStats(
                                profit_mean  = (results['long_profits'][0] + results['short_profits'][0]) * 0.5,
                                profit_std   = (results['long_profits'][1] + results['short_profits'][1]) * 0.5/SCALE_PROFIT,
                                profit_median= (results['long_profits'][4] + results['short_profits'][4]) * 0.5,
                                hold_mean    = (results['long_holds'][0] + results['short_holds'][0]) * 0.5 / 3600,
                                hold_std     = (results['long_holds'][1] + results['short_holds'][1]) * 0.5 / 3600/SCALE_HOLD,
                                hold_median  = (results['long_holds'][4] + results['short_holds'][4]) * 0.5 / 3600,
                                profit_shape = (results['profit_shape']),
                                hold_shape   = (results['hold_shape']),
                                indicator    = indicator,
                                params       = results['params'],
                                code         = code,
                                year         = year,
                                month        = month,
                                ))
        return {
            'fourier': ind_fourier,
            'bi': ind_bi,
            'vwma': ind_vwma,
            'longshort': ind_longshort,
        }

    def create_ellipse_points(self, center_x: float, center_y: float, 
                            width: float, height: float, n_points: int = 100) -> tuple:
        t = np.linspace(0, 2*np.pi, n_points)
        return center_x + width * np.cos(t), center_y + height * np.sin(t)

    def get_base_layout(self, n:int):
        """Common layout settings for all plots"""
        return {
            'hovermode': 'closest',
            'plot_bgcolor': 'white',
            'width': 1200,
            'height': 800*n,
            'margin': dict(l=50, r=50, t=50, b=50),
            'font': dict(size=12),
            'xaxis': dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True),
            'yaxis': dict(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True)
        }

    def adjust_color_for_date(self, base_color: str, year: int, month: int) -> str:
        """Adjust color based on date - earlier dates are lighter"""
        # Convert hex to RGB
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        # Calculate date factor (assume data range is roughly 2020-2024)
        # Earlier dates will have lower factors
        min_year = 2023
        max_year = 2024
        date_factor = (year - min_year) * 12 + month
        max_factor = (max_year - min_year) * 12 + 12  # Maximum possible date
        min = 0.8
        max = 1
        opacity = min + ((max-min) * date_factor / max_factor)
        
        # Adjust RGB values
        r = int(255 - (255 - r) * opacity)
        g = int(255 - (255 - g) * opacity)
        b = int(255 - (255 - b) * opacity)
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def plot_performance(self, output_path: str = ''):
        n = 4
        # masks = [cfg_cpt.analyze_longshort, cfg_cpt.analyze_bi, cfg_cpt.analyze_vwma, cfg_cpt.analyze_fourier]
        
        stats_list = self.load_data(cfg_cpt.stats_result)
        fig = make_subplots(rows=n, cols=1, subplot_titles=(
            "Take-Profit/Stop-Loss Method Analysis",
            "Chan-Bi Analysis - Period vs (%)delta",
            "VWMA Analysis - Period vs Max Deviation",
            "Fourier Transform Analysis - Period vs Amplitude",
            ),
            vertical_spacing=0.08,
        )
        
        # Create all three visualizations
        if cfg_cpt.analyze_longshort:
            fig = self.plot_longshort_performance(fig, stats_list['longshort'])
        if cfg_cpt.analyze_bi:
            fig = self.plot_bi_performance(fig, stats_list['bi'])
        if cfg_cpt.analyze_vwma:
            fig = self.plot_vwma_performance(fig, stats_list['vwma'])
        if cfg_cpt.analyze_fourier:
            fig = self.plot_fourier_performance(fig, stats_list['fourier'])
        
        # Create legend positions in a cleaner way
        legend_positions = {
            f'legend{i+1}': dict(
                y=0.9 - i*0.4,  # Evenly space legends vertically
                x=1.05,
                xanchor='left',
                yanchor=['top', 'middle', 'bottom'][i]
            ) for i in range(n) if i < 3
        }
        
        # Update layout with legend positions
        fig.update_layout(
            **self.get_base_layout(n), # type: ignore
            **legend_positions,
            showlegend=True
        )
        
        if output_path:
            fig.write_html(output_path)
        fig.show()

    def plot_longshort_performance(self, fig:go.Figure, stats_list: List[IndicatorLSStats]):
        print('Plotting Longshort Perf...')
        row = 1
        
        # Add grouped legend entries
        # First group: Point types (Mean/Median)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='diamond', size=15, color='gray'),
            name='Mean',
            showlegend=True,
            legend=f'legend{row}',
            legendgroup=f'point_types_{row}',
            legendgrouptitle_text="Point Types"
            ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='circle', size=15, color='gray'),
            name='Median',
            showlegend=True,
            legend=f'legend{row}',
            legendgroup=f'point_types_{row}'
            ), row=row, col=1)

        # Second group: Indicators
        for indicator, color in self.colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(color=color, size=15),
                name=indicator,
                showlegend=True,
                legend=f'legend{row}',
                legendgroup=f'indicators_{row}',
                legendgrouptitle_text="Indicators"
                ), row=row, col=1)

        # Plot ellipses and points
        for stat in stats_list:
            base_color = self.colors[stat.indicator]
            adjusted_color = self.adjust_color_for_date(base_color, stat.year, stat.month)
            
            # Add ellipse
            x, y = self.create_ellipse_points(
                stat.profit_mean, stat.hold_mean,
                stat.profit_std, stat.hold_std
            )
            fig.add_trace(go.Scatter(
                x=x, y=y,
                fill="toself",
                fillcolor=f'rgba{tuple(list(int(adjusted_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
                line=dict(color=adjusted_color, width=0.5),
                showlegend=False,
                hoverinfo="skip"
                ), row=row, col=1)

            # Add line connecting mean and median
            fig.add_trace(go.Scatter(
                x=[stat.profit_mean, stat.profit_median],
                y=[stat.hold_mean, stat.hold_median],
                mode='lines',
                line=dict(color=adjusted_color, width=0.5, dash='dot'),
                showlegend=False,
                hoverinfo="skip"
                ), row=row, col=1)

            # Add mean point
            fig.add_trace(go.Scatter(
                x=[stat.profit_mean],
                y=[stat.hold_mean],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=15,
                    color=adjusted_color,
                    opacity=0.7,
                    line=dict(color=adjusted_color, width=1)
                ),
                showlegend=False,
                hovertext=f"Mean<br>Params:{stat.params}<br>{stat.year}-{stat.month}<br>Profit:{stat.profit_mean:.2f}%<br>Hold:{stat.hold_mean:.2f}h%<br>Shapes:{stat.profit_shape}/{stat.hold_shape}",
                hoverinfo="text"
                ), row=row, col=1)

            # Add median point
            fig.add_trace(go.Scatter(
                x=[stat.profit_median],
                y=[stat.hold_median],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=15,
                    color=adjusted_color,
                    opacity=0.7,
                    line=dict(color=adjusted_color, width=1)
                ),
                showlegend=False,
                hovertext=f"Median<br>Params:{stat.params}<br>{stat.year}-{stat.month}<br>Profit:{stat.profit_median:.2f}%<br>Hold:{stat.hold_median:.2f}h",
                hoverinfo="text"
                ), row=row, col=1)
            
        # Update layout
        fig.update_xaxes(title_text="Profit (%)", row=row, col=1)
        fig.update_yaxes(title_text="Hold Time (hours)", row=row, col=1)
        
        fig.update_layout(
            xaxis1=dict(range=[-0.5, 0.5]),
            yaxis1=dict(range=[0, 5])
        )
        return fig

    def plot_bi_performance(self, fig:go.Figure, stats_list:List[IndicatorBiStats]):
        print('Plotting Bi Perf...')
        row = 2
        # Color scheme for different levels
        bi_colors = {
            'K_1M': '#FF6B6B',
            'K_5M': '#4ECDC4',
            'K_15M': '#45B7D1',
            'K_30M': '#96CEB4',
            'K_60M': '#FFCC5C',
            # 'K_2H': '#FF6F69',
            # 'K_4H': '#88D8B0',
        }
        
        # Add legend entries for point types
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='diamond', size=15, color='gray'),
            name='Mean',
            showlegend=True,
            legend=f'legend{row}',
            legendgroup=f'point_types_{row}',
            legendgrouptitle_text="Point Types"
            ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='circle', size=15, color='gray'),
            name='Median',
            showlegend=True,
            legend=f'legend{row}',
            legendgroup=f'point_types_{row}'
            ), row=row, col=1)
        
        # Plot data for each level
        for stat in stats_list:  # Assuming 'bi' contains level data
            base_color = bi_colors[stat.lv]
            
            # Extract data
            delta_mean   =stat.delta_mean   
            delta_std    =stat.delta_std    
            delta_median =stat.delta_median 
            period_mean  =stat.period_mean  
            period_std   =stat.period_std   
            period_median=stat.period_median
            delta_shape  =stat.delta_shape  
            period_shape =stat.period_shape 
            code         =stat.code 
            
            # Add ellipse
            x, y = self.create_ellipse_points(
                delta_mean, period_mean,  # mean values
                delta_std, period_std   # std values
            )
            fig.add_trace(go.Scatter(
                x=x, y=y,
                fill="toself",
                fillcolor=f'rgba{tuple(list(int(base_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
                line=dict(color=base_color, width=0.5),
                name=f"{stat.lv}",
                showlegend=False,
                # legend=f'legend{row}',
                # legendgroup=f'levels_{row}',
                # legendgrouptitle_text="Levels"
                ), row=row, col=1)
            
            # Add connection line
            fig.add_trace(go.Scatter(
                x=[delta_mean, delta_median],  # mean to median
                y=[period_mean, period_median],
                mode='lines',
                line=dict(color=base_color, width=0.5, dash='dot'),
                showlegend=False
                ), row=row, col=1)
            
            # Add mean point
            fig.add_trace(go.Scatter(
                x=[delta_mean],
                y=[period_mean],
                mode='markers',
                marker=dict(symbol='diamond', size=15, color=base_color),
                showlegend=False,
                hovertext=f"Mean<br>Code:{code}<br>Level:{stat.lv}<br>{stat.year}-{stat.month}<br>Delta:{delta_mean:.2f}%<br>Period:{period_mean:.2f}h<br>Shapes:{delta_shape}/{period_shape}",
                hoverinfo="text"
                ), row=row, col=1)
            
            # Add median point
            fig.add_trace(go.Scatter(
                x=[delta_std],
                y=[period_std],
                mode='markers',
                marker=dict(symbol='circle', size=15, color=base_color),
                showlegend=False,
                hovertext=f"Median<br>Code:{code}<br>Level:{stat.lv}<br>{stat.year}-{stat.month}<br>Delta:{delta_std:.2f}%<br>Period:{period_std:.2f}h",
                hoverinfo="text"
                ), row=row, col=1)
            
        fig.update_xaxes(title_text="Delta (%)", row=row, col=1)
        fig.update_yaxes(title_text="Period (hours)", row=row, col=1)
        
        return fig
    
    def plot_vwma_performance(self, fig:go.Figure, stats_list: List[IndicatorVwmaStats]):
        print('Plotting VWMA Perf...')
        row = 3
        def add_vwma_traces(fig, period_mean, dev_mean, period_std, dev_std,
                            period_median, dev_median, color, window_size, window_size_atr, position, year, month):
            # Add ellipse
            x, y = self.create_ellipse_points(dev_mean, period_mean, dev_std, period_std)
            fig.add_trace(go.Scatter(
                x=x, y=y,
                fill="toself",
                fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}',
                line=dict(color=color, width=0.5),
                name=f'{position}:{window_size}-{window_size_atr}({year}/{month})',
                showlegend=False,
                # legend=f'legend{row}',
                # legendgroup=f'window_size_{row}',
                # legendgrouptitle_text="window_size",
                # hoverinfo="skip"
                ), row=row, col=1)
            
            # Add connection line
            fig.add_trace(go.Scatter(
                x=[dev_mean, dev_median],
                y=[period_mean, period_median],
                mode='lines',
                line=dict(color=color, width=0.5, dash='dot'),
                showlegend=False
                ), row=row, col=1)

            # Add mean and median points
            fig.add_trace(go.Scatter(
                x=[dev_mean],
                y=[period_mean],
                mode='markers',
                marker=dict(symbol='diamond', size=15, color=color),
                showlegend=False,
                hovertext=f"Mean<br>Window:{window_size}<br>Window_Atr:{window_size_atr}<br>{stat.year}-{stat.month}<br>Position:{position}<br>Dev:{dev_mean:.2f}<br>Period:{period_mean:.2f}h",
                hoverinfo="text"
                ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=[dev_median],
                y=[period_median],
                mode='markers',
                marker=dict(symbol='circle', size=15, color=color),
                showlegend=False,
                hovertext=f"Median<br>Window:{window_size}<br>Window_Atr:{window_size_atr}<br>{stat.year}-{stat.month}<br>Position:{position}<br>Dev:{dev_median:.2f}<br>Period:{period_median:.2f}h",
                hoverinfo="text"
                ), row=row, col=1)
            return fig
        
        # Generate colors based on window sizes
        unique_windows = sorted(set(stat.params['window_size'] for stat in stats_list ))
        
        import colorsys
        n_colors = len(unique_windows)
        HSV_tuples = [(x*1.0/n_colors, 0.5, 0.9) for x in range(n_colors)]
        RGB_tuples = ['#%02x%02x%02x' % tuple(int(x*255) for x in colorsys.hsv_to_rgb(*hsv))
                      for hsv in HSV_tuples]
        
        color_scale = RGB_tuples
        window_colors = dict(zip(unique_windows, color_scale))
        
        for stat in stats_list:
            window_size = stat.params['window_size']
            window_size_atr = stat.params['window_size_atr']
            base_color = window_colors[window_size]
            
            # Plot long position data
            fig = add_vwma_traces(
                fig, 
                stat.period_long_mean, stat.max_dev_long_mean,
                stat.period_long_std, stat.max_dev_long_std,
                stat.period_long_median, stat.max_dev_long_median,
                base_color, window_size, window_size_atr, 'Long', stat.year, stat.month)
            
            # Plot short position data
            fig = add_vwma_traces(
                fig, 
                stat.period_short_mean, stat.max_dev_short_mean,
                stat.period_short_std, stat.max_dev_short_std,
                stat.period_short_median, stat.max_dev_short_median,
                base_color, window_size, window_size_atr, 'Short', stat.year, stat.month)
            
        fig.update_xaxes(title_text="Max Deviation", row=row, col=1)
        fig.update_yaxes(title_text="Period (hours)", row=row, col=1)
        return fig

    def plot_fourier_performance(self, fig: go.Figure, stats_list: List[Dict]):
        """Plot Fourier transform analysis results"""
        print('Plotting Fourier Perf...')
        row = 4  # Add as fourth plot
        
        # Add traces for each code's Fourier analysis
        for stat in stats_list:
            # Get Fourier data
            periods = np.array(stat['periods'])
            amplitude = np.array(stat['direct_dft_amplitude'])
            amplitude_smoothed = np.array(stat['direct_dft_amplitude_smoothed'])
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=amplitude,
                    mode='lines',
                    line=dict(width=1),
                    opacity=0.1,
                    showlegend=False,
                    # name=f"{stat['year']}_{stat['month']}",
                    # legend=f'legend{row}',
                    # legendgroup=f'codes_{row}',
                    # legendgrouptitle_text="Symbols",
                    hovertext=f"<br>{stat['year']}-{stat['month']}-dft",
                    hoverinfo="text",
                ), row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=amplitude_smoothed,
                    mode='lines',
                    line=dict(width=2),
                    opacity=1,
                    showlegend=False,
                    # name=f"{stat['year']}_{stat['month']}",
                    # legend=f'legend{row}',
                    # legendgroup=f'codes_{row}',
                    # legendgrouptitle_text="Symbols",
                    hovertext=f"<br>{stat['year']}-{stat['month']}-smoothed",
                    hoverinfo="text",
                ), row=row, col=1
            )
        
        # Update axes
        fig.update_xaxes(
            title_text="Period (hours)",
            # type='log',  # Logarithmic scale for better period visualization
            row=row, col=1
        )
        fig.update_yaxes(
            title_text="Amplitude",
            # type='log',  # Logarithmic scale for amplitude
            row=row, col=1
        )
        
        return fig

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

def analyze_distribution_shape(data):
    """
    Analyzes a list of floats by first trying to fit standard distributions,
    only considering gaussian mixture if no good fits are found.
    
    Parameters:
    data (list): List of float values
    
    Returns:
    str: Name of the best matching distribution using standard abbreviations
    """
    # Convert to numpy array and remove NaN values
    data = np.array(data)
    data = data[~np.isnan(data)]
    
    if len(data) < 8:
        return "unif"
        
    # Normalize data for consistent testing
    data_norm = (data - np.mean(data)) / np.std(data)
    
    # Dictionary to store test results
    fits = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 1. Test for standard distributions first
        # Normal distribution (using Shapiro-Wilk test)
        _, fits['normal'] = stats.shapiro(data_norm)
        
        # Uniform distribution
        _, fits['uniform'] = stats.kstest(data, 'uniform', 
                                        args=(data.min(), data.max() - data.min()))
        
        # Exponential (shift to positive)
        shifted_data = data - min(data) + 1e-10
        _, fits['exponential'] = stats.kstest(shifted_data, 'expon')
        
        # Log-normal
        if min(data) > 0:
            _, fits['lognormal'] = stats.kstest(np.log(data), 'norm')
        else:
            fits['lognormal'] = 0
            
        # Student's t
        _, fits['students_t'] = stats.kstest(data_norm, 't', args=(10,))
        
        # Beta (scale to [0,1])
        scaled_data = (data - min(data)) / (max(data) - min(data))
        _, fits['beta'] = stats.kstest(scaled_data, 'beta', args=(2, 2))
        
        # Gamma (shift to positive)
        _, fits['gamma'] = stats.kstest(shifted_data, 'gamma', args=(2,))

        # Weibull
        _, fits['weibull'] = stats.kstest(shifted_data, 'weibull_min', args=(2,))
        
        # Cauchy
        _, fits['cauchy'] = stats.kstest(data_norm, 'cauchy')
        
        # Get the best fit among standard distributions
        best_fit = max(fits.items(), key=lambda x: x[1])
        
        # If no good fit found (using 0.05 as threshold), check for gaussian mixture
        if best_fit[1] < 0.05:
            # Simple and robust bimodality test using kernel density
            kde = stats.gaussian_kde(data)
            x_grid = np.linspace(min(data), max(data), 100)
            density = kde(x_grid)
            
            # Count peaks with significant height
            peaks = []
            for i in range(1, len(density) - 1):
                if density[i-1] < density[i] > density[i+1]:
                    peaks.append(i)
            
            # If we find exactly two significant peaks, classify as gaussian mixture
            if len(peaks) == 2:
                return "gauss_mix"
        
        # Map distribution names to standard abbreviations
        distribution_mapping = {
            'normal': 'gaus',
            'uniform': 'unif',
            'exponential': 'exp',
            'lognormal': 'lognml',
            'gamma': 'gamma',
            'beta': 'beta',
            'cauchy': 'cauchy',
            'weibull': 'weibull',
            'students_t': 'students_t'
        }
        
        return distribution_mapping[best_fit[0]]

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
        long_profits = distribution_info(self.long_profits)
        short_profits = distribution_info(self.short_profits)
        long_holds = distribution_info(self.long_holds)
        short_holds = distribution_info(self.short_holds)
        profit_shape = analyze_distribution_shape(self.long_profits+self.short_profits)
        hold_shape = analyze_distribution_shape(self.long_holds+self.short_holds)
        if long_profits and short_profits:
            print(f"Avg Profit({params}):{(long_profits[0]+short_profits[0])*0.5:03.2f}({profit_shape}/{hold_shape})")
        return [self.long_switchs,self.short_switchs,long_profits,short_profits,long_holds,short_holds,profit_shape,hold_shape]
