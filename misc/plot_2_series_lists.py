import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_2_series_lists(series_list1, series_list2, names1=None, names2=None, 
                     title1="Plot 1", title2="Plot 2", main_title="Multiple Series Plot"):
    """
    Create and display two subplots, each containing multiple series.
    
    Args:
        series_list1: List of series for first subplot
        series_list2: List of series for second subplot
        names1: List of names for series in first subplot
        names2: List of names for series in second subplot
        title1: Title for first subplot
        title2: Title for second subplot
        main_title: Main title for entire figure
    """
    # Handle default series names
    if names1 is None:
        names1 = [f"Series 1.{i+1}" for i in range(len(series_list1))]
    if names2 is None:
        names2 = [f"Series 2.{i+1}" for i in range(len(series_list2))]
        
    # Create figure with shared x-axis
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(title1, title2),
        vertical_spacing=0.15,
        shared_xaxes=True
    )

    # Add traces for first subplot
    for series, name in zip(series_list1, names1):
        fig.add_trace(
            go.Scatter(x=list(range(len(series))), y=series, name=name),
            row=1, col=1
        )

    # Add traces for second subplot
    for series, name in zip(series_list2, names2):
        fig.add_trace(
            go.Scatter(x=list(range(len(series))), y=series, name=name),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        title_text=main_title,
        showlegend=True,
        margin=dict(t=50)
    )

    # Update axes labels
    fig.update_xaxes(title_text="Index", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    # Show the plot directly
    fig.show()

# Example usage:
if __name__ == "__main__":
    import numpy as np
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    
    # First subplot series
    sin_series = np.sin(x)
    sin_2x_series = np.sin(2*x)
    
    # Second subplot series
    cos_series = np.cos(x)
    cos_2x_series = np.cos(2*x)
    
    # Create and show plot
    plot_2_series_lists(
        series_list1=[sin_series, sin_2x_series],
        series_list2=[cos_series, cos_2x_series],
        names1=['sin(x)', 'sin(2x)'],
        names2=['cos(x)', 'cos(2x)'],
        title1="Sine Functions",
        title2="Cosine Functions",
        main_title="Trigonometric Functions Comparison"
    )