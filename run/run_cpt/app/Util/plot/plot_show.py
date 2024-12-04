import plotly.graph_objects as go
def plot_show(fig:go.Figure):
    """Show or Save Plotly fig"""

    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
        'displaylogo': False,
        'doubleClick': 'reset+autosize',
        'editable': True,
        'showTips': True
    }
    print('Showing plot...')
    fig.show(config=config)
    print('Saving to HTML...')
    fig.write_html('chan_plot.html',config=config)