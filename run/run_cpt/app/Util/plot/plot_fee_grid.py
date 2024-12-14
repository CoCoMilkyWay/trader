# Binance fee:
# https://www.binance.com/zh-CN/support/faq/%E5%B9%A3%E5%AE%89%E5%90%88%E7%B4%84%E8%B3%87%E9%87%91%E8%B2%BB%E7%8E%87%E7%B0%A1%E4%BB%8B-360033525031
# https://www.binance.com/zh-CN/futures/funding-history/perpetual/funding-fee-history

# 0.02%/0.05% futures Maker/Taker fee + 0.01% ~ 0.04% funding rate * multiplier(0.01%*n) * 4/8 hrs
# consider using BNB and USDC to minimize funding rate

# consider funding fee of 
# 0.05*2(open+close)+
# 0.02(avg) * (10x leverage + 1x funding)
# =0.3%
import plotly.graph_objects as go
def plot_fee_grid(fig:go.Figure, dtick:float):
    """Add x% interval grid lines to plotly figure using direct yaxis configuration"""
    
    print(f'Plotting Fee Grid...')
    fig.update_layout(
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            dtick=dtick,
            gridwidth=0.5,
            showticklabels=False,
            # nticks=10,  # Show approximately 10 tick labels
        )
    )
    return fig