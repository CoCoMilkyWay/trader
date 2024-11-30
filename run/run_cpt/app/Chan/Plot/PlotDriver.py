import plotly.graph_objects as go
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict

from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import FX_TYPE, KLINE_DIR
from Chan.KLine.KLine_List import CKLine_List
from Chan.KLine.KLine_Unit import CKLine_Unit
from Chan.Common.ChanException import CChanException, ErrCode


class ChanPlotter:
    def __init__(self):
        self.fig = go.Figure()
        self.lv_lst = []  # Levels list if needed
        self.plot_para = CChanConfig().plot_para

    def draw_klu(self):
        """Draw K-line units"""
        print('Drawing KLU...')

        plot_mode = self.plot_para['klu']['plot_mode']
        rugd = False  # red up green down

        up_color = 'red' if rugd else 'green'
        down_color = 'green' if rugd else 'red'

        if plot_mode == "kl":
            # Separate data for up and down candlesticks for efficiency
            up_data = dict(open=[], high=[], low=[], close=[], x=[])
            down_data = dict(open=[], high=[], low=[], close=[], x=[])

            # Single pass through data to separate up/down candlesticks
            for kl in self.klc_list.klu_iter():
                if kl.close > kl.open:
                    data = up_data
                else:
                    data = down_data

                data['x'].append(kl.idx)
                data['open'].append(kl.open)
                data['high'].append(kl.high)
                data['low'].append(kl.low)
                data['close'].append(kl.close)

            # Add up candlesticks if we have any
            if up_data['x']:
                self.fig.add_trace(go.Candlestick(
                    x=up_data['x'],
                    open=up_data['open'],
                    high=up_data['high'],
                    low=up_data['low'],
                    close=up_data['close'],
                    increasing_line_color=up_color,
                    decreasing_line_color=up_color,
                    name='Up',
                    showlegend=False
                ))

            # Add down candlesticks if we have any
            if down_data['x']:
                self.fig.add_trace(go.Candlestick(
                    x=down_data['x'],
                    open=down_data['open'],
                    high=down_data['high'],
                    low=down_data['low'],
                    close=down_data['close'],
                    increasing_line_color=down_color,
                    decreasing_line_color=down_color,
                    name='Down',
                    showlegend=False
                ))

        else:
            # For line plots, collect data in a single pass
            x_data = []
            y_data = []

            for kl in self.klc_list.klu_iter():
                x_data.append(kl.idx)
                if plot_mode == "close":
                    y_data.append(kl.close)
                elif plot_mode == "high":
                    y_data.append(kl.high)
                elif plot_mode == "low":
                    y_data.append(kl.low)
                elif plot_mode == "open":
                    y_data.append(kl.open)
                else:
                    raise CChanException(
                        f"unknown plot mode={
                            plot_mode}, must be one of kl/close/open/high/low",
                        ErrCode.PLOT_ERR
                    )

            if x_data:
                self.fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=plot_mode,
                    showlegend=False
                ))

        # Update layout specific to candlestick/line charts
        self.fig.update_layout(
            xaxis_rangeslider_visible=False,  # Disable rangeslider for better performance
            dragmode='zoom',  # Enable zoom by default
        )

        return self.fig

    def draw_klc(self):
        """Draw K-Line Clusters efficiently by adding all shapes at once"""
        print('Drawing KLC...')
        color_type = {
            FX_TYPE.TOP: 'red',
            FX_TYPE.BOTTOM: 'blue',
            KLINE_DIR.UP: 'green',
            KLINE_DIR.DOWN: 'green'
        }
        width = self.plot_para['klc']['width']
        plot_single_kl = False

        # Create all shapes at once
        shapes = [
            {
                "type": "rect",
                "x0": klc.lst[0].idx - width,
                "y0": klc.low,
                "x1": klc.lst[-1].idx + width,
                "y1": klc.high,
                "line": {
                    "color": color_type[klc.fx if klc.fx != FX_TYPE.UNKNOWN else klc.dir],
                    "width": 1,
                },
                "fillcolor": "rgba(0,0,0,0)"
            }
            for klc in self.klc_list
            if not (klc.lst[0].idx == klc.lst[-1].idx and not plot_single_kl)
        ]

        # Add all shapes in a single update
        self.fig.update_layout(shapes=shapes)
        return self.fig

    def draw_bi(self):
        """Draw Bi-Directional Lines"""
        print('Drawing Bi...')
        color = 'black'
        show_num = self.plot_para['bi']['show_num']
        num_fontsize = 15
        num_color = "red"
        sub_lv_cnt = None
        facecolor = 'green'
        alpha = 0.1
        disp_end = self.plot_para['bi']['disp_end']
        end_color = 'black'
        end_fontsize = 10

        bi_list = self.klc_list.bi_list
        # Draw lines and annotations
        for bi_idx, bi in enumerate(bi_list):

            begin_x = bi.get_begin_klu().idx
            end_x = bi.get_end_klu().idx
            begin_y = bi.get_begin_val()
            end_y = bi.get_end_val()

            # Draw the line
            if bi.is_sure:
                self.fig.add_trace(go.Scatter(
                    x=[begin_x, end_x],
                    y=[begin_y, end_y],
                    mode='lines',
                    line=dict(color=color),
                    showlegend=False
                ))
            else:
                self.fig.add_trace(go.Scatter(
                    x=[begin_x, end_x],
                    y=[begin_y, end_y],
                    mode='lines',
                    line=dict(color=color, dash='dash'),
                    showlegend=False
                ))

            # Add number label if requested
            if show_num:
                self.fig.add_annotation(
                    x=(begin_x + end_x)/2,
                    y=(begin_y + end_y)/2,
                    text=str(bi.idx),
                    showarrow=False,
                    font=dict(size=num_fontsize, color=num_color)
                )

            # Add end label if requested
            if disp_end:
                self.fig.add_annotation(
                    x=end_x,
                    y=end_y,
                    text=str(bi_idx),
                    showarrow=False,
                    font=dict(size=end_fontsize, color=end_color)
                )

        # Add sub-level highlighting if needed
        if sub_lv_cnt is not None and sub_lv_cnt < len(bi_list):
            begin_idx = bi_list[-sub_lv_cnt].get_begin_klu().idx
            end_idx = bi_list[-1].get_end_klu().idx

            self.fig.add_vrect(
                x0=begin_idx,
                x1=end_idx,
                fillcolor=facecolor,
                opacity=alpha,
                line_width=0
            )

        # # Update layout
        # self.fig.update_layout(
        #     showlegend=False,
        #     hovermode='closest'
        # )

        return self.fig

    def plot(self,
             klc_list: CKLine_List,
             **kwargs):
        """Convenience method to draw both KLC and Bi elements"""
        self.klc_list = klc_list

        self.draw_klu()
        self.draw_klc()
        self.draw_bi()

        # Update layout
        self.fig.update_layout(
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            ),
            plot_bgcolor='white',
            width=1200,  # Set default width
            height=800,  # Set default height
            dragmode='pan',     # Enable box zoom by default
            modebar_add=[       # Add more tools to the mode bar
                'pan',
                'select',
                'lasso2d',
                'zoomIn',
                'zoomOut',
                'autoScale',
                'resetScale'
            ]
        )

        self.config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
            'displaylogo': False,
            'doubleClick': 'reset+autosize',
            'editable': True,
            'showTips': True
        }

        self.fig.show(config=self.config)
        print('Saving to HTML...')
        # self.fig.write_html('chan_plot.html',config=self.config)

        return self.fig
