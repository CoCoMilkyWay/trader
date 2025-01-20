import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Dict, Optional, Union

from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KL_TYPE, FX_TYPE, KLINE_DIR
from Chan.KLine.KLine_List import CKLine_List
from Chan.Common.ChanException import CChanException, ErrCode

from config.cfg_cpt import cfg_cpt

class ChanPlotter:
    def __init__(self):
        self.fig = go.Figure()
        self.lv_lst = []  # Levels list if needed
        config = CChanConfig()
        self.plot_para = config.plot_para

        self.lv_list = config.lv_list
        self.lv_type_list = [lv[0] for lv in self.lv_list]
        self.color_list = [lv[3] for lv in self.lv_list]
        self.opacity_list = [lv[4] for lv in self.lv_list]

        # self.traces1:Dict[KL_TYPE,List[go.Scatter]] = {}
        # self.layout_annotations:Dict[int,List[Dict]] = {}
        # for lv in self.lv_type_list:
        #     self.traces1[lv] = []
        #     self.layout_annotations[lv] = []

        self.traces0: List[go.Scatter] = []
        self.traces1: List[go.Candlestick | go.Scatter | go.Bar] = []
        self.traces2: List[go.Bar] = []
        self.shapes: List[Dict] = []
        self.annotations: List[Dict] = []

    def draw_klu(self):
        """Draw K-line units"""
        print(f'Drawing KLU({self.lv})...')

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

                data['x'].append(kl.time.ts)
                data['open'].append(kl.open)
                data['high'].append(kl.high)
                data['low'].append(kl.low)
                data['close'].append(kl.close)

            # Add up candlesticks if we have any
            if up_data['x']:
                self.traces1.append(go.Candlestick(
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
                self.traces1.append(go.Candlestick(
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
                x_data.append(kl.time.ts)
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
                self.traces1.append(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines',
                    name=plot_mode,
                    showlegend=False
                ))
        return self.fig

    def draw_klc(self):
        """Draw K-Line Clusters efficiently by adding all shapes at once"""
        print(f'Drawing KLC({self.lv})...')
        color_type = {
            FX_TYPE.TOP: 'red',
            FX_TYPE.BOTTOM: 'blue',
            KLINE_DIR.UP: 'green',
            KLINE_DIR.DOWN: 'green'
        }
        width = self.plot_para['klc']['width']
        plot_single_kl = False

        # Create all shapes at once
        self.shapes.extend([
            {
                "type": "rect",
                "x0": klc.lst[0].time.ts,
                "y0": klc.low,
                "x1": klc.lst[-1].time.ts,
                "y1": klc.high,
                "line": {
                    "color": color_type[klc.fx if klc.fx != FX_TYPE.UNKNOWN else klc.dir],
                    "width": 1,
                },
                "fillcolor": "rgba(0,0,0,0)"
            }
            for klc in self.klc_list
            if not (klc.lst[0].time.ts == klc.lst[-1].time.ts and not plot_single_kl)
        ])

        return self.fig

    def draw_bi(self):
        """Draw Bi-Directional Lines"""
        
        bi_list = self.klc_list.bi_list
        print(f'Drawing Bi({self.lv}: {len(bi_list)})...')
        # color = 'black'
        show_num = self.plot_para['bi']['show_num']
        num_fontsize = 15
        num_color = "red"
        sub_lv_cnt = None
        facecolor = 'green'
        alpha = 0.1
        disp_end = self.plot_para['bi']['disp_end']
        end_color = 'black'
        end_fontsize = 10

        # Draw lines and annotations
        for bi_idx, bi in enumerate(bi_list):

            begin_x = bi.get_begin_klu().time.ts
            end_x = bi.get_end_klu().time.ts
            begin_y = bi.get_begin_val()
            end_y = bi.get_end_val()

            # Draw the line
            if bi.is_sure:
                self.traces1.append(go.Scatter(
                    x=[begin_x, end_x],
                    y=[begin_y, end_y],
                    mode='lines',
                    opacity=self.opacity,
                    line=dict(color=self.color),
                    showlegend=False
                ))
            else:
                self.traces1.append(go.Scatter(
                    x=[begin_x, end_x],
                    y=[begin_y, end_y],
                    mode='lines',
                    opacity=self.opacity,
                    line=dict(color=self.color, dash='dot'),
                    showlegend=False
                ))

            # Add number label if requested
            if show_num:
                self.annotations.append({
                    'x': (begin_x + end_x)/2,
                    'y': (begin_y + end_y)/2,
                    'text': str(bi.idx),
                    'showarrow': False,
                    'font': dict(size=num_fontsize, color=num_color),
                })

            # Add end label if requested
            if disp_end:
                self.annotations.append({
                    'x': end_x,
                    'y': end_y,
                    'text': str(bi_idx),
                    'showarrow': False,
                    'font': dict(size=end_fontsize, color=end_color),
                })

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

    def draw_charts(self, text: bool):
        "Draw Chart Patterns"
        _, shapes = self.klc_list.PA_Core.get_static_shapes(
            complete=True, potential=False)
        if not shapes:
            return self.fig
        print(f'Drawing {len(shapes['conv_type'])} Charts({self.lv})...')

        for shape in shapes['conv_type']:
            # note that level 1m would not be plotted because lv_idx_rev = 0
            self.traces1.extend([
                go.Scatter(x=[v.ts for v in shape.vertices], y=[v.value for v in shape.vertices],
                           mode='lines', line=dict(color=self.color, width=4),
                           opacity=0.2, showlegend=False),
                go.Scatter(x=shape.top_ts, y=shape.top_y,
                           mode='lines', line=dict(color='red', width=2*self.lv_idx_rev),
                           showlegend=False),
                go.Scatter(x=shape.bot_ts, y=shape.bot_y,
                           mode='lines', line=dict(color='blue', width=2*self.lv_idx_rev),
                           showlegend=False)
            ])
            if text:
                arrow_dir = shape.entry_dir  # 1 for up
                self.annotations.append({
                    'x': shape.vertices[-2].ts,
                    'y': shape.vertices[-2].value,
                    'text': shape.name,
                    'showarrow': True,
                    'arrowsize': 1,
                    'arrowwidth': 1,
                    'arrowcolor': self.color,
                    'arrowhead': 2,
                    'ax': 0,
                    'ay': arrow_dir * 100,
                    'font': {'color': self.color},
                    'yanchor': 'bottom',
                    'xanchor': 'center',
                    'opacity': 0.6
                })

        return self.fig

    def draw_liquidity_zones(self):
        """Draw liquidity zones"""
        print(f'Drawing Liquidity Zones({self.lv})...')

        # Get barrier zones and prepare collections
        try:
            liquidity_class = self.klc_list.PA_Core.PA_Liquidity
        except:
            return
        barrier_zones = liquidity_class.barrier_zones[0] + \
            liquidity_class.barrier_zones[1]

        # Helper function for shape and annotation creation
        def add_zone(x0, x1, bottom, top, is_supply):
            color = "rgb(255, 0, 0)" if is_supply else "rgb(0, 255, 0)"
            self.shapes.append({
                "type": "rect", "xref": "x", "yref": "y",
                "x0": x0, "x1": x1, "y0": bottom, "y1": top,
                "fillcolor": f"rgba{color[3:-1]}, 0.3)", "line_width": 0
            })

        # x=[ver.ts for ver in liquidity_class.vertices]
        # y=[ver.value for ver in liquidity_class.vertices]
        # self.traces1.extend([
        #     go.Scatter(x=x, y=y,
        #                mode='lines', line=dict(color='purple', width=10),
        #                opacity=0.2, showlegend=False),
        # ])

        for zone in barrier_zones:
            default_end = 1 << 31
            if zone.right0 == default_end:
                zone.right0 = self.current_ts
            if zone.right1 == default_end:
                zone.right1 = self.current_ts

            # 0: demand
            # 1: supply
            # 2: demand (1st break supply)
            # 3: supply (1st break demand)
            is_supply = zone.type in [1, 2]

            # Add main zone
            add_zone(zone.left, zone.right0, zone.bottom, zone.top, is_supply)
            # Add strength annotation
            self.annotations.append({
                "x": zone.left,
                "y": zone.top if is_supply else zone.bottom,
                "text": f"{zone.num_touch}{'*' * zone.strength_rating}",
                "showarrow": False, "font": {"size": 10},
                "xanchor": "left",
                "yanchor": "middle",
            })

            # Add order block line
            if zone.OB:
                line_y = zone.top if is_supply else zone.bottom
                self.shapes.append({
                    "type": "line", "xref": "x", "yref": "y",
                    "x0": zone.left, "x1": zone.right0,
                    "y0": line_y, "y1": line_y,
                    "line": {
                        "color": "black" if zone.ChoCh else ("red" if is_supply else "green"),
                        "width": 1
                    }
                })

            # Add break zone if applicable
            if zone.type in [2, 3]:
                add_zone(zone.right0 + 1, zone.right1,
                         zone.bottom, zone.top, zone.type == 3)
                if zone.BB:
                    line_y = zone.top if zone.type == 3 else zone.bottom
                    self.shapes.append({
                        "type": "line", "xref": "x", "yref": "y",
                        "x0": zone.right0 + 1, "x1": zone.right1,
                        "y0": line_y, "y1": line_y,
                        "line": {
                            "color": "black" if zone.ChoCh else ("orange" if zone.type == 3 else "blue"),
                            "width": 1
                        }
                    })

            # Add Break of Structure indicators
            if zone.BoS:
                if zone.ChoCh:
                    self.shapes.append({
                        "type": "line", "xref": "x", "yref": "y",
                        "x0": zone.BoS[0], "x1": zone.BoS[1],
                        "y0": zone.BoS[2], "y1": zone.BoS[2],
                        "line": {"color": "red", "width": 2, "dash": "dot"}
                    })
                    self.annotations.append({
                        "x": zone.BoS[0], "y": zone.BoS[2],
                        "text": "ChoCh", "showarrow": False,
                        "font": {"size": 8, "color": "red"}, "yanchor": "bottom"
                    })
                else:
                    self.annotations.append({
                        "x": zone.BoS[0], "y": zone.BoS[2],
                        "text": "BoS", "showarrow": False,
                        "font": {"size": 8, "color": "black"}, "yanchor": "bottom"
                    })

        return self.fig

    def draw_volume_profile(self):
        """Draw Volume Profiles"""
        try:
            PA = self.klc_list.PA_Core
            liquidity_class = self.klc_list.PA_Core.PA_Liquidity
            volume_profile_class = self.klc_list.PA_Core.PA_Volume_Profile
        except:
            return
        print(f'Drawing Volume Profile({self.lv})...')

        # Initialize values
        idx_min, idx_max = volume_profile_class.volume_idx_min, volume_profile_class.volume_idx_max
        price_bin_width = volume_profile_class.price_bin_width
        # y_pos = np.round(np.arange(idx_min, idx_max + 1) * price_bin_width, 2)
        y_pos = np.arange(idx_min, idx_max + 1) * price_bin_width
        x_begin = PA.ts_1st
        x_end = PA.ts_cur
        x_extend = (x_end - x_begin) * 0.05

        # Get profiles for different timeframes
        # [0] buyside,
        # [1] sellside,
        # [2] total,
        # [3] buyside_curve,
        # [4] sellside_curve,
        # [5] volume_weighted_cost,
        # [6] percentile_30,
        # [7] percentile_50,
        # [8] percentile_70,
        profiles = {
            'history': volume_profile_class.get_adjusted_volume_profile(max_mapped=x_extend, type='history', sigma=1.5)
        }

        # Calculate session volume
        total_session = [0] * (idx_max + 1 - idx_min)
        session_length = len(total_session)
        for zone in liquidity_class.barrier_zones[1]:
            if zone.type in [0, 1]:
                # pmv = price_mapped_volume
                for pmv in [zone.enter_bi_VP, zone.leaving_bi_VP]:
                    if pmv and len(pmv[0]):
                        idx_his = next(i for i, p in enumerate(
                            y_pos) if p == pmv[0][0])
                        for idx_ses, volume_ses in enumerate(pmv[1]):
                            total_session[idx_his + idx_ses] += int(volume_ses)
                # thd = zone.bottom if zone.type == 0 else zone.top
                # print(zone.index, thd, zone.enter_bi_VP, zone.leaving_bi_VP)

        max_vol = max(total_session) if total_session else 1
        total_session_adjusted = np.array(
            [v/max_vol*x_extend for v in total_session])

        # Define profile configurations
        configs = [
            # (profile_type, x_offset, buyside_color, sellside_color, label)
            ('history', 2, 'rgba(255,0,0,0.4)', 'rgba(0,255,0,0.4)', 'history VP')
        ]

        # Add volume bars and curves for each profile type
        for ptype, offset, buy_color, sell_color, label in configs:
            profile = profiles[ptype]
            base_x = x_end + offset * x_extend

            self.traces1.extend([
                # Add bars
                go.Bar(x=profile[0], y=y_pos, orientation='h', marker_color=buy_color.replace('0.4', '1'),
                       base=base_x, width=price_bin_width, showlegend=False),
                go.Bar(x=profile[1], y=y_pos, orientation='h', marker_color=sell_color.replace('0.4', '1'),
                       base=[base_x + b for b in profile[0]], width=price_bin_width, showlegend=False),
                # Add curves
                go.Scatter(x=[base_x + x for x in profile[2]], y=y_pos,
                           mode='lines', line=dict(color=buy_color.replace('0.4', '1')), showlegend=False),
                go.Scatter(x=[base_x + x for x in (profile[2] + profile[3])], y=y_pos,
                           mode='lines', line=dict(color=sell_color.replace('0.4', '1')), showlegend=False)
            ])

            # Add lines for volume weighted cost and percentiles
            for val, (width, color) in zip(profile[4:8], [(4, 'black'), (2, 'gray'), (2, 'gray'), (2, 'gray')]):
                self.traces1.append(go.Scatter(x=[base_x, base_x + x_extend], y=[val, val],
                                              mode='lines', line=dict(width=width, color=color), showlegend=False))

            # Add label
            self.annotations.append(dict(x=base_x, y=y_pos[-1] - (y_pos[-1] - y_pos[0])*0.05,
                                         text=label, showarrow=False, font=dict(size=10, color='black')))

        # Add session volume profile
        self.traces1.extend([
            go.Bar(x=total_session_adjusted, y=y_pos, width=price_bin_width, orientation='h',
                   marker_color='rgba(0,0,0,1)', base=x_end + x_extend, showlegend=False),
            go.Scatter(x=[x_end + x_extend + x for x in total_session_adjusted], y=y_pos,
                       mode='lines', line=dict(color='black', width=2), showlegend=False)
        ])

        # Add session label
        self.annotations.append(dict(x=x_end + x_extend, y=y_pos[-1] - (y_pos[-1] - y_pos[0])*0.05,
                                     text='liquidity VP', showarrow=False, font=dict(size=10, color='black')))

        return self.fig

    def draw_markers(self):
        print(f'Drawing bsp and markers({len(self.markers)})...')
        for mark in self.markers:
            if 'long' in mark[2]:
                arrow_dis = 50
            elif 'short' in mark[2]:
                arrow_dis = -50
            else:
                arrow_dis = 0
            self.annotations.append({
                'x': mark[0],
                'y': mark[1],
                'text': mark[2],
                'showarrow': True,
                'arrowsize': 1,
                'arrowwidth': 2,
                'arrowcolor': mark[3],
                'arrowhead': 2,
                'ax': 0,
                'ay': arrow_dis,
                'font': {'color': mark[3], 'size': 14},
                'yanchor': 'middle',
                'xanchor': 'center',
                'opacity': 1
            })

    def draw_volume(self):
        """Draw volume bars"""
        print(f'Drawing Volume...')

        # Separate volume data for up and down days
        up_volume = dict(x=[], y=[])
        down_volume = dict(x=[], y=[])

        for kl in self.klc_list.klu_iter():
            if kl.close >= kl.open:  # Up or equal day
                up_volume['x'].append(kl.time.ts)
                up_volume['y'].append(kl.volume)
            else:  # Down day
                down_volume['x'].append(kl.time.ts)
                down_volume['y'].append(kl.volume)

        # Add up volume bars
        if up_volume['x']:
            self.traces2.append(
                go.Bar(x=up_volume['x'],
                       y=up_volume['y'],
                       marker_color='green',
                       name='Up Volume',
                       showlegend=False)
            )

        # Add down volume bars
        if down_volume['x']:
            self.traces2.append(
                go.Bar(x=down_volume['x'],
                       y=down_volume['y'],
                       marker_color='red',
                       name='Down Volume',
                       showlegend=False)
            )

    def draw_ind(self):
        "Draw Indicators: "
        
        from Math.Chandelier_Stop import ChandelierIndicator
        from Math.ChandeKroll_Stop import ChandeKrollStop
        from Math.Parabolic_SAR_Stop import ParabolicSARIndicator
        from Math.VolumeWeightedBands import VolumeWeightedBands
        from Math.Adaptive_SuperTrend import AdaptiveSuperTrend
        
        # plot_chandelier = False
        # plot_chandekroll = False
        # plot_parabolic_sar = False
        # plot_vwma_bands = True
        # plot_atr_bands = True
        plot_bi_shapes = False
        plot_bsp = False
        plot_adaptive_supertrend = True
        plot_labels = True
        
        if cfg_cpt.dump_ind:
            # ind_c:ChandelierIndicator = self.indicators[n]
            # ind_k:ChandeKrollStop = self.indicators[n]
            # ind_ps:ParabolicSARIndicator = self.indicators[n]
            # ind_vb:VolumeWeightedBands = self.indicators[n]
            ind_st:AdaptiveSuperTrend = self.indicators[-4]
            timestamps:list[float] = self.indicators[-3]
            closes:list[float] = self.indicators[-2]
            labels:list[float] = self.indicators[-1]

            # if plot_chandelier:
            #     # print(f'    Chandelier Stop({ind_c.long_idx} switches)...')
            #     # for i in range(ind_c.long_idx):
            #     #     self.traces1.extend([
            #     #         go.Scatter(x=ind_c.his_longts[i], y=ind_c.his_longcs[i],
            #     #                 mode='lines',
            #     #                 line=dict(color='red', width=2), # dash='dot'
            #     #                 opacity=0.6, showlegend=False),
            #     #     ])
            #     # for i in range(ind_c.short_idx):
            #     #     self.traces1.extend([
            #     #         go.Scatter(x=ind_c.his_shortts[i], y=ind_c.his_shortcs[i],
            #     #                 mode='lines',
            #     #                 line=dict(color='blue', width=2), # dash='dot'
            #     #                 opacity=0.6, showlegend=False),
            #     #     ])
            #     print(f'    Chandelier Stop({ind_c.switch_idx} switches)...')
            #     self.traces1.extend([
            #         go.Scatter(x=ind_c.his_ts, y=ind_c.his_longcs,
            #                 mode='lines',
            #                 line=dict(color='blue', width=1), # dash='dot'
            #                 opacity=0.6, showlegend=False),
            #         go.Scatter(x=ind_c.his_ts, y=ind_c.his_shortcs,
            #                 mode='lines',
            #                 line=dict(color='red', width=1), # dash='dot'
            #                 opacity=0.6, showlegend=False),
            #         go.Scatter(x=ind_c.his_switch_ts, y=ind_c.his_switch_vs,
            #                 mode='markers',
            #                 marker=dict(color='black', size=6), # dash='dot'
            #                 opacity=1, showlegend=False),
            #     ])
            # 
            # if plot_chandekroll:
            #     print(f'    ChandeKroll Stop...')
            #     self.traces1.extend([
            #         go.Scatter(x=ind_k.his_ts, y=ind_k.his_upper,
            #                 mode='lines',
            #                 line=dict(color='brown', width=1), # dash='dot'
            #                 opacity=0.6, showlegend=False),
            #         go.Scatter(x=ind_k.his_ts, y=ind_k.his_lower,
            #                 mode='lines',
            #                 line=dict(color='brown', width=1), # dash='dot'
            #                 opacity=0.6, showlegend=False),
            #     ])
            # 
            # if plot_parabolic_sar:
            #     print(f'    Parabolic SAR({len(ind_ps.his_ep)} extreme points) Stop...')
            #     self.traces1.extend([
            #         go.Scatter(x=ind_ps.his_ts, y=ind_ps.his_sar,
            #                    mode='lines',
            #                    line=dict(color='black', width=2, dash='dot'), # dash='dot'
            #                    opacity=0.6, showlegend=False),
            #     ])
            # 
            # if plot_vwma_bands:
            #     print(f'    Volume weighted Bands...')
            #     self.traces1.extend([
            #         go.Scatter(x=ind_vb.his_ts, y=ind_vb.his_vavg,
            #                    mode='lines',
            #                    line=dict(color='black', width=2), # dash='dot
            #                    opacity=1, showlegend=False),
            #         # go.Scatter(x=ind_vb.his_ts, y=ind_vb.his_tavg,
            #         #            mode='lines',
            #         #            line=dict(color='blue', width=2), # dash='dot
            #         #            opacity=1, showlegend=False),
            #         go.Scatter(x=ind_vb.his_ts, y=ind_vb.his_b1up,
            #                    mode='lines',
            #                    line=dict(color='gray', width=1), # dash='dot
            #                    opacity=0.3,
            #                    fill=None,
            #                    showlegend=False),
            #         go.Scatter(x=ind_vb.his_ts, y=ind_vb.his_b1lo,
            #                    mode='lines',
            #                    line=dict(color='gray', width=1), # dash='dot
            #                    opacity=0.3,
            #                    fill='tonexty',  # Fill to the trace before this one
            #                    fillcolor='rgba(128, 128, 128, 0.2)',  # Light gray with transparency
            #                    showlegend=False),
            #     ])
            
            # if plot_atr_bands:
            #     print(f'    ATR Bands...')
            #     
            #     ts      = np.array(ind_ts)
            #     atr     = np.array(ind_value1)
            #     signal  = np.array(ind_value2)
            #     
            #     self.traces1.extend([
            #         go.Scatter(x=ts, y=signal,
            #                    mode='lines',
            #                    line=dict(color='black', width=2), # dash='dot
            #                    opacity=1, showlegend=False),
            #         go.Scatter(x=ts, y=signal+atr,
            #                    mode='lines',
            #                    line=dict(color='gray', width=1), # dash='dot
            #                    opacity=0.3,
            #                    fill=None,
            #                    showlegend=False),
            #         go.Scatter(x=ts, y=signal-atr,
            #                    mode='lines',
            #                    line=dict(color='gray', width=1), # dash='dot
            #                    opacity=0.3,
            #                    fill='tonexty',  # Fill to the trace before this one
            #                    fillcolor='rgba(128, 128, 128, 0.2)',  # Light gray with transparency
            #                    showlegend=False),
            #     ])
            
            # if plot_bi_shapes:
            #     print(f'    Shapes...')
            #     for i, txt in enumerate(ind_text):
            #         if 'v' in txt:
            #             color = 'green'
            #         elif '^' in txt:
            #             color = 'red'
            #         else:
            #             color = 'black'
            #         self.traces1.extend([
            #             go.Scatter(x=ind_ts[i], y=ind_value[i],
            #                        mode='lines',
            #                        line=dict(color=color, width=2), # dash='dot'
            #                        opacity=1, showlegend=False),
            #         ])
            #         self.annotations.append({
            #             'x': ind_ts[i][-1],
            #             'y': ind_value[i][-1],
            #             'text': ind_text[i],
            #             'showarrow': True,
            #             'arrowsize': 1,
            #             'arrowwidth': 1,
            #             'arrowcolor': color,
            #             'arrowhead': 2,
            #             'ax': 0,
            #             'ay': 30,
            #             'font': {'color': 'black', 'size': 10},
            #             'yanchor': 'middle',
            #             'xanchor': 'center',
            #             'opacity': 1
            #         })
            
            # if plot_bsp:
            #     for i, txt in enumerate(ind_text):
            #         if i%10==0:
            #             
            #             if txt[0]:
            #                 color = 'yellow'
            #             else:
            #                 color = 'orange'
            #                 
            #             self.annotations.append({
            #                 'x': ind_ts[i],
            #                 'y': ind_value[i],
            #                 'text': txt[1],
            #                 'showarrow': False,
            #                 'font': {'color': color, 'size': 15 if '|' not in txt[1] else 55},
            #                 'yanchor': 'middle',
            #                 'xanchor': 'center',
            #                 'opacity': 1
            #             })
            
            if plot_adaptive_supertrend:
                print(f'    Adaptive(k-means) SuperTrend...')
                for idx, upper in enumerate(ind_st.his_val_upper): # down_trend
                    self.traces1.extend([
                        go.Scatter(x=ind_st.his_ts_upper[idx], y=upper,
                                   mode='lines',
                                   line=dict(color='red', width=3), # dash='dot'
                                   opacity=1, showlegend=False),
                    ])

                for idx, lower in enumerate(ind_st.his_val_lower): # up_trend
                    self.traces1.extend([
                        go.Scatter(x=ind_st.his_ts_lower[idx], y=lower,
                                   mode='lines',
                                   line=dict(color='green', width=3), # dash='dot'
                                   opacity=1, showlegend=False),
                    ])
                    
            if plot_labels:
                self.traces0.extend([
                    go.Scatter(x=timestamps, y=labels,
                               mode='lines',
                               line=dict(color='blue', width=2), # dash='dot'
                               opacity=1, showlegend=False),
                ])
                
                self.traces0.extend([
                    go.Scatter(x=timestamps, y=[0.0] * len(timestamps),
                               mode='lines',
                               line=dict(color='black', width=2), # dash='dot'
                               opacity=1, showlegend=False),
                ])
                
        return self.fig

    def plot(self,
             kl_datas: Dict[KL_TYPE, CKLine_List],
             markers: List[Tuple],
             indicators: List,
             **kwargs):
        """Convenience method to draw both KLC and Bi elements"""
        self.markers = markers
        self.indicators = indicators
        
        # Create a subplot for volume
        self.fig = make_subplots(rows=3, cols=1,
                                 row_heights=[0.3, 0.6, 0.1],
                                 vertical_spacing=0.05,
                                 shared_xaxes=True)
        
        # plot from small to big
        no_lv = len(self.lv_type_list)
        for lv_idx_rev, lv in enumerate(reversed(self.lv_type_list)):
            if lv_idx_rev == 0:
                self.current_ts = kl_datas[lv][-1][-1].time.ts
            self.lv = lv
            self.lv_idx_rev = lv_idx_rev
            self.lv_idx = no_lv - 1 - lv_idx_rev
            self.klc_list = kl_datas[lv]
            self.color = self.color_list[self.lv_idx]
            self.opacity = self.opacity_list[self.lv_idx]
            if self.lv_idx == no_lv - 1 - 2: # 5M
                self.draw_klu()
                self.draw_klc()
                self.draw_volume()
            self.draw_charts(text=False)
            if self.lv_idx in range(no_lv-1-2, no_lv): # 1M, 5M, 15M
                self.draw_bi()
            self.draw_liquidity_zones()
            self.draw_volume_profile()
        self.draw_markers()
        self.draw_ind()

        # Add traces
        self.fig.add_traces(self.traces0, rows=1, cols=1)
        self.fig.add_traces(self.traces1, rows=2, cols=1)
        self.fig.add_traces(self.traces2, rows=3, cols=1)

        all_shapes = []
        for shape in self.shapes:
            shape_copy = shape.copy()
            shape_copy['yref'] = 'y2'  # Second row y-axis
            shape_copy['xref'] = 'x2'  # Second row x-axis
            all_shapes.append(shape_copy)
            
        all_annotations = []
        for annotation in self.annotations:
            # Force annotation to be on second row's axes
            annotation_copy = annotation.copy()
            annotation_copy['yref'] = 'y2'  # Second row y-axis
            annotation_copy['xref'] = 'x2'  # Second row x-axis
            all_annotations.append(annotation_copy)
        
        # Add shapes
        self.fig.update_layout(shapes=all_shapes)

        # Update annotations
        self.fig.update_layout(annotations=all_annotations)

        # Update layout
        self.fig.update_layout(
            showlegend=False,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(128, 128, 128, 0.3)',  # Gray with 10% opacity
            width=800,  # Set default width
            height=600,  # Set default height
            dragmode='zoom',     # Enable box zoom by default
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
        # Update y-axis label for ML_labels
        self.fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        self.fig.update_yaxes(title_text="Label", row=1, col=1)
        # Update both x and y axes for main and volume charts
        self.fig.update_xaxes(title_text="Timestamp", rangeslider_visible=False, row=2, col=1, showgrid=True,
                              showline=True, linewidth=1, linecolor='black', mirror=True,
                              # type='date', tickformat='%Y-%m-%d %H:%M:%S', dtick='auto',
                              )
        self.fig.update_yaxes(title_text="Price", row=2, col=1, showgrid=True,
                              showline=True, linewidth=1, linecolor='black', mirror=True)
        # Update y-axis label for volume
        self.fig.update_xaxes(rangeslider_visible=False, row=3, col=1)
        self.fig.update_yaxes(title_text="Volume", row=3, col=1)

        return self.fig
