from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver

def main():
    code = "sh.688099"
    begin_time = "2024-01-01"
    end_time = None
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig({
        "bi_strict": True,
        "trigger_step": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": False,
        "plot_zs": True,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": True,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    plot_para = {
        "seg": {
            "plot_trendline": True,
        },
        "bi": {
            "show_num": True,
            "disp_end": True,
        },
        "figure": {
            "x_range": 200,
        },
        "marker": {
            "markers": {  # text, position, color
                '2023/06/01': ('marker here', 'up', 'red'),
                '2023/06/08': ('marker here', 'down')
            },
        },
        "animation_pause_time": 0
    }
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    if not config.trigger_step:
        plot_driver = CPlotDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
        # plot_driver.save2img('./image.png')
        plot_driver.figure.show()
    else:
        CAnimateDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )
# input()

if __name__ == "__main__":
    main()