from math import inf
from click import pause
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from Chan import CChan

from .PlotDriver import CPlotDriver


class CAnimateDriver:
    def __init__(self, chan: CChan, plot_config=None, plot_para=None):
        if plot_config is None:
            plot_config = {}
        if plot_para is None:
            plot_para = {}
        pause_time = plot_para.get('animation_pause_time', 0.0)
        for idx, _ in enumerate(chan.step_load()):
            if idx < inf:
                g = CPlotDriver(chan, plot_config, plot_para)
                # g.save2img(f'{idx}.png')
                clear_output(wait=True)
                display(g.figure)
                
                time.sleep(pause_time)
                plt.close(g.figure)
