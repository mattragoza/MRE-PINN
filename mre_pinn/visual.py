import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets


def wave_color_map(n_colors=255):
    '''
    Create a colormap for MRE wave images
    from yellow, red, black, blue, to cyan.
    '''
    cyan   = (0, 1, 1)
    blue   = (0, 0, 1)
    black  = (0, 0, 0)
    red    = (1, 0, 0)
    yellow = (1, 1, 0)

    colors = [cyan, blue, black, red, yellow]

    return mpl.colors.LinearSegmentedColormap.from_list(
        name='wave', colors=colors, N=n_colors
    )


def elast_color_map(n_colors=255):
    '''
    Create a colormap for MRE elastrograms
    from dark, blue, cyan, green, yellow, to red.
    '''
    p = 0.0
    c = 0.9 #0.6
    y = 0.9
    g = 0.8

    dark   = (p, 0, p)
    blue   = (0, 0, 1)
    cyan   = (0, c, 1)
    green  = (0, g, 0)
    yellow = (1, y, 0)
    red    = (1, 0, 0)

    colors = [dark, blue, cyan, green, yellow, red]

    return mpl.colors.LinearSegmentedColormap.from_list(
        name='elast', colors=colors, N=n_colors
    )


class Player(FuncAnimation):
    def __init__(
        self, fig, func, frames=None, init_func=None, fargs=None,
        save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs
    ):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(
            self, self.fig, self.update, frames=self.play(), init_func=init_func,
            fargs=fargs, save_count=save_count, **kwargs
        )    

    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i + self.forwards - (not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(sliderax, '', self.min, self.max, valinit=self.i)
        self.slider.on_changed(self.set_pos)

    def set_pos(self,i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        self.slider.set_val(i)
