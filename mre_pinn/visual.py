import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets


class Player(FuncAnimation):

    def __init__(
        self, fig, func, frames=None, init_func=None, fargs=None,
        save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs
    ):
        self.n_frames = frames
        self.curr_frame = 0
        self.playing = True
        self.forwards = True

        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(
            self, self.fig, self.update, frames=frames, init_func=init_func,
            fargs=fargs, save_count=save_count, **kwargs
        )

    def start(self):
        self.playing = True
        self.event_source.start()

    def stop(self, event=None):
        print('STOPPING')
        self.playing = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def step_forward(self, event=None):
        self.forwards = True
        self.step(1)

    def step_backward(self, event=None):
        self.forwards = False
        self.step(-1)

    def step(self, increment):
        new_frame = (self.curr_frame + increment) % self.n_frames
        self.curr_frame = new_frame
        self.func(new_frame)
        self.slider.set_val(new_frame)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        player_ax = self.fig.add_axes([pos[0], pos[1], 0.72, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(player_ax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        slider_ax = divider.append_axes("right", size="500%", pad=0.15)

        self.button_step_backward = mpl.widgets.Button(player_ax, label='$\u29CF$')
        self.button_backward = mpl.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = mpl.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = mpl.widgets.Button(fax, label='$\u25B6$')
        self.button_step_forward = mpl.widgets.Button(ofax, label='$\u29D0$')

        self.button_step_backward.on_clicked(self.step_backward)
        self.button_backward.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_step_forward.on_clicked(self.step_forward)

        self.slider = mpl.widgets.Slider(
            slider_ax, '', 0, self.n_frames, valinit=self.curr_frame
        )
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self,i):
        self.slider.set_val(i)


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


def plot_image_2d(a, resolution, ax, xlabel=None, ylabel=None, **kwargs):
    n_x, n_y = a.shape
    extent = (0, n_x * resolution, 0, n_y * resolution)
    ax.autoscale(enable=True, tight=True)
    im = ax.imshow(a.T, origin='lower', extent=extent, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def plot_points_2d(x, u, dims, ax, xlabel=None, ylabel=None, **kwargs):
    sc = ax.scatter(x[:,0], x[:,1], c=u, marker='o', s=0.2, **kwargs)
    ax.set_aspect(dims[1] / dims[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return sc


def plot_colorbar(obj, ax, label=None):
    plt.colorbar(obj, cax=ax)
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')
    ax.set_ylabel(label)
