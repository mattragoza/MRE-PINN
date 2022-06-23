import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import xarray as xr
import deepxde


class NDArrayViewer(object):
    
    def __init__(
        self, array, x='x', y='y', labels=None, dpi=25, **kwargs
    ):
        if isinstance(array, xr.DataArray) and labels is None:
            labels = array.dims
        labels = list(labels)

        assert x in labels
        assert y in labels

        # permute so that x and y are at the end
        permute = list(range(array.ndim))
        x_dim = labels.index(x)
        permute.append(permute.pop(x_dim))
        labels.append(labels.pop(x_dim))
        y_dim = labels.index(y)
        permute.append(permute.pop(y_dim))
        labels.append(labels.pop(y_dim))
        array = np.transpose(array, permute)

        # set up grid of subplots
        n_x, n_y  = array.shape[-2:]
        ax_height = n_y / dpi
        ax_width  = n_x / dpi
        bar_width = 1 / 4

        assert array.ndim >= 2
        assert len(labels) == array.ndim

        self.fig, self.axes = subplot_grid(
            n_rows=1,
            n_cols=array.ndim,
            ax_height=ax_height,
            ax_width=[bar_width] * (array.ndim - 2) + [ax_width, bar_width],
            space=[0.3, 0.6],
            pad=[0.9, 1.0, 0.7, 0.4]
        )

        self.array = array
        self.index = (0,) * (array.ndim - 2)

        # create index sliders
        self.sliders = []
        for i in range(array.ndim - 2):
            if isinstance(array, xr.DataArray):
                values = array.coords[labels[i]].to_numpy()
            else:
                values = np.arange(array.shape[i])
            slider = plot_slider(
                self.axes[0,i],
                update=self.index_updater(i),
                values=values,
                label=labels[i]
            )
            self.sliders.append(slider)

        # create image and color bar
        self.image = plot_image_2d(
            self.axes[0,-2],
            self.array[self.index],
            resolution=1,
            xlabel=labels[-2],
            ylabel=labels[-1],
            **kwargs
        )
        self.cbar = plot_colorbar(self.axes[0,-1], self.image)

    def index_updater(self, i):
        def update(new_value):
            curr_index = list(self.index)
            curr_index[i] = new_value
            self.index = tuple(curr_index)
            self.image.set_array(self.array[self.index].T)
            self.fig.canvas.draw()
        return update


class TrainingPlot(deepxde.display.TrainingDisplay):

    def __init__(self):
        self.initialized = False

    def initialize(self):

        self.fig, self.axes = subplot_grid(
            n_rows=1,
            n_cols=1,
            ax_height=4,
            ax_width=4,
            space=0.3,
            pad=[1.2, 0.4, 0.7, 0.4]
        )

        # training loss plot
        labels = ['PDE', 'data']
        self.lines = [
            self.axes[0,0].plot([], [], label=labels[i])[0] for i in range(2)
        ]
        self.axes[0,0].legend(frameon=False)
        self.axes[0,0].grid(linestyle=':')
        self.axes[0,0].set_xlabel('iteration')
        self.axes[0,0].set_ylabel('loss')
        self.axes[0,0].set_yscale('log')
        self.initialized = True

    def __call__(self, train_state):

        if not self.initialized:
            self.initialize()
        
        for i, line in enumerate(self.lines):
            new_x = train_state.step
            new_y = train_state.loss_train[i]
            line.set_xdata(np.append(line.get_xdata(), new_x))
            line.set_ydata(np.append(line.get_ydata(), new_y))
            
        self.axes[0,0].relim()
        self.axes[0,0].autoscale_view()
        self.fig.canvas.draw()


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


def as_iterable(x, length=1):
    if not isinstance(x, (list, tuple)):
        return [x] * length
    return x


def subplot_grid(n_rows, n_cols, ax_height, ax_width, space=0, pad=0):
    
    ax_height = as_iterable(ax_height, n_rows)
    ax_width = as_iterable(ax_width, n_cols)
    hspace, wspace = as_iterable(space, 2)
    lpad, rpad, bpad, tpad = as_iterable(pad, 4)

    fig_height = sum(ax_height) + (n_rows - 1) * hspace + bpad + tpad
    fig_width = sum(ax_width) + (n_cols - 1) * wspace + lpad + rpad

    return plt.subplots(
        n_rows, n_cols,
        squeeze=False,
        figsize=(fig_width, fig_height),
        gridspec_kw=dict(
            height_ratios=ax_height,
            width_ratios=ax_width,
            hspace=hspace,
            wspace=wspace,
            left=lpad/fig_width,
            right=1 - rpad/fig_width,
            bottom=bpad/fig_height,
            top=1 - tpad/fig_height
        )
    )


def plot_image_2d(ax, a, resolution, xlabel=None, ylabel=None, **kwargs):
    n_x, n_y = a.shape
    extent = (0, n_x * resolution, 0, n_y * resolution)
    ax.autoscale(enable=True, tight=True)
    im = ax.imshow(a.T, origin='lower', extent=extent, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def plot_points_2d(ax, x, u, dims, xlabel=None, ylabel=None, **kwargs):
    sc = ax.scatter(x[:,0], x[:,1], c=u, marker='o', s=0.2, **kwargs)
    ax.set_aspect(dims[1] / dims[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return sc


def plot_colorbar(ax, obj, label=None, location='left'):
    plt.colorbar(obj, cax=ax)
    #ax.yaxis.set_ticks_position('left')
    #ax.yaxis.set_label_position('left')
    ax.set_ylabel(label)


def plot_slider(ax, update, values=None, label=None, **kwargs):

    n_values = len(values)
    slider = matplotlib.widgets.Slider(
        ax,
        label=None,
        valmin=0,
        valmax=n_values - 1,
        valstep=1,
        orientation='vertical',
        handle_style=dict(size=20)
    )
    slider.on_changed(update)

    #slider.track.set_xy((0, 0))
    #slider.track.set_width(1.0)
    #slider.poly.set_visible(False)
    #slider.hline.set_visible(False)

    slider.label.set_y(1.05)
    slider.valtext.set_y(-0.05)
    slider.valtext.set_visible(False)
    slider._handle.set_marker('s')
    slider._handle.set_zorder(10)

    ax.set_axis_on()
    ax.set_xticks([])

    #ax.hlines(range(n_values), 0.25, 0.75, color='0.0', lw=0.7, zorder=9)
    ax.set_yticks(range(n_values))
    ax.set_yticklabels(values)
    #ax.set_xlim(0, 1)

    ax.set_ylabel(label)
    for s in ax.spines:
        ax.spines[s].set_visible(False)

    return slider
