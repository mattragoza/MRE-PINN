import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import xarray as xr
import deepxde
import ipywidgets

from .utils import exists, print_if, as_iterable
from . import discrete

DPI = 50


class XArrayViewer(object):

    def __init__(
        self,
        xarray,
        x='x',
        y='y',
        hue=None,
        row=None,
        col=None,
        verbose=False,
        **kwargs
    ):
        xarray = self.preprocess_array(xarray)

        self.establish_dimensions(xarray.dims, x, y, hue, row, col)

        print_if(verbose, self.index_dims, self.value_dims)
        print_if(verbose, self.value_dim_map)

        self.set_array_internals(xarray)
        self.initialize_subplots(**kwargs)

    def preprocess_array(self, xarray):
        '''
        Concatenate spatial with frequency domain.
        Also concatenate real and imaginary parts.
        '''
        if 'domain' not in xarray.dims:
            xarray = xr.concat(
                [xarray, discrete.sfft(xarray)],
                dim=xr.DataArray(['space', 'frequency'], dims=['domain'])
            )

        if np.iscomplexobj(xarray):
            xarray = xr.concat(
                [xarray.real, xarray.imag],
                dim=xr.DataArray(['real', 'imag'], dims=['part'])
            )

        return xarray

    def establish_dimensions(self, dims, x, y, hue, row, col):
        '''
        Establish the mapping from array
        dimensions to subplot components.
        '''
        value_dims = []
        value_dim_map = {}

        if exists(row):
            value_dims.append(row)
            value_dim_map['row'] = row

        if exists(col):
            value_dims.append(col)
            value_dim_map['col'] = col

        value_dims.append(x)
        value_dim_map['x'] = x

        if exists(y):
            value_dims.append(y)
            value_dim_map['y'] = y

        if exists(hue):
            value_dims.append(hue)
            value_dim_map['hue'] = hue

        self.value_dims = value_dims
        self.value_dim_map = value_dim_map

        self.index_dims = [d for d in dims if d not in value_dims]

    def set_array_internals(self, xarray):

        # permute value dims to the end
        xarray = xarray.transpose(*(self.index_dims + self.value_dims))

        # set internal array, dims, and coords
        self.array = xarray.to_numpy()
        self.dims = list(xarray.dims)
        self.coords = {d: list(xarray.coords[d].to_numpy()) for d in self.dims}

        # initial index state
        self.index = (0,) * len(self.index_dims)

    def get_index_and_labels(self, i, j):
        '''
        Return the array index and axes labels
        associated with a given row and column.
        '''
        index = self.index
        row_label = col_label = ''
        if exists(self.row_dim):
            index += (i,)
            if j == 0: # first column
                row_label = str(self.coords[self.row_dim][i]) + '\n'
        if exists(self.col_dim):
            index += (j,)
            if i == 0: # first row
                col_label = str(self.coords[self.col_dim][j])
        return index, row_label, col_label

    def initialize_subplots(
        self, ax_height=None, ax_width=None, dpi=None, **kwargs
    ):
        # determine number of axes
        n_rows, n_cols = (1, 1)
        row_dim = self.value_dim_map.get('row', None)
        col_dim = self.value_dim_map.get('col', None)
        if row_dim is not None:
            n_rows = len(self.coords[row_dim])
        if col_dim is not None:
            n_cols = len(self.coords[col_dim])

        self.row_dim = row_dim
        self.col_dim = col_dim
        self.n_rows = n_rows
        self.n_cols = n_cols

        # determine plot type
        x_dim = self.value_dim_map.get('x', None)
        y_dim = self.value_dim_map.get('y', None)
        hue_dim = self.value_dim_map.get('hue', None)
        do_line_plot  = (y_dim is None)
        do_image_plot = (hue_dim is None and not do_line_plot)
        assert do_line_plot or do_image_plot

        # determine axes size
        n_x, n_y = (1, 1)
        if exists(x_dim):
            n_x = len(self.coords[x_dim])
        if exists(y_dim):
            n_y = len(self.coords[y_dim])
        else: # line plot
            n_y = n_x // 2

        if ax_height is None and ax_width is None:
            dpi = dpi or DPI
            ax_height = n_y / dpi
            ax_width  = n_x / dpi
        elif ax_height is None:
            ax_height = ax_width * n_y / n_x
        elif ax_width is None:
            ax_width = ax_height * n_x / n_y

        ax_height = [ax_height] * n_rows
        ax_width  = [ax_width]  * n_cols

        # create the subplot grid
        self.fig, self.axes, self.cbar_ax = subplot_grid(
            n_rows=n_rows,
            n_cols=n_cols,
            ax_height=ax_height,
            ax_width=ax_width,
            cbar_width=0.25 * do_image_plot,
            space=[0.25, 0.50],
            pad=[0.85, 0.35, 0.55, 0.45]
        )

        # plot the array data and store the artists
        self.artists = [
            [None for i in range(n_cols)] for j in range(n_rows)
        ]
        for i in range(n_rows):
            for j in range(n_cols):
                index, row_label, col_label = self.get_index_and_labels(i, j)
                x_res = self.coords[x_dim][1] - self.coords[x_dim][0]
                if do_line_plot:
                    lines = plot_line_1d(
                        self.axes[i,j],
                        self.array[index],
                        resolution=x_res,
                        xlabel=x_dim,
                        ylabel=row_label,
                        title=col_label
                        **kwargs
                    )
                    if len(lines) > 1:
                        self.artists[i][j] = lines
                    else:
                        self.artists[i][j] = lines[0]
                else: # plot image
                    image = plot_image_2d(
                        self.axes[i,j],
                        self.array[index],
                        resolution=x_res,
                        xlabel=x_dim,
                        ylabel=row_label + y_dim,
                        title=col_label,
                        **kwargs
                    )
                    self.artists[i][j] = image

        if do_image_plot: # create colorbar
            self.cbar = plot_colorbar(self.cbar_ax, image)
            for i in range(n_rows):
                for j in range(n_cols):
                    self.artists[i][j].set_norm(self.cbar.norm)

        # create interactive sliders for index dims
        self.sliders = []
        for d in self.index_dims:
            slider = ipywidgets.SelectionSlider(
                options=[(c, i) for i, c in enumerate(self.coords[d])],
                description=d
            )
            self.sliders.append(slider)

        ipywidgets.interact(
            self.update_index,
            **{d: s for d, s in zip(self.index_dims, self.sliders)}
        )

    def update_index(self, **kwargs):
        coords = {d: self.coords[d][i] for d, i in kwargs.items()}
        self.index = tuple([kwargs[d] for d in self.index_dims])
        self.update_artists()

    def update_array(self, xarray):
        xarray = self.preprocess_array(xarray)
        xarray = xarray.transpose(*(self.index_dims + self.value_dims))
        self.array = xarray.to_numpy()
        self.update_artists()

    def update_artists(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                index, row_label, col_label = self.get_index_and_labels(i, j)
                artist = self.artists[i][j]
                if isinstance(artist, list): # multiple lines
                    for k, artist in enumerate(artist):
                        artist.set_ydata(self.array[index].T[k])
                elif isinstance(artist, matplotlib.lines.Line2D): # one line
                    artist.set_ydata(self.array[index])
                else: # image
                    artist.set_array(self.array[index].T)
        self.fig.canvas.draw()


class TrainingPlot(deepxde.display.TrainingDisplay):

    def __init__(self, losses, metrics):
        self.losses = losses
        self.metrics = metrics
        self.initialized = False

    def initialize(self):

        self.fig, self.axes = subplot_grid(
            n_rows=1,
            n_cols=2,
            ax_height=3,
            ax_width=3,
            space=[0.3, 0.3],
            pad=[1.2, 0.4, 0.7, 0.4]
        )

        # training loss and metric plots
        self.loss_lines = []
        for loss in self.losses:
            line, = self.axes[0,0].plot([], [], label=loss)
            self.loss_lines.append(line)

        self.metric_lines = []
        for metric in self.metrics:
            line, = self.axes[0,1].plot([], [], label=metric)
            self.metric_lines.append(line)

        self.axes[0,0].set_ylabel('loss')
        self.axes[0,1].set_ylabel('metric')

        for ax in self.axes.flatten():
            ax.set_xlabel('iteration')
            ax.set_yscale('log')
            #ax.grid(linestyle=':')

        if self.loss_lines:
            self.axes[0,0].legend(frameon=True, edgecolor='0.2')

        if self.metric_lines:
            self.axes[0,1].legend(frameon=True, edgecolor='0.2')

        self.initialized = True

    def __call__(self, train_state):

        if not self.initialized:
            self.initialize()
        
        for i, line in enumerate(self.loss_lines):
            new_x = train_state.step
            new_y = train_state.loss_test[i]
            line.set_xdata(np.append(line.get_xdata(), new_x))
            line.set_ydata(np.append(line.get_ydata(), new_y))

        for i, line in enumerate(self.metric_lines):
            new_x = train_state.step
            new_y = train_state.metrics_test[i]
            line.set_xdata(np.append(line.get_xdata(), new_x))
            line.set_ydata(np.append(line.get_ydata(), new_y))
        
        for ax in self.axes.flatten():
            ax.relim()
            ax.autoscale_view()

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


def elast_color_map(n_colors=255, symmetric=False):
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
    if symmetric:
        colors = colors[::-1] + colors[1:]

    return mpl.colors.LinearSegmentedColormap.from_list(
        name='elast', colors=colors, N=n_colors
    )


class Colorbar(matplotlib.colorbar.Colorbar):

    def drag_pan(self, button, key, x, y):
        points = self.ax._get_pan_points(button, key, x, y)

        if points is not None:
            if self.orientation == 'horizontal':
                vmin, vmax = points[:, 0]
            elif self.orientation == 'vertical':
                vmin, vmax = points[:, 1]

        if button == 3: # fix the zero position
            old_vrange = self.norm.vmax - self.norm.vmin
            new_vrange = vmax - vmin
            self.norm.vmin *= new_vrange / old_vrange
            self.norm.vmax *= new_vrange / old_vrange
        else:
            self.norm.vmin = vmin
            self.norm.vmax = vmax


def subplot_grid(n_rows, n_cols, ax_height, ax_width, cbar_width=0, space=0.3, pad=0):
    '''
    Args:
        n_rows
        n_cols
        ax_height
        ax_width
        cbar_width
        space: (vertical, horizontal)
        pad: (left, right, bottom, top)
    Returns:
        fig, axes, cbar_ax
    '''
    ax_height = as_iterable(ax_height, n_rows)
    ax_width = as_iterable(ax_width, n_cols)
    hspace, wspace = as_iterable(space, 2)
    lpad, rpad, bpad, tpad = as_iterable(pad, 4)

    fig_height = sum(ax_height) + (n_rows - 1) * hspace + bpad + tpad
    fig_width  = sum(ax_width)  + (n_cols - 1) * wspace + lpad + rpad

    if cbar_width:
        extra_width = cbar_width + wspace
        fig_width += extra_width
    else:
        extra_width = 0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        squeeze=False,
        figsize=(fig_width, fig_height),
        gridspec_kw=dict(
            height_ratios=ax_height,
            width_ratios=ax_width,
            hspace=hspace,
            wspace=wspace,
            left=lpad/fig_width,
            right=1.0 - (rpad + extra_width)/fig_width,
            bottom=bpad/fig_height,
            top=1.0 - tpad/fig_height
        )
    )
    if cbar_width:
        cbar_ax = fig.add_axes([
            (sum(ax_width) + n_cols * wspace + lpad)/fig_width,
            bpad/fig_height * 1.1,
            cbar_width/fig_width,
            1.0 - (bpad + tpad)/fig_height * 1.1
        ])
    else:
        cbar_ax = None
    return fig, axes, cbar_ax


def plot_line_1d(ax, a, resolution, xlabel=None, ylabel=None, title=None, **kwargs):
    if a.ndim == 2:
        n_x, n_hue = a.shape
    else:
        n_x, = a.shape
    x = np.arange(n_x) * resolution
    lines = ax.plot(x, a)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(kwargs.get('vmin', None), kwargs.get('vmax', None))
    return lines


def imshow(ax, a, resolution=1, **kwargs):
    if im.ndim == 2:
        n_x, n_y = a.shape
        a_T = a.T
    elif im.ndim == 3:
        n_x, n_y, n_c = a.shape
        a_T = np.transpose(a, (1, 0, 2))
    extent = (0, n_x * resolution, 0, n_y * resolution)
    return ax.imshow(a_T, origin='lower', extent=extent, **kwargs)


def plot_image_2d(ax, a, resolution, xlabel=None, ylabel=None, title=None, **kwargs):
    n_x, n_y = a.shape
    extent = (0, n_x * resolution, 0, n_y * resolution)
    ax.autoscale(enable=True, tight=True)
    if 'vmax' in kwargs and 'vmin' not in kwargs:
        kwargs['vmin'] = -kwargs['vmax']
    im = ax.imshow(a.T, origin='lower', extent=extent, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im


def plot_points_2d(ax, x, u, dims, xlabel=None, ylabel=None, **kwargs):
    sc = ax.scatter(x[:,0], x[:,1], c=u, marker='o', s=0.2, **kwargs)
    ax.set_aspect(dims[1] / dims[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return sc


def plot_colorbar(ax, mappable, label=None):
    cbar = Colorbar(ax, mappable)
    #ax.yaxis.set_ticks_position('left')
    #ax.yaxis.set_label_position('left')
    ax.set_ylabel(label)
    return cbar


def plot_slider(ax, update, values=None, label=None, **kwargs):

    n_values = len(values)
    slider = matplotlib.widgets.Slider(
        ax,
        label=None,
        valmin=0,
        valmax=max(n_values - 1, 1e-5),
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
