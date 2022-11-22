import numpy as np
import xarray as xr
import deepxde

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import seaborn as sns
import ipywidgets

from .utils import exists, print_if, as_iterable

DPI = 50


class Viewer(object):

    def to_png(self, png_file):
        self.fig.savefig(png_file, bbox_inches='tight')


class XArrayViewer(Viewer):

    def __init__(
        self,
        xarray,
        x='x',
        y='y',
        hue=None,
        row=None,
        col=None,
        polar=False,
        verbose=False,
        **kwargs
    ):
        self.polar = polar
        xarray = self.preprocess_array(xarray, polar=polar)

        self.establish_dimensions(xarray.dims, x, y, hue, row, col)

        print_if(verbose, self.index_dims, self.value_dims)
        print_if(verbose, self.value_dim_map)

        default_kws = get_color_kws(xarray)
        default_kws['ax_height'] = 4
        default_kws.update(**kwargs)

        self.set_array_internals(xarray)
        self.initialize_subplots(**default_kws)

    def preprocess_array(self, xarray, polar=False):
        '''
        Concatenate spatial with frequency domain.
        Also concatenate real and imaginary parts.
        '''
        if 'domain' not in xarray.dims:
            xarray = xr.concat(
                [xarray, xarray.field.fft()],
                dim=xr.DataArray(['space', 'frequency'], dims=['domain'])
            )

        if np.iscomplexobj(xarray):
            if polar:
                xarray = xr.concat(
                    [np.abs(xarray), np.arctan(xarray.imag / xarray.real)],
                    dim=xr.DataArray(['abs', 'angle'], dims=['part'])
                )
            else:
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
        self,
        ax_height=None,
        ax_width=None,
        dpi=None,
        cbar_width=0.25,
        space=[0.25, 0.50],
        pad=[0.95, 0.75, 0.65, 0.45],
        interact=True,
        **kwargs
    ):
        # determine number of axes
        n_rows, n_cols = (1, 1)
        row_dim = self.value_dim_map.get('row')
        col_dim = self.value_dim_map.get('col')
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
            cbar_width=cbar_width * do_image_plot,
            space=space,
            pad=pad
        )

        # plot the array data and store the artists
        self.artists = [
            [None for i in range(n_cols)] for j in range(n_rows)
        ]
        for i in range(n_rows):
            for j in range(n_cols):
                index, row_label, col_label = self.get_index_and_labels(i, j)
                x0 = self.coords[x_dim][0]
                y0 = self.coords[y_dim][0]
                x_res = self.coords[x_dim][1] - self.coords[x_dim][0]
                y_res = self.coords[y_dim][1] - self.coords[y_dim][0]
                if do_line_plot:
                    lines = plot_line_1d(
                        self.axes[i,j],
                        self.array[index],
                        resolution=x_res,
                        xlabel=x_dim,
                        ylabel=row_label,
                        title=col_label,
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
                        origin=[x0, y0],
                        resolution=[x_res, y_res],
                        xlabel=x_dim,
                        ylabel=row_label + y_dim,
                        title=col_label,
                        interpolation_stage='rgba',
                        **kwargs
                    )
                    self.artists[i][j] = image

        if do_image_plot: # create colorbar
            self.cbar = plot_colorbar(self.cbar_ax, image)
            for i in range(n_rows):
                for j in range(n_cols):
                    self.artists[i][j].set_norm(self.cbar.norm)

        if interact: # create interactive sliders for index dims
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
        xarray = self.preprocess_array(xarray, polar=self.polar)
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


def my_line_plot(data, x, y, hue, hue_order, colors, ax, **kwargs):
    lines = []
    data = data.set_index(hue).sort_values(x)
    for hue_level, color in zip(hue_order, colors):
        if hue_level not in data.index:
            lines.append(None)
            continue
        hue_data = data.loc[hue_level]
        line, = ax.plot(hue_data[x], hue_data[y], color=color, label=hue_level)
        lines.append(line)
    return lines


class DataViewer(Viewer):
    '''
    XArrayViewer but for a pd.DataFrame.
    '''
    def __init__(
        self, data, x, y, hue=None, row=None, col=None, **kwargs
    ):
        data = data.reset_index()
        self.establish_variables(x, y, hue, row, col)
        self.set_data_internals(data)
        self.initialize_subplots(**kwargs)

    def establish_variables(self, x, y, hue, row, col):
        variable_map = {'x': x, 'y': y}
        index_vars = []
 
        if exists(row):
            variable_map['row'] = row
            index_vars.append(row)

        if exists(col):
            variable_map['col'] = col
            index_vars.append(col)

        if exists(hue):
            variable_map['hue'] = hue
            index_vars.append(hue)

        self.variable_map = variable_map
        self.index_vars = index_vars

    def set_data_internals(self, data):

        # set internal data frame and index levels
        if self.index_vars:
            levels = {
                v: data[v].unique() for v in self.index_vars
            }
            data = data.set_index(self.index_vars)
        else:
            levels = {}

        self.data = data.sort_index()
        self.levels = levels

    def get_index_and_labels(self, i, j):
        '''
        Return the data index and axes labels
        associated with a given row and column.
        '''
        index = ()
        row_label = col_label = ''
        row_var = self.variable_map.get('row')
        col_var = self.variable_map.get('col')
        if exists(row_var):
            row_level = self.levels[row_var][i]
            index += (row_level,)
            row_label = str(row_level) + ' '
        if exists(col_var):
            col_level = self.levels[col_var][j]
            index += (col_level,)
            if i == 0: # first row
                col_label = str(col_level)
        return index, row_label, col_label

    def initialize_subplots(
        self, ax_height=2, ax_width=1.5, lgd_width=0.75, palette=None, **kwargs
    ):
        n_rows, n_cols = (1, 1)
        row_var = self.variable_map.get('row')
        col_var = self.variable_map.get('col')
        if row_var is not None:
            n_rows = len(self.levels[row_var])
        if col_var is not None:
            n_cols = len(self.levels[col_var])
        n_axes = n_rows * n_cols

        self.n_rows = n_rows
        self.n_cols = n_cols

        # determine plot type
        x_var = self.variable_map.get('x')
        y_var = self.variable_map.get('y')
        hue_var = self.variable_map.get('hue')
        hue_order = self.levels.get(hue_var)
        n_hues = len(hue_order) if hue_var else None
        colors = sns.color_palette(palette, n_hues)

        # create subplot grid
        self.fig, self.axes, self.lgd_ax = subplot_grid(
            n_rows=n_rows,
            n_cols=n_cols,
            ax_height=ax_height,
            ax_width=ax_width,
            cbar_width=lgd_width * bool(hue_var),
            space=[0.30, 0.50],
            pad=[1.00, 0.45, 0.65, 0.45]
        )

        # plot the data and store the artists
        self.artists = {}
        for i in range(n_rows):
            for j in range(n_cols):

                index, row_label, col_label = self.get_index_and_labels(i, j)
                columns = [x_var, y_var]
                data = self.data.loc[index][columns].dropna().reset_index()

                ax = self.axes[i,j]
                ax.set_title(col_label)
                ax.grid(linestyle=':')

                lines = my_line_plot(
                    data=data,
                    x=x_var,
                    y=y_var,
                    hue=hue_var,
                    hue_order=hue_order,
                    colors=colors,
                    ax=ax,
                    **kwargs
                )
                ax.set_yscale('log')

                for hue_level, line in zip(hue_order, lines):
                    self.artists[index + (hue_level,)] = line

                if ax.legend_:
                    ax.legend_.remove()

                if j == 0: # first column
                    ax.set_ylabel(row_label + y_var)
                else:
                    ax.set_ylabel(None)
                if i + 1 >= len(self.levels.get(row_var, [])): # last row
                    ax.set_xlabel(x_var)
                else:
                    ax.set_xlabel(None)

        sns.despine(self.fig)

        if hue_var: # create legend on extra axes
            lgd = self.lgd_ax.legend(
                *ax.get_legend_handles_labels(),
                title=hue_var,
                loc='upper left',
                bbox_to_anchor=[-0.5,1.15],
                frameon=False
            )
            lgd._legend_box.align = 'left'
            self.lgd_ax.xaxis.set_visible(False)
            self.lgd_ax.yaxis.set_visible(False)
            sns.despine(ax=self.lgd_ax, left=True, right=True, bottom=True, top=True)

        self.fig.canvas.draw()

    def update_data(self, data):
        data = data.reset_index()
        data = data.set_index(self.index_vars)
        data = data.sort_index()
        self.data = data
        self.update_artists()

    def update_artists(self):
        x_var = self.variable_map.get('x')
        y_var = self.variable_map.get('y')
        hue_var = self.variable_map.get('hue')
        hue_order = self.levels.get(hue_var)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                index, row_label, col_label = self.get_index_and_labels(i, j)
                columns = [x_var, y_var]
                data = self.data.loc[index][columns].dropna()
                for hue_level in hue_order:
                    if hue_level not in data.index:
                        continue
                    artist = self.artists[index + (hue_level,)]
                    hue_data = data.loc[hue_level]
                    artist.set_xdata(hue_data[x_var].values)
                    artist.set_ydata(hue_data[y_var].values)
                self.axes[i,j].relim()
                self.axes[i,j].autoscale_view()

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


def grayscale_color_map(n_colors=255, reverse=False, symmetric=False):
    '''
    Create a colormap for MRI magnitude images
    from black to white.
    '''
    black = (0, 0, 0)
    white = (1, 1, 1)

    if symmetric and reverse:
        colors = [black, white, black]
    elif symmetric:
        colors = [white, black, white]
    elif reverse:
        colors = [white, black]
    else:
        colors = [black, white]

    return mpl.colors.LinearSegmentedColormap.from_list(
        name='magnitude', colors=colors, N=n_colors
    )


def wave_color_map(n_colors=255, reverse=False):
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

    if reverse:
        colors = colors[::-1]

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


def region_color_map(n_colors=255, has_background=False):
    '''
    Create a colormap for segmentation regions
    from white, red, yellow, green, to blue.
    '''
    black  = (0, 0, 0)
    white  = (1, 1, 1)
    red    = (1, 0, 0)
    yellow = (1, 1, 0)
    green  = (0, 0.8, 0)
    blue   = (0, 0, 1)
    purple = (1, 0, 1)

    if has_background:
        colors = [black, white, red, yellow, green, blue]
    else:
        colors = [white, red, yellow, green, blue]

    return mpl.colors.LinearSegmentedColormap.from_list(
        name='elast', colors=colors, N=n_colors
    )


def get_color_kws(array, pct=99, scale=1.1):
    '''
    Get a dictionary of colormap arguments
    for visualizing the provided xarray.
    '''
    if array.name in {'sr', 'region', 'spatial_region'}:
            cmap = region_color_map(n_colors=6, has_background=True)
            return dict(cmap=cmap, vmin=-0.5, vmax=5.5)
    elif array.name in {'a', 'A', 'anat', 'anatomy', 'anatomic', 'mre_raw', 'dwi'} or array.name.startswith('t1') or array.name.startswith('t2'):
        cmap = grayscale_color_map()
        vmin = 0
        vmax = np.percentile(np.abs(array), pct) * scale
        return dict(cmap=cmap, vmin=vmin, vmax=vmax)
    elif array.name in {'mre', 'mu', 'Mu', 'elast', 'elastogram', 'baseline', 'mre', 'Mwave'}:
        cmap = wave_color_map(reverse=True)
        vmax = np.percentile(np.abs(array), pct) * scale #2e4
    elif array.name == 'compare':
        cmap = grayscale_color_map(symmetric=True)
        vmax = np.percentile(np.abs(array), pct) * scale #2e4
    elif array.name == 'mask':
        cmap = grayscale_color_map()
        return dict(cmap=cmap, vmin=0, vmax=1)
    else:
        cmap = wave_color_map()
        vmax = np.percentile(np.abs(array), pct) * scale
    return dict(cmap=cmap, vmax=vmax)


class Colorbar(matplotlib.colorbar.Colorbar):
    '''
    A colorbar in which the zero position is fixed
    during an interactive drag_pan operation. This
    makes operations like increasing/decreasing the
    "luminance" of the image more intuitive.
    '''
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
    A replacement for the plt.subplots function that enables
    more precise control over the figure size and layout.

    Instead of providing the figure size and having all other
    aspects determined relative to that, you provide the axes
    sizes and layout spacing, and the figure size is computed.

    Args:
        n_rows: int
        n_cols: int
        ax_height: float or iterable of floats
        ax_width: float or iterable of floats
        cbar_width: optional float
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
        return fig, axes, cbar_ax
    else:
        return fig, axes, None


def plot_line_1d(ax, a, resolution, **kwargs):
    if a.ndim == 2:
        n_x, n_hue = a.shape
    else:
        n_x, = a.shape
    x = np.arange(n_x) * resolution
    lines = ax.plot(x, a)
    ax.set_yscale(kwargs.get('yscale', 'linear'))
    ax.set_xscale(kwargs.get('xscale', 'linear'))
    ax.set_xlabel(kwargs.get('xlabel'))
    ax.set_ylabel(kwargs.get('ylabel'))
    ax.set_title(kwargs.get('title'))
    ax.set_ylim(kwargs.get('vmin', None), kwargs.get('vmax', None))
    return lines


def imshow(ax, a, resolution=1, **kwargs):
    if a.ndim == 2:
        n_x, n_y = a.shape
        a_T = a.T
    elif a.ndim == 3:
        n_x, n_y, n_c = a.shape
        a_T = np.transpose(a, (1, 0, 2))
    extent = (0, (n_x - 1) * resolution, 0, (n_y - 1) * resolution)
    return ax.imshow(a_T, origin='lower', extent=extent, **kwargs)


def plot_image_2d(
    ax, a, origin, resolution, xlabel=None, ylabel=None, title=None, **kwargs
):
    n_x, n_y = a.shape
    x0, y0 = origin
    x_res, y_res = resolution
    extent = (x0, x0 + n_x * x_res, y0, y0 + n_y * y_res)
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
