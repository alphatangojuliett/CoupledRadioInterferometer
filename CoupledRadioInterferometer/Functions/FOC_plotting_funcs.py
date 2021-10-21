import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import warnings 
import copy
import astropy.constants as c
import uvtools
import hera_pspec as ps

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

warnings.filterwarnings('ignore')


def fourier_transform_waterfalls_2(
    data,
    antpairpol=None,
    freqs=None,
    times=None,
    lsts=None,
    time_or_lst="lst",
    plot_units=None,
    data_units="Jy",
    mode="log",
    set_title=True,
    figsize=(12,5),
    dpi=100,
    aspect="auto",
    draw_colorbar=True,
    cmap="best",
    fontsize=14,
    tickfontsize=13,
    dynamic_range=None,
    plot_range=None,
    plot_limits=None,
    freq_taper=None,
    freq_taper_kwargs=None,
    time_taper=None,
    time_taper_kwargs=None,
):
    """
    Plot a 1x2 grid of waterfalls showing delay waterfall and delay vs. fringe.

    Moving clockwise from the top-left, the plots are as follows:
        time vs frequency
        fringe-rate vs frequency
        fringe-rate vs delay
        time vs delay

    Parameters
    ----------
    data: array-like of complex, or :class:`pyuvdata.UVData` instance
        Object containing visibility data. If an array is passed, then ``freqs``
        and either ``times`` or ``lsts`` must be provided, and the array must
        have shape (``lsts.size``, ``freqs.size``). Otherwise, an ``antpairpol``
        key must be provided.
    antpairpol: tuple
        (ant1, ant2, pol) tuple specifying the baseline and polarization to
        pull data for if ``data`` is a :class:`pyuvdata.UVData` instance. Ignored
        if ``data`` is an array-like object.
    freqs: array-like of float
        Frequencies corresponding to the observed data, in Hz. Required if ``data``
        is an array-like object; ignored otherwise.
    times: array-like of float
        Observation times, in JD. Required if ``data`` is an array-like object and
        ``lsts`` is not provided.
    lsts: array-like of float
        Observed LSTs, in radians. Required if ``data`` is an array-like object
        and ``times`` is not provided.
    time_or_lst: str, optional
        Either "time" or "lst". Used to specify whether the time axis should be
        in JD or LST. If ``data`` is an array-like object and only one of ``times``
        or ``lsts`` is provided, then this parameter is ignored.
    plot_units: dict, optional
        Dictionary mapping axis dimension to plotting units. Keys must come from
        ("lst", "time", "freq", "fringe-rate", "delay"); values must have supported
        conversion methods in ``astropy``. LST units may be specified either as
        radian-equivalent units or day-equivalent units. Default is:
            {
                "lst": "hour",
                "time": "day",
                "freq": "MHz",
                "fringe-rate": "mHz",
                "delay": "ns"
            }
    data_units: str, optional
        Units for the provided data. If ``data`` is a :class:`pyuvdata.UVData`
        instance, then these units are pulled from the object, if they are defined
        in it (accessed via the ``vis_units`` attribute). Default is to assume the
        data units are in Jy.
    mode: str, optional
        Plotting mode to use; must be one of ("log", "phs", "abs", "real", "imag").
        Default is "log", which plots the base-10 logarithm of the absolute value
        of the data. See :func:`data_mode` documentation for details.
    set_title: bool or str, optional
        Whether to set a title for the figure. If a string is passed, then that
        string is used for the figure title. Default is to use the provided
        ``antpairpol`` as the title.
    figsize: tuple of float, optional
        Size of the figure to be produced, in inches. Default is 14x10.
    dpi: float, optional
        Dots-per-inch of the figure. Default is 100.
    aspect: float or str, optional
        Aspect ratio to use for each subplot. Default is "auto".
    fontsize: float, optional
        Font size to use for plotting labels, in points.
    cmap: str or :class:`plt.cm.colors.Colormap` instance
        Color map to be used when drawing waterfalls. Default is to have the choice
        be based on the data mode selected: if ``mode`` is "phs", then "twilight"
        is used; otherwise, "inferno" is used.
    dynamic_range: dict, optional
        Dictionary mapping strings to number of orders-of-magnitude to restrict
        the plot dynamic range to. Accepted strings are as follows:
            ("time", "freq", "delay", "fringe-rate"): specifying one of these
            will limit the dynamic range for the associated row or column. For
            example, passing {"time": 5} will limit the dynamic range of the left
            column to five orders-of-magnitude, clipping values on the low-end.

            Any length-2 combination of an entry from the following pairs:
                ("time", "fringe-rate"), ("freq", "delay")
            This type of mapping will limit the dynamic range for a single plot
            with axes specified by the pair specified. For example, passing
            {("fringe-rate", "delay"): 5} will only limit the dynamic range for
            the bottom-right plot.
    plot_range: dict, optional
        Dictionary mapping same string options as dynamic_range to vmin/vmax.
        Only use plot_range for length-2 combinations and only when dynamic_range is None.
    plot_limits: dict, optional
        Dictionary mapping strings to length-2 tuples. The keys designate the
        dimension ("time", "freq", "fringe-rate", "delay") to crop, and the values
        give the lower- and upper-bounds of the cropped region. For example, passing
        {"delay": (-500, 500)} will crop the delay axis to only show delays between
        -500 ns and +500 ns (assuming delays are plotted in ns). The values passed
        must be in the same units as the plot units used; see the description of
        the ``plot_units`` parameter for details on default units.
    freq_taper: str, optional
        Name of the taper to be applied along the frequency-axis when performing
        Fourier transforms. Must be a taper supported by :func:`dspec.gen_window`.
        Default is no taper (an implicit top-hat or boxcar).
    freq_taper_kwargs: dict, optional
        Keyword arguments to be used in generating the frequency taper.
    time_taper: str, optional
        Name of the taper to be applied along the time-axis when performing Fourier
        transforms. Default is the same as for the frequency taper.
    time_taper_kwargs: dict, optional
        Keyword arguments to be used in generating the time taper.
    draw_colorbar: for labelled_waterfall functionality, to draw or not draw colorbar.

    Returns
    -------
    fig: :class:`plt.Figure` instance
        Figure containing 2x2 grid of plots visualizing the data in the selected
        mode for all possible Fourier transforms, with axis labels and colorbars.
    """
    import matplotlib.pyplot as plt

    # Convert potential None-types to empty dictionaries where needed.
    dynamic_range = dynamic_range or {}
    plot_limits = plot_limits or {}

    # Figure setup
    fig = plt.figure(figsize=figsize, dpi=dpi) 
    axes = fig.subplots(1,2)
    transform_axes = (1, -1)
    axes_dims = (
        ("delay", "time"),
        ("delay", "fringe-rate")
    )

    # Make the plots.
    for i, ax in enumerate(axes.ravel()):
        # Determine any adjustments to be made to axes in plotting routine.
        x_dim, y_dim = axes_dims[i]
        possible_drng_keys = (x_dim, y_dim, (x_dim, y_dim), (y_dim, x_dim))
        transform_axis = transform_axes[i]
        limit_dynamic_range = list(
            key in dynamic_range.keys()
            for key in possible_drng_keys
        )
        limit_range = list( key in plot_range.keys() for key in possible_drng_keys) #ATJ for vmin/vmax control
        if any(limit_dynamic_range):
            drng = dynamic_range[possible_drng_keys[limit_dynamic_range.index(True)]]
        else:
            drng = None
        if any(limit_range): #ATJ for vmin/vmax control
            rng = plot_range[possible_drng_keys[limit_range.index(True)]]
            vmin=rng[0]
            vmax=rng[1]
            #print('for ax = '+str(ax)+', (vmin, vmax) = ('+str(vmin)+', '+str(vmax)+')')
        else:
            vmin=None
            vmax=None
        # Adjust the plot boundaries if requested.
        if x_dim in plot_limits:
            ax.set_xlim(*plot_limits[x_dim])
        if y_dim in plot_limits:
            ax.set_ylim(*plot_limits[y_dim])
        # Actually make the plot.
        ax = uvtools.plot.labeled_waterfall(
            data=data,
            antpairpol=antpairpol,
            freqs=freqs,
            times=times,
            lsts=lsts,
            time_or_lst=time_or_lst,
            plot_units=plot_units,
            data_units=data_units,
            mode=mode,
            set_title=False,
            ax=ax,
            aspect=aspect,
            fontsize=fontsize,
            tickfontsize=tickfontsize,
            draw_colorbar=draw_colorbar,
            cmap=cmap,
            dynamic_range=drng,
            fft_axis=transform_axis,
            freq_taper=freq_taper,
            freq_taper_kwargs=freq_taper_kwargs,
            time_taper=time_taper,
            time_taper_kwargs=time_taper_kwargs,
            vmin=vmin,
            vmax=vmax
        )[1]

    # Set a figure title if desired.
    if set_title:
        if type(set_title) is bool:
            set_title = antpairpol
        if set_title is not None:
            # Though complicated, this ensures that the figure title is positioned well.
            axes = fig.get_axes()
            uppermost_y = max(ax.get_position().y1 for ax in axes)
            top_row = [
                ax for ax in axes
                if np.isclose(ax.get_position().y1, uppermost_y)
            ]
            axes_widths = [
                ax.get_position().x1 - ax.get_position().x0
                for ax in top_row
            ]
            colorbars = [
                ax for ax, width in zip(top_row, axes_widths)
                if not np.isclose(width, max(axes_widths))
            ]
            plots = [ax for ax in top_row if ax not in colorbars]


            # Find the visual horizontal center of the figure.
            x1 = min(cbar.get_position().x1 for cbar in colorbars)
            x2 = max(plot.get_position().x0 for plot in plots)
            title_position = (0.5 * (x1 + x2), 1.05*uppermost_y)

            # Position the title at the apparent "top center" of the figure.
            fig.text(
                *title_position,
                set_title,
                ha="center",
                va="bottom",
                fontsize=fontsize
            )


    return fig

def plot_fringe_vs_delay(
    data,
    ax,
    antpairpol=None,
    freqs=None,
    times=None,
    lsts=None,
    time_or_lst="lst",
    plot_units=None,
    data_units="Jy",
    mode="log",
    set_title=True,
    figsize=(12,5),
    dpi=100,
    aspect="auto",
    draw_colorbar=True,
    remove_colorbar=True,
    cbar_labelpad =None,
    cbar_pad=None,
    cmap="best",
    fontsize=14,
    tickfontsize=13,
    dynamic_range=None,
    plot_range=None,
    plot_limits=None,
    freq_taper=None,
    freq_taper_kwargs=None,
    time_taper=None,
    time_taper_kwargs=None,
):

    """
    Plot a grid of waterfalls showing delay waterfall and delay vs. fringe.

    Moving clockwise from the top-left, the plots are as follows:
        time vs frequency
        fringe-rate vs frequency
        fringe-rate vs delay
        time vs delay

    Parameters
    ----------
    data: array-like of complex, or :class:`pyuvdata.UVData` instance
        Object containing visibility data. If an array is passed, then ``freqs``
        and either ``times`` or ``lsts`` must be provided, and the array must
        have shape (``lsts.size``, ``freqs.size``). Otherwise, an ``antpairpol``
        key must be provided.
    ax: Matplotlib axis object
        Passing the function an axis will allow the generated plot to be incorporated into a larger matplotlib figure
    antpairpol: tuple
        (ant1, ant2, pol) tuple specifying the baseline and polarization to
        pull data for if ``data`` is a :class:`pyuvdata.UVData` instance. Ignored
        if ``data`` is an array-like object.
    freqs: array-like of float
        Frequencies corresponding to the observed data, in Hz. Required if ``data``
        is an array-like object; ignored otherwise.
    times: array-like of float
        Observation times, in JD. Required if ``data`` is an array-like object and
        ``lsts`` is not provided.
    lsts: array-like of float
        Observed LSTs, in radians. Required if ``data`` is an array-like object
        and ``times`` is not provided.
    time_or_lst: str, optional
        Either "time" or "lst". Used to specify whether the time axis should be
        in JD or LST. If ``data`` is an array-like object and only one of ``times``
        or ``lsts`` is provided, then this parameter is ignored.
    plot_units: dict, optional
        Dictionary mapping axis dimension to plotting units. Keys must come from
        ("lst", "time", "freq", "fringe-rate", "delay"); values must have supported
        conversion methods in ``astropy``. LST units may be specified either as
        radian-equivalent units or day-equivalent units. Default is:
            {
                "lst": "hour",
                "time": "day",
                "freq": "MHz",
                "fringe-rate": "mHz",
                "delay": "ns"
            }
    data_units: str, optional
        Units for the provided data. If ``data`` is a :class:`pyuvdata.UVData`
        instance, then these units are pulled from the object, if they are defined
        in it (accessed via the ``vis_units`` attribute). Default is to assume the
        data units are in Jy.
    mode: str, optional
        Plotting mode to use; must be one of ("log", "phs", "abs", "real", "imag").
        Default is "log", which plots the base-10 logarithm of the absolute value
        of the data. See :func:`data_mode` documentation for details.
    set_title: bool or str, optional
        Whether to set a title for the figure. If a string is passed, then that
        string is used for the figure title. Default is to use the provided
        ``antpairpol`` as the title.
    figsize: tuple of float, optional
        Size of the figure to be produced, in inches. Default is 14x10.
    dpi: float, optional
        Dots-per-inch of the figure. Default is 100.
    aspect: float or str, optional
        Aspect ratio to use for each subplot. Default is "auto".
    fontsize: float, optional
        Font size to use for plotting labels, in points.
    tickfontsize: float, optional
        Font size to use for plotting ticks in figures, in points.
    cmap: str or :class:`plt.cm.colors.Colormap` instance
        Color map to be used when drawing waterfalls. Default is to have the choice
        be based on the data mode selected: if ``mode`` is "phs", then "twilight"
        is used; otherwise, "inferno" is used.
    dynamic_range: dict, optional
        Dictionary mapping strings to number of orders-of-magnitude to restrict
        the plot dynamic range to. Accepted strings are as follows:
            ("time", "freq", "delay", "fringe-rate"): specifying one of these
            will limit the dynamic range for the associated row or column. For
            example, passing {"time": 5} will limit the dynamic range of the left
            column to five orders-of-magnitude, clipping values on the low-end.

            Any length-2 combination of an entry from the following pairs:
                ("time", "fringe-rate"), ("freq", "delay")
            This type of mapping will limit the dynamic range for a single plot
            with axes specified by the pair specified. For example, passing
            {("fringe-rate", "delay"): 5} will only limit the dynamic range for
            the bottom-right plot.
    plot_range: dict, optional
        Dictionary mapping same string options as dynamic_range to vmin/vmax.
        Only use plot_range for length-2 combinations and only when dynamic_range is None.
    plot_limits: dict, optional
        Dictionary mapping strings to length-2 tuples. The keys designate the
        dimension ("time", "freq", "fringe-rate", "delay") to crop, and the values
        give the lower- and upper-bounds of the cropped region. For example, passing
        {"delay": (-500, 500)} will crop the delay axis to only show delays between
        -500 ns and +500 ns (assuming delays are plotted in ns). The values passed
        must be in the same units as the plot units used; see the description of
        the ``plot_units`` parameter for details on default units.
    freq_taper: str, optional
        Name of the taper to be applied along the frequency-axis when performing
        Fourier transforms. Must be a taper supported by :func:`dspec.gen_window`.
        Default is no taper (an implicit top-hat or boxcar).
    freq_taper_kwargs: dict, optional
        Keyword arguments to be used in generating the frequency taper.
    time_taper: str, optional
        Name of the taper to be applied along the time-axis when performing Fourier
        transforms. Default is the same as for the frequency taper.
    time_taper_kwargs: dict, optional
        Keyword arguments to be used in generating the time taper.
    draw_colorbar: for labelled_waterfall functionality, to draw or not draw colorbar.

    Returns
    -------
    ax: :class:`matplotlib.axes.Axes` instance
        containing 2x2 grid of plots visualizing the data in the selected
        mode for all possible Fourier transforms, with axis labels and colorbars.
    """
    # Convert potential None-types to empty dictionaries where needed.
    dynamic_range = dynamic_range or {}
    plot_limits = plot_limits or {}

    x_dim, y_dim = ("delay", "fringe-rate")
    possible_drng_keys = (x_dim, y_dim, (x_dim, y_dim), (y_dim, x_dim))
    limit_range = list( key in plot_range.keys() for key in possible_drng_keys) #ATJ for vmin/vmax control
    limit_dynamic_range = list(key in dynamic_range.keys() for key in possible_drng_keys)
    if any(limit_dynamic_range):
        drng = dynamic_range[possible_drng_keys[limit_dynamic_range.index(True)]]
    else:
        drng = None
    if any(limit_range): #ATJ for vmin/vmax control
        rng = plot_range[possible_drng_keys[limit_range.index(True)]]
        vmin=rng[0]
        vmax=rng[1]
        #print('for ax = '+str(ax)+', (vmin, vmax) = ('+str(vmin)+', '+str(vmax)+')')
    else:
        vmin=None
        vmax=None
    # Adjust the plot boundaries if requested.
    if x_dim in plot_limits:
        ax.set_xlim(*plot_limits[x_dim])
    if y_dim in plot_limits:
        ax.set_ylim(*plot_limits[y_dim])
    ax = uvtools.plot.labeled_waterfall(
            data=data,
            antpairpol=antpairpol,
            freqs=freqs,
            times=times,
            lsts=lsts,
            time_or_lst=time_or_lst,
            plot_units=plot_units,
            data_units=data_units,
            mode=mode,
            set_title=False,
            ax=ax,
            aspect=aspect,
            fontsize=fontsize,
            tickfontsize=tickfontsize,
            draw_colorbar=draw_colorbar,
            remove_colorbar=remove_colorbar,
            cbar_labelpad=cbar_labelpad,
            cbar_pad=cbar_pad,
            cmap=cmap,
            dynamic_range=drng,
            fft_axis=-1,
            freq_taper=freq_taper,
            freq_taper_kwargs=freq_taper_kwargs,
            time_taper=time_taper,
            time_taper_kwargs=time_taper_kwargs,
            vmin=vmin,
            vmax=vmax
        )[1]
    return ax

def delay_wedge(uvp, spw, pol, blpairs=None, times=None, error_weights=None, fold=False, delay=True,
                rotate=False, component='abs-real', log10=True, loglog=False,
                red_tol=1.0, center_line=False, horizon_lines=False,
                title=None, ax=None, cmap='viridis', figsize=(8, 6),
                deltasq=False, colorbar=False, cbax=None, vmin=None, vmax=None,
                edgecolor='none', flip_xax=False, flip_yax=False, lw=2, 
                set_bl_tick_major=False, set_bl_tick_minor=False, 
                xtick_size=10, xtick_rot=0, ytick_size=10, ytick_rot=0, cbar_label_size=14,cbar_title_size=14, cbar_tick_size=14, title_size=14,xlabel_size=14,ylabel_size=14,
                **kwargs):
    """
    Plot a 2D delay spectrum (or spectra) from a UVPSpec object. Note that
    all integrations and redundant baselines are averaged (unless specifying 
    times) before plotting.
    Note: this deepcopies input uvp before averaging.
    
    Parameters
    ----------
    uvp : UVPSpec
        UVPSpec object containing delay spectra to plot.
    spw : integer
        Which spectral window to plot.
    pol : int or tuple
        Polarization-pair integer or tuple, e.g. ('pI', 'pI')
    blpairs : list of tuples, optional
        List of baseline-pair tuples to use in plotting.
    times : list, optional
        An ndarray or list of times from uvp.time_avg_array to
        select on before plotting. Default: None.
    error_weights : string, optional
         error_weights specify which kind of errors we use for weights 
         during averaging power spectra.
    fold : bool, optional
        Whether to fold the power spectrum in k_parallel. 
        Default: False.
    delay : bool, optional
        Whether to plot the axes in tau (ns). If False, axes will
        be plotted in cosmological units.
        Default: True.
    rotate : bool, optional
        If False, use baseline-type as x-axis and delay as y-axis,
        else use baseline-type as y-axis and delay as x-axis.
        Default: False
    component : str, optional
        Component of complex spectra to plot. Options=['real', 'imag', 'abs', 'abs-real', 'abs-imag']
        abs-real is abs(real(data)), whereas 'real' is real(data)
        Default: 'abs-real'.
    log10 : bool, optional
        If True, take log10 of data before plotting. Default: True
    loglog : bool, optional
        If True, turn x-axis and y-axis into log-log scale. Default: False
    red_tol : float, optional
        Redundancy tolerance when solving for redundant groups in meters.
        Default: 1.0
    center_line : bool, optional
        Whether to plot a dotted line at k_perp = 0.
        Default: False.
    horizon_lines : bool, optional
        Whether to plot dotted lines along the horizon.
        Default: False.
    title : string, optional
        Title for subplot.  Default: None.
    ax : matplotlib.axes, optional
        If not None, use this axes as a subplot for delay wedge.
    cmap : str, optional
        Colormap of wedge plot. Default: 'viridis'
    figsize : len-2 integer tuple, optional
        If ax is None, this is the new figure size.
    deltasq : bool, optional
        Convert to Delta^2 before plotting. This is ignored if delay=True.
        Default: False
    colorbar : bool, optional
        Add a colorbar to the plot. Default: False
    cbax : matplotlib.axes, optional
        Axis object for adding colorbar if True. Default: None
    vmin : float, optional
        Minimum range of colorscale. Default: None
    vmax : float, optional
        Maximum range of colorscale. Default: None
    edgecolor : str, optional
        Edgecolor of bins in pcolormesh. Default: 'none'
    flip_xax : bool, optional
        Flip xaxis if True. Default: False
    flip_yax : bool, optional
        Flip yaxis if True. Default: False
    lw : int, optional
        Line-width of horizon and center lines if plotted. Default: 2.
    set_bl_tick_major : bool, optional
        If True, use the baseline lengths as major ticks, rather than default 
        uniform grid.
    set_bl_tick_minor : bool, optional
        If True, use the baseline lengths as minor ticks, which have no labels.
    kwargs : dictionary
        Additional keyword arguments to pass to pcolormesh() call.
    Returns
    -------
    fig : matplotlib.pyplot.Figure
        Matplotlib Figure instance if ax is None.
    """


    # type checking
    uvp = copy.deepcopy(uvp)
    assert isinstance(uvp, ps.uvpspec.UVPSpec), "input uvp must be a UVPSpec object"
    assert isinstance(spw, (int, np.integer))
    assert isinstance(pol, (int, np.integer, tuple))
    fix_negval = component in ['real', 'imag'] and log10

    # check pspec units for little h
    little_h = 'h^-3' in uvp.norm_units

    # Create new ax if none specified
    new_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        new_plot = True
    else:
        fig = ax.get_figure()

    # Select out times and blpairs if provided
    if times is not None:
        uvp.select(blpairs=blpairs, times=times, inplace=True)

    # Average across redundant groups and time
    # this also ensures blpairs are ordered from short_bl --> long_bl
    blp_grps, lens, angs, tags = ps.utils.get_blvec_reds(uvp, bl_error_tol=red_tol, 
                                                      match_bl_lens=True)
    uvp.average_spectra(blpair_groups=blp_grps, time_avg=True, error_weights=error_weights, inplace=True)

    # get blpairs and order by len and enforce bl len ordering anyways
    blpairs, blpair_seps = uvp.get_blpairs(), uvp.get_blpair_seps()
    osort = np.argsort(blpair_seps)
    blpairs, blpair_seps = [blpairs[oi] for oi in osort], blpair_seps[osort]

    # Convert to DeltaSq
    if deltasq and not delay:
        uvp.convert_to_deltasq(inplace=True)

    # Fold array
    if fold:
        uvp.fold_spectra()

    # Format ticks
    if delay:
        x_axis = uvp.get_dlys(spw) * 1e9
        y_axis = blpair_seps
    else:
        x_axis = uvp.get_kparas(spw, little_h=little_h)
        y_axis = uvp.get_kperps(spw, little_h=little_h)
    if rotate:
        _x_axis = y_axis
        y_axis = x_axis
        x_axis = _x_axis

    # Conigure Units
    psunits = "({})^2\ {}".format(uvp.vis_units, uvp.norm_units)
    if "h^-1" in psunits: psunits = psunits.replace("h^-1", "h^{-1}\ ")
    if "h^-3" in psunits: psunits = psunits.replace("h^-3", "h^{-3}\ ")
    if "Hz" in psunits: psunits = psunits.replace("Hz", r"{\rm Hz}\ ")
    if "str" in psunits: psunits = psunits.replace("str", r"\,{\rm str}\ ")
    if "Mpc" in psunits and "\\rm" not in psunits: 
        psunits = psunits.replace("Mpc", r"{\rm Mpc}")
    if "pi" in psunits and "\\pi" not in psunits: 
        psunits = psunits.replace("pi", r"\pi")
    if "beam normalization not specified" in psunits:
        psunits = psunits.replace("beam normalization not specified", 
                                 r"{\rm unnormed}")

    # get data with shape (Nblpairs, Ndlys)
    data = [uvp.get_data((spw, blp, pol)).squeeze() for blp in blpairs]

    # get component
    if component == 'real':
        data = np.real(data)
    elif component == 'abs-real':
        data = np.abs(np.real(data))
    elif component == 'imag':
        data = np.imag(data)
    elif component == 'abs-imag':
        data = np.abs(np.imag(data))
    elif component == 'abs':
        data = np.abs(data)
    else:
        raise ValueError("Did not understand component {}".format(component))

    # if real or imag and log is True, set negative values to near zero
    # this is done so that one can use cmap.set_under() and cmap.set_bad() separately
    if fix_negval:
        data[data < 0] = np.abs(data).min() * 1e-6 + 1e-10

    # take log10
    if log10:
        data = np.log10(data)

    # loglog
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # rotate
    if rotate:
        data = np.rot90(data[:, ::-1], k=1)

    # Get bin edges
    xdiff = np.diff(x_axis)
    x_edges = np.array([x_axis[0]-xdiff[0]/2.0] + list(x_axis[:-1]+xdiff/2.0) + [x_axis[-1]+xdiff[-1]/2.0])
    ydiff = np.diff(y_axis)
    y_edges = np.array([y_axis[0]-ydiff[0]/2.0] + list(y_axis[:-1]+ydiff/2.0) + [y_axis[-1]+ydiff[-1]/2.0])
    X, Y = np.meshgrid(x_edges, y_edges)

    # plot 
    cax = ax.pcolormesh(X, Y, data, cmap=cmap, edgecolor=edgecolor, lw=0.01,
                        vmin=vmin, vmax=vmax, **kwargs)

    # Configure ticks
    if set_bl_tick_major:
        if rotate:
            ax.set_xticks([np.around(x, _get_sigfig(x)+2) for x in x_axis])
        else:
            ax.set_yticks([np.around(x, _get_sigfig(x)+2) for x in y_axis])
    if set_bl_tick_minor:
        if rotate:
            ax.set_xticks([np.around(x, _get_sigfig(x)+2) for x in x_axis], 
                          minor=True)
        else:
            ax.set_yticks([np.around(x, _get_sigfig(x)+2) for x in y_axis], 
                          minor=True)

    # Add colorbar
    if colorbar:
        if fix_negval:
            cb_extend = 'min'
        else:
            cb_extend = 'neither'
        if cbax is None:
            cbax = ax
        cbar = fig.colorbar(cax, ax=cbax, extend=cb_extend)
        if deltasq:
            p = "\Delta^2"
        else:
            p = "P"
        if delay:
            p = "{}({},\ {})".format(p, r'\tau', r'|\vec{b}|')
        else:
            p = "{}({},\ {})".format(p, r'k_\parallel', r'k_\perp')
        if log10:
            psunits = r"$\log_{{10}}\ {}\ [{}]$".format(p, psunits)
        else:
            psunits = r"${}\ [{}]$".format(p, psunits)
        cbar.set_label(psunits, fontsize=cbar_label_size)
        cbar.ax.tick_params(labelsize=cbar_tick_size)
        if fix_negval:
            cbar.ax.set_title("$< 0$",y=-0.05, fontsize=cbar_title_size)

    # Configure tick labels
    if delay:
        xlabel = r"$\tau$ $[{\rm ns}]$"
        ylabel = r"$|\vec{b}|$ $[{\rm m}]$"
    else:
        xlabel = r"$k_{\parallel}\ [h\ \rm Mpc^{-1}]$"
        ylabel = r"$k_{\perp}\ [h\ \rm Mpc^{-1}]$"
    if rotate:
        _xlabel = ylabel
        ylabel = xlabel
        xlabel = _xlabel
    if ax.get_xlabel() == '':
        ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if ax.get_ylabel() == '':
        ax.set_ylabel(ylabel, fontsize=ylabel_size)

    # Configure center line
    if center_line:
        if rotate:
            ax.axhline(y=0, color='#000000', ls='--', lw=lw)
        else:
            ax.axvline(x=0, color='#000000', ls='--', lw=lw)

    # Plot horizons
    if horizon_lines:
        # get horizon in ns
        horizons = blpair_seps / ps.conversions.units.c * 1e9

        # convert to cosmological wave vector
        if not delay:
            # Get average redshift of spw
            avg_z = uvp.cosmo.f2z(np.mean(uvp.freq_array[uvp.spw_to_freq_indices(spw)]))
            horizons *= uvp.cosmo.tau_to_kpara(avg_z, little_h=little_h) / 1e9

        # iterate over bins and plot lines
        if rotate:
            bin_edges = x_edges
        else:
            bin_edges = y_edges
        for i, hor in enumerate(horizons):
            if rotate:
                ax.plot(bin_edges[i:i+2], [hor, hor], color='#ffffff', ls='--', lw=lw)
                if not uvp.folded:
                    ax.plot(bin_edges[i:i+2], [-hor, -hor], color='#ffffff', ls='--', lw=lw)
            else:
                ax.plot([hor, hor], bin_edges[i:i+2], color='#ffffff', ls='--', lw=lw)
                if not uvp.folded:
                    ax.plot([-hor, -hor], bin_edges[i:i+2], color='#ffffff', ls='--', lw=lw)

    # flip axes
    if flip_xax:
        fig.sca(ax)
        fig.gca().invert_xaxis()
    if flip_yax:
        fig.sca(ax)
        fig.gca().invert_yaxis()

    # add title
    if title is not None:
        ax.set_title(title, fontsize=title_size)

    # Configure tick sizes and rotation
    [tl.set_size(xtick_size) for tl in ax.get_xticklabels()]
    [tl.set_rotation(xtick_rot) for tl in ax.get_xticklabels()]
    [tl.set_size(ytick_size) for tl in ax.get_yticklabels()]
    [tl.set_rotation(ytick_rot) for tl in ax.get_yticklabels()]

    # return figure
    if new_plot:
        return fig
