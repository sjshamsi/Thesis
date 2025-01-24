from numpy import subtract, triu_indices_from, log10, histogram2d, linspace
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_differenciation(magnitudes, times, log_dt=False):
    """Returns the 'differenciation' when given an array of magnitudes and corresponding times.
    (According to Mahabal et. al. 2017)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        Magnitudes (or any other measure of brightness) corresponding to observation times.
    times : numpy.ndarray
        MJD (or any other sequential time format) values of observation times. Not intendid to be in log10.
    log_dt : bool, optional
        Will apply log10 to the dt array before returning, by default True.

    Returns
    -------
    touple of numpy.ndarray
        Returns (dm, dt) arrays.
    """
    dm = subtract.outer(magnitudes, magnitudes)  # M_i - M_j
    dt = subtract.outer(times, times) # t_i - j

    # Flatten the upper triangle of the matrices (excluding diagonal)
    upper_triangle_indices = triu_indices_from(dt, k=1)
    
    dm = dm[upper_triangle_indices] * -1
    dt = dt[upper_triangle_indices] * -1

    if log_dt:
        dt = log10(dt)
    return dm, dt

def get_2Dhistogram(dmagnitudes, dmagnitudes_bins, dtimes, dtimes_bins, normalise=False, scale_factor=255):
    """Returns a histogram of the d^n(m) and d^n(t) pairs (n can be > 1) with provided bins.
    Histogram is inverted before return.

    Parameters
    ----------
    dmagnitudes : numpy.ndarray
        Will be binned along the y-axis
    dmagnitudes_bins : int, numpy.ndarray
        Will be passed directly to np.histogram2d.
    dtimes : numpy.ndarray
        Will be binned along the x-axis
    dtimes_bins : int, numpy.ndarray
        Will be passed directly to np.histogram2d.
    normalised : bool, optional
        Divide the histogram by the total pairs of points binned, by default False
    scale_factor : int, optional
        Scales the outputs from a 0-255 (ints). Only works when normalised=True, by default 255

    Returns
    -------
    touple of numpy.ndarray, numpy.ndarray, numpy.ndarray
        Returns the 2D histogram, dmagnitude bins, and dtime bins.
    """
    hist, _dm_edges, _dt_edges = histogram2d(dmagnitudes, dtimes, bins=[dmagnitudes_bins, dtimes_bins])

    if normalise:
        hist = hist / hist.sum()

    if scale_factor and normalise:
        hist = ((scale_factor * hist) + 0.99999).astype(int)

    return hist, _dm_edges, _dt_edges, 


def plot_dm_dt(dm_dt_hist, band, dm_bins, dt_bins, dm_nticks=10, dt_nticks=10, title=None):
    dm_indices = linspace(0, len(dm_bins) - 1, dm_nticks, dtype=int)
    dt_indices = linspace(0, len(dt_bins) - 1, dt_nticks, dtype=int)

    dm_ticks_labels = []
    for index in dm_indices:
        formatted_string = f"{dm_bins[index]:.2f}"
        dm_ticks_labels.append(formatted_string)

    dt_ticks_labels = []
    for index in dt_indices:
        formatted_string = f"{dt_bins[index]:.1e}"
        base, exponent = formatted_string.split('e')
        dt_ticks_labels.append(f"${base} \\times 10^{{{int(exponent)}}}$")

    if len(band) == 2:
        fig_size = (12, 8)
        alpha = 0.5
    else:
        fig_size = (10, 8)
        alpha = 1

    fig, ax = plt.subplots(figsize=fig_size)

    if 'r' in band:
        r_ch = ax.imshow(dm_dt_hist[:, :, 0], origin='lower', cmap='Reds', aspect='auto', extent=[0, len(dt_bins)-1, 0, len(dm_bins) - 1], alpha=alpha)
        r_cbar = plt.colorbar(r_ch, ax=ax, shrink=0.8, label="$r$ band")
    if 'g' in band:
        g_ch = ax.imshow(dm_dt_hist[:, :, 1], origin='lower', cmap='Greens', aspect='auto', extent=[0, len(dt_bins)-1, 0, len(dm_bins) - 1], alpha=alpha)
        g_cbar = plt.colorbar(g_ch, ax=ax, shrink=0.8, label="$g$ band", location='left')

    plt.yticks(ticks=dm_indices, labels=dm_ticks_labels)
    plt.xticks(ticks=dt_indices, labels=dt_ticks_labels, size=7)

    plt.xlabel(f'{dt_bins.min():.2f} $< dt <$ {dt_bins.max():.2f} (days) {len(dt_bins) - 1} bins', size=12)
    plt.ylabel(f'{dm_bins.min():.2f} $< dm <$ {dm_bins.max():.2f} (magnitude) {len(dm_bins) - 1} bins', size=12)

    if title:
        fig.suptitle(title, size=12)

    plt.grid(False)
    fig.tight_layout()
    plt.show()


def plot_dm_dt_rgb(dm_dt_hist, dm_bins, dt_bins, dm_nticks=10, dt_nticks=10, title=None):
    dm_indices = linspace(0, len(dm_bins) - 1, dm_nticks, dtype=int)
    dt_indices = linspace(0, len(dt_bins) - 1, dt_nticks, dtype=int)

    dm_ticks_labels = []
    for index in dm_indices:
        formatted_string = f"{dm_bins[index]:.2f}"
        dm_ticks_labels.append(formatted_string)

    dt_ticks_labels = []
    for index in dt_indices:
        formatted_string = f"{dt_bins[index]:.1e}"
        base, exponent = formatted_string.split('e')
        dt_ticks_labels.append(f"${base} \\times 10^{{{int(exponent)}}}$")

    fig, ax = plt.subplots(figsize=(9, 8))

    dm_dt_hist[:, :, 0] = dm_dt_hist[:, :, 0] / dm_dt_hist[:, :, 0].max()
    dm_dt_hist[:, :, 1] = dm_dt_hist[:, :, 1] / dm_dt_hist[:, :, 1].max()

    ax.imshow(dm_dt_hist, origin='lower', aspect='auto', extent=[0, len(dt_bins)-1, 0, len(dm_bins) - 1])
    # cbar_red = plt.colorbar(plt.cm.ScalarMappable(cmap='Reds'), ax=ax, shrink=0.8, pad=0.000000000001)
    # cbar_red.set_label("Red Intensity", rotation=270, labelpad=15)

    # cbar_green = plt.colorbar(plt.cm.ScalarMappable(cmap='Greens'), ax=ax, shrink=0.8, pad=0.01)
    # cbar_green.set_label("Green Intensity", rotation=270, labelpad=15)

    plt.yticks(ticks=dm_indices, labels=dm_ticks_labels)
    plt.xticks(ticks=dt_indices, labels=dt_ticks_labels, size=7)

    plt.xlabel(f'{dt_bins.min():.2f} $< dt <$ {dt_bins.max():.2f} (days) {len(dt_bins) - 1} bins', size=12)
    plt.ylabel(f'{dm_bins.min():.2f} $< dm <$ {dm_bins.max():.2f} (magnitude) {len(dm_bins) - 1} bins', size=12)

    if title:
        fig.suptitle(title, size=12)

    plt.grid(False)
    fig.tight_layout()
    plt.show()
    

def save_dm_dt_image(dm_dt_hist, name, cmap=cm.viridis):
    """Saves a histogram image as it is shown in plt.imshow()

    Parameters
    ----------
    dm_dt_hist : numpy.ndarray
        The 2D histogram.
    name : str
        Name for the saved image
    cmap : matplotlib.colors.ListedColormap, optional
        Colourmap to be used, by default cm.viridis
    """
    image = cmap(dm_dt_hist)
    plt.imsave(f'{name}', image)