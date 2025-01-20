from numpy import subtract, triu_indices_from, log10, histogram2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_differenciation(magnitudes, times, log_dt=True):
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


def plot_dm_dt(dm_dt_hist, dm_edges, dt_edges, title=None, *kwargs):
    """Plots the provided histogram.

    Parameters
    ----------
    dm_dt_hist : numpy.ndarray
        The 2D image to be plotted.
    dm_edges : numpy.ndarray
        dm   bin edges
    dt_edges : numpy.ndarray
        dt bin edges
    """
    plt.imshow(dm_dt_hist, origin='lower', cmap='viridis', aspect='auto', extent=[dt_edges[0], dt_edges[-1], dm_edges[0], dm_edges[-1]], *kwargs)
    plt.colorbar(label=f'Count Intensity ({dm_dt_hist.min()}-{dm_dt_hist.max()})')
    
    if not title:
        title = "2D Binned Representation of $\Delta$t vs. $\Delta$mag"
    plt.title(title)
    plt.xlabel(r'$\log_{10} \; \Delta$t (days)')
    plt.ylabel(r'$\Delta$mag')

    plt.grid(False)

    plt.legend()
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