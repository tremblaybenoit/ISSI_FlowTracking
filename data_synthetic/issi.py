import os
from astropy.io import fits
import numpy as np
from typing import Union, Any
from numpy import floating
import scipy.stats as stats
from numpy import floating
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter,zoom
import re
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
from scipy.fft import fft2, ifft2, fftfreq
import glob
import os
import imageio.v3 as iio
import imageio
from PIL import ImageFont, ImageDraw, Image
from matplotlib.colors import LogNorm

def temporal_avg(data_in, start, end):
    tmp_avg = np.mean(data_in[start:end+1], axis=0)
    return tmp_avg

def load_muram_vars(path, tau0,iter1, iter2, muram, issi, fwhm=None, pixelsize=None):
    """
    Load and average Muram magnetic and velocity fields between iter1 and iter2,
    apply Gaussian filtering, and compute vertical Poynting flux. Assumes Muram slices
    are available every 50 iterations.

    Parameters:
        path : str
            Path to MURaM data.
        tau0 : float
            Optical depth level.
        iter1 : int
            Starting iteration index.
        iter2 : int
            Ending iteration index.
        muram : module
            Module to read MuramTauSlice.
        issi : module
            Module containing vertical_poynting_flux.
        fwhm : float, optional
            Full width at half maximum for Gaussian filter [same units as pixelsize].
        pixelsize : float, optional
            Pixel size in the same units as fwhm.

    Returns:
        dict containing:
            b_muram_x, b_muram_y, b_muram_z,
            v_muram_x, v_muram_y, v_muram_z,
            s_muram_z
    """
    step = 50  # Muram slices available every 50 iterations
    iter_indices = list(range(iter1, iter2 + 1, step))
 
    def load_slice(i):
        if tau0 in [1.0, 0.1]:
            return muram.MuramTauSlice(path, i, tau0)
        elif tau0 in [384, 400]:
            return muram.MuramSlice(path, i, 'yz','0'+str(tau0))
        else:
            raise ValueError("tau0 must be 1.0 or 0.1 (for MuramTauSlice) or 384 or 400 (for MuramXYSlice)")

    if len(iter_indices) == 1:
        #tau = muram.MuramTauSlice(path, iter_indices[0], tau0)
        tau = load_slice(iter_indices[0])
        b_muram_x = tau.By
        b_muram_y = tau.Bz
        b_muram_z = tau.Bx
        v_muram_x = tau.vz / 1e5  #in km
        v_muram_y = tau.vy / 1e5
        v_muram_z = tau.vx / 1e5
    else:
        bxs, bys, bzs = [], [], []
        vxs, vys, vzs = [], [], []

        for i in iter_indices:
            tau = load_slice(i)
            bxs.append(tau.By)
            bys.append(tau.Bz)
            bzs.append(tau.Bx)
            vxs.append(tau.vz / 1e5)
            vys.append(tau.vy / 1e5)
            vzs.append(tau.vx / 1e5)

        b_muram_x = temporal_avg(np.stack(bxs), 0, len(iter_indices)-1)
        b_muram_y = temporal_avg(np.stack(bys), 0, len(iter_indices)-1)
        b_muram_z = temporal_avg(np.stack(bzs), 0, len(iter_indices)-1)
        v_muram_x = temporal_avg(np.stack(vxs), 0, len(iter_indices)-1)
        v_muram_y = temporal_avg(np.stack(vys), 0, len(iter_indices)-1)
        v_muram_z = temporal_avg(np.stack(vzs), 0, len(iter_indices)-1)

    # Apply Gaussian filtering if fwhm and pixelsize are provided
    if fwhm is not None and pixelsize is not None:
        sigma = fwhm / 1.665 / pixelsize
        print('Sigma [pix] is ',sigma)
        v_muram_x = gaussian_filter(v_muram_x, sigma, mode="wrap")
        v_muram_y = gaussian_filter(v_muram_y, sigma, mode="wrap")
        v_muram_z = gaussian_filter(v_muram_z, sigma, mode="wrap")
        b_muram_x = gaussian_filter(b_muram_x, sigma, mode="wrap")
        b_muram_y = gaussian_filter(b_muram_y, sigma, mode="wrap")
        b_muram_z = gaussian_filter(b_muram_z, sigma, mode="wrap")

    s_muram_z, s_muram_z_emergence, s_muram_z_shear = issi.vertical_poynting_flux(
        v_muram_x, v_muram_y, v_muram_z,
        b_muram_x, b_muram_y, b_muram_z
    )

    if fwhm is not None and pixelsize is not None:
        s_muram_z = gaussian_filter(s_muram_z, sigma, mode="wrap")
        s_muram_z_emergence = gaussian_filter(s_muram_z_emergence, sigma, mode="wrap")
        s_muram_z_shear = gaussian_filter(s_muram_z_shear, sigma, mode="wrap")

    return {
        'b_muram_x': b_muram_x,
        'b_muram_y': b_muram_y,
        'b_muram_z': b_muram_z,
        'v_muram_x': v_muram_x,
        'v_muram_y': v_muram_y,
        'v_muram_z': v_muram_z,
        's_muram_z': s_muram_z,
        's_muram_z_emergence': s_muram_z_emergence,
        's_muram_z_shear': s_muram_z_shear,

    }

# Create MURAM video
def plot_muram_single_snapshot(muram_data, output_path="muram_snapshot.jpg", vminmax=None, iter_num=None):
    """
    Plot a single 3x3 grid of 9 MURaM variables and save as a JPG.

    Parameters:
    -----------
    muram_data : dict
        Dictionary containing 2D arrays for each variable with shape (ny, nx).
    output_path : str
        File path to save the output JPG image.
    vminmax : dict or None
        Optional dict specifying (vmin, vmax) per variable.
    iter_num : int or None
        Iteration number to include in title labels.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    variables = [
        'b_muram_x', 'b_muram_y', 'b_muram_z',
        'v_muram_x', 'v_muram_y', 'v_muram_z',
        'e_muram_x', 'e_muram_y', 'e_muram_z'
    ]

    units = {
        'b': '[G]',
        'v': '[km/s]',
        'e': '[V/cm]'
    }

    # Compute default vmin/vmax if not given
    if vminmax is None:
        vminmax = {}
        for var in variables:
            data = muram_data[var]
            vminmax[var] = (np.nanpercentile(data, 1), np.nanpercentile(data, 99))

    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    for idx, var in enumerate(variables):
        ax = axs[idx // 3, idx % 3]
        im = ax.imshow(muram_data[var], cmap='RdBu_r', origin='lower',
                       vmin=vminmax[var][0], vmax=vminmax[var][1])

        prefix = var.split("_")[0]  # 'b', 'v', or 'e'
        unit_label = units.get(prefix, '')
        label = f"{var} {unit_label}"
        if iter_num is not None:
            label += f", iter={iter_num}"
        ax.set_title(label)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return f"📸 Snapshot saved to {output_path}"

def save_single_snapshot_with_timestamp_local(ik, path, directory, tau0, fwhm, pixelsize, dt, vminmax):
    try:
        import issi
        import muram

        muram_data = issi.load_muram_vars_with_inductivity(
            path, tau0, ik, ik, muram, issi,
            fwhm=fwhm, pixelsize=pixelsize, dt_s=dt
        )

        outname = os.path.join(directory, "muram_frames", f"tauz_{tau0}_{ik:06d}_muram_snapshot.jpg")
        issi.plot_muram_single_snapshot(muram_data, output_path=outname, vminmax=vminmax,iter_num=ik)

        # Load image
        img = Image.open(outname).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Use a larger default font if available
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        label = f"Iteration {ik} (t = {ik * dt} s)"
        text_width, text_height = draw.textsize(label, font=font)

        # Draw background rectangle and text in upper-left corner
        padding = 10
        draw.rectangle([padding, padding, padding + text_width + 10, padding + text_height + 10], fill="black")
        draw.text((padding + 5, padding + 5), label, fill="white", font=font)

        img.save(outname)
        return f"✅ Annotated snapshot saved for iteration {ik}"

    except Exception as e:
        return f"❌ Failed for iteration {ik}: {e}"

def cr_muram_video_parallel(path, directory, tau0, fwhm, pixelsize, dt,
                                               vminmax=None, step=50, start=0, stop=11000,
                                               n_jobs=4, generate_images=True):
    """
    Parallel snapshot generation with in-function imports to avoid module pickling.
    Optionally only generates video from existing frames.

    Parameters:
    -----------
    generate_images : bool
        If True, generates annotated snapshot images. If False, only builds video from existing JPGs.
    """
    from multiprocessing import Pool
    from functools import partial

    os.makedirs(os.path.join(directory, "muram_frames"), exist_ok=True)
    ik_values = list(range(start, stop + 1, step))

    results = []

    if generate_images:
        with Pool(processes=n_jobs) as pool:
            func = partial(save_single_snapshot_with_timestamp_local,
                           path=path, directory=directory, tau0=tau0,
                           fwhm=fwhm, pixelsize=pixelsize, dt=dt,
                           vminmax=vminmax)
            results = pool.map(func, ik_values)
    else:
        print("🖼️ Skipping image generation. Only creating video from existing frames.")

    # Generate video regardless
    video_path = os.path.join(directory, f"muram_video_tauz_{tau0}.mp4")
    make_video_from_frames(image_dir=os.path.join(directory, "muram_frames"),
                           output_video=video_path, fps=5, iter1=start)
    return results, video_path


def make_video_from_frames(image_dir="muram_frames", output_video="muram_movie.mp4", fps=5, iter1=0):
    """
    Create an MP4 video using FFmpeg backend with correct format enforcement.

    Parameters:
    -----------
    image_dir : str
        Directory with JPG frames (e.g., tauz_*.jpg).
    output_video : str
        Path to save the MP4 video.
    fps : int
        Frames per second.
    iter1 : int
        Start iteration (optional, unused).
    """
    jpg_files = sorted(glob.glob(os.path.join(image_dir, "tauz_*.jpg")))
    if not jpg_files:
        print("❌ No JPG frames found.")
        return None

    # Explicitly force ffmpeg backend
    writer = imageio.get_writer(
        output_video,
        format='FFMPEG',
        mode='I',
        fps=fps,
        codec='libx264'
    )

    for filename in jpg_files:
        img = imageio.imread(filename)
        writer.append_data(img)

    writer.close()
    print(f"🎞️ Video saved to {output_video}")
    return output_video

# Define function to extract vx and vy variables from FITS file
def extract_masha_vxvy(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        vx = data[0].T
        vy = data[1].T

    return vx, vy

def add_scatter_subplot(x, y, subplot_index, label, lims=None):
    plt.subplot(3, 3, subplot_index)
    x_flat = x.flatten()
    y_flat = y.flatten()

    plt.scatter(x_flat, y_flat, s=1, alpha=0.3)

    # Compute metrics
    corr, _ = stats.spearmanr(x_flat, y_flat)
    rmse = mean_squared_error(x_flat, y_flat, squared=False)

    # Set limits
    if lims is not None:
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims, 'r--', linewidth=1)
    else:
        all_vals = np.concatenate([x_flat, y_flat])
        lims = [np.min(all_vals), np.max(all_vals)]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims, 'r--', linewidth=1)

    # Labels and title
    plt.xlabel('Muram')
    plt.ylabel('Recovered')
    plt.title(f'{label}\nCorr={corr:.2f}, RMSE={rmse:.2f}')
    plt.grid(True)

def add_density_subplot(x, y, subplot_index, label, lims=None, bins=100):
    plt.subplot(3, 3, subplot_index)

    # Flatten and remove NaNs
    x_flat = x.flatten()
    y_flat = y.flatten()
    mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]

    # Set limits
    if lims is not None:
        range_ = [[lims[0], lims[1]], [lims[0], lims[1]]]
    else:
        min_val = min(np.min(x_flat), np.min(y_flat))
        max_val = max(np.max(x_flat), np.max(y_flat))
        range_ = [[min_val, max_val], [min_val, max_val]]
        lims = [min_val, max_val]

    # Plot 2D histogram
    h = plt.hist2d(x_flat, y_flat, bins=bins, range=range_, cmap='plasma', norm='log')[3]
    plt.colorbar(h, label='Counts')

    # Overlay x=y line
    plt.plot(lims, lims, 'k--', linewidth=1)

    # Metrics
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
  

    corr, _ = stats.spearmanr(x_flat, y_flat)
    rmse = mean_squared_error(x_flat, y_flat, squared=False)
    
    # Linear fit
    slope, intercept = np.polyfit(x_flat, y_flat, 1)
    fit_line = slope * np.array(lims) + intercept
    plt.plot(lims, fit_line, color='orange',linestyle='--',linewidth=1.5, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    plt.legend(loc='upper left')

    issi_metrics = Metrics(v1x=x, v2x=y)

    # Labels and title
    plt.xlabel('Actual')
    plt.ylabel('Recovered')
    plt.title(f'{label}\nCorr={corr:.2f}, RMSE={rmse:.2f}, MEA={issi_metrics.mean_absolute_relative_error():.2f}, Energy ratio={issi_metrics.energy_ratio():.2f}')
    plt.grid(True)
    
# Define function to find FITS files
def find_fits_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.fits')]

def find_welsch_fits_files4iter(directory, iter_num, fwhm, dt, tau):
    """
    Find FITS files matching FLCT format with given fwhm, dt, tau, and iteration number.

    Parameters:
    -----------
    directory : str
        Path to directory with files.
    iter_num : int
        Iteration number to match (e.g. 150 → matches '000150').
    fwhm : int
        FWHM value to match in filename.
    dt : int
        Delta t value to match in filename.
    tau : float or str
        Tau value to match (e.g. 1.000).

    Returns:
    --------
    List[str] : Matching file names.
    """
    if tau == 0.1:
        subdir = "Tau_0.1"
        tau_str = f"{float(tau):.3f}" if isinstance(tau, (float, int)) else tau
    elif tau == 1.0:
        subdir = "Tau_1.0"
        tau_str = f"{float(tau):.3f}" if isinstance(tau, (float, int)) else tau
    elif tau == 384:
        subdir = "Yz_0384"
        tau_str =  '0'+f"{int(tau)}" if isinstance(tau, (float, int)) else tau
    elif tau == 400:
        subdir = "Yz_0400"
        tau_str = '0'+f"{int(tau)}" if isinstance(tau, (float, int)) else tau
    else:
        raise ValueError("tau0 must be one of 0.1, 1.0, 384, 400")


    iter_str = f"{iter_num:06d}.fits"
    #print('tau ',tau_str)
    return [
        subdir+'/'+f for f in os.listdir(directory+subdir)
        if f"fwhm{fwhm}" in f.lower()
        and f"dt{dt}" in f.lower()
        and f"_slice_{tau_str}" in f.lower()
        and f.endswith(f".{iter_str}")
    ]

#
def find_meetu_fits_files4iter(directory, iter_num, fwhm, dt, tau,iout_or_bz):
    """
    Find FITS files matching FLCT format with given fwhm, dt, tau, and iteration number.

    Parameters:
    -----------
    directory : str
        Path to directory with files.
    iter_num : int
        Iteration number to match (e.g. 150 → matches '000150').
    fwhm : int
        FWHM value to match in filename.
    dt : int
        Delta t value to match in filename.
    tau : float or str
        Tau value to match (e.g. 1.000).
    iout_or_bz : str
        Keyword in filename: 'Iout' or 'Bz' (case-insensitive).


    Returns:
    --------
    List[str] : Matching file names.
    """

    keyword = iout_or_bz.strip().lower()
    if keyword not in ['iout', 'bz']:
        raise ValueError("iout_or_bz must be 'Iout' or 'Bz'")

    if tau == 0.1:
        #subdir = "Tau_0.1"
        tau_str = f"{float(tau):.3f}" if isinstance(tau, (float, int)) else tau
    elif tau == 1.0:
        #subdir = "Tau_1.0"
        tau_str = f"{float(tau):.3f}" if isinstance(tau, (float, int)) else tau
    elif tau == 384:
        #subdir = "Yz_0384"
        tau_str =  '0'+f"{int(tau)}" if isinstance(tau, (float, int)) else tau
    elif tau == 400:
        #subdir = "Yz_0400"
        tau_str = '0'+f"{int(tau)}" if isinstance(tau, (float, int)) else tau
    else:
        raise ValueError("tau0 must be one of 0.1, 1.0, 384, 400")


    iter_str = f"{iter_num:06d}.fts"
    dt = f"{dt:04d}"
    
    fwhm = f"{fwhm:04d}"
    
    #print(directory,f"fwhm{fwhm}",f"dt{dt}",f"_slice_{tau_str}",f".{iter_str}")
    return [
        f for f in os.listdir(directory)
        if f"fwhm{fwhm}" in f.lower()
        and f"dt{dt}" in f.lower()
        and f"_slice_{tau_str}" in f.lower()
        and f.endswith(f".{iter_str}")
    ]

def find_jiayi_fits_files4iter(directory, iter_num, fwhm, dt, tau, method):
    """
    Find FITS files matching DAVE4VM/DAVE4VMWDV format with given fwhm, dt, tau, method, and iteration number.

    Parameters:
    -----------
    directory : str
        Path to directory with files.
    iter_num : int
        Iteration number to match (e.g. 150 → matches '000150').
    fwhm : int
        FWHM value to match in filename.
    dt : int
        Delta t value to match in filename.
    tau : float or str
        Tau value to match (e.g. 1.0 or '0.1').
    method : str
        Method name to match (e.g., "DAVE4VM" or "DAVE4VMWDV").

    Returns:
    --------
    List[str] : Matching file names.
    """
    print(f"Calling with: iter={iter_num}, fwhm={fwhm}, dt={dt}, tau={tau}, method={method}")
    
    if tau == 0.1:
        subdir = "Tau_0.1"
        tauzslice="tau_slice"
    elif tau == 1.0:
        subdir = "Tau_1.0"
        tauzslice="tau_slice"
    elif tau == 384:
        subdir = "YZ_384"
        tauzslice="z_slice"
    elif tau == 400:
        subdir = "YZ_400"
        tauzslice="z_slice"
    else:
        raise ValueError("tau0 must be one of 0.1, 1.0, 384, 400")

    
    iter_str = f"{int(iter_num):06d}"
    tau_str = str(float(tau)) if isinstance(tau, float) else str(int(tau))
    method = method.upper()

    # Match e.g., Liu_DAVE4VM_vx_vy_vz_fwhm0_dt30_yz_slice_384.001050.fits
    pattern = re.compile(
        rf".*{method}.*fwhm{fwhm}.*dt{dt}.*{tauzslice}_{re.escape(tau_str)}\.{iter_str}\.fits$",
        re.IGNORECASE
    )
    
    return [subdir+'/'+f for f in os.listdir(directory+subdir) if pattern.match(f)]

def find_masha_fits_files4iter(directory, iter_num, fwhm, dt, tau,method):
    """
    Find FITS files matching mkd_PDFI_ex_ey_fwhm100_dt10_tau_slice_1.000.000050.fits format with given fwhm, dt, tau, and iteration number.

    Parameters:
    -----------
    directory : str
        Path to directory with files.
    iter_num : int
        Iteration number to match (e.g. 150 → matches '000150').
    fwhm : int
        FWHM value to match in filename.
    dt : int
        Delta t value to match in filename.
    tau : float or str
        Tau value to match (e.g. 1.0 or '0.1').


    Returns:
    --------
    List[str] : Matching file names.
    """
    print(f"Calling with: iter={iter_num}, fwhm={fwhm}, dt={dt}, tau={tau}")
    
    if tau == 0.1:
        subdir = "Tau_0.1"
        tauzslice="tau_slice"
    elif tau == 1.0:
        subdir = "Tau_1.0"
        tauzslice="tau_slice"
    elif tau == 384:
        subdir = "YZ_384"
        tauzslice="yz_slice"
    elif tau == 400:
        subdir = "YZ_400"
        tauzslice="yz_slice"
    else:
        raise ValueError("tau0 must be one of 0.1, 1.0, 384, 400")


    iter_str = f"{int(iter_num):06d}"
    tau_str = str(float(tau))+'00' if isinstance(tau, float) else str(int(tau))
    method = method.upper()

    # Match e.g., Liu_DAVE4VM_vx_vy_vz_fwhm0_dt30_yz_slice_384.001050.fits
   # Match e.g., Liu_DAVE4VM_vx_vy_vz_fwhm0_dt30_yz_slice_384.001050.fits
    pattern = re.compile(
        rf".*{method}.*fwhm{fwhm}.*dt{dt}.*{tauzslice}_{re.escape(tau_str)}\.{iter_str}\.fits$",
        re.IGNORECASE)
    return [subdir+'/'+f for f in os.listdir(directory+subdir) if pattern.match(f)]
    
def find_teodor_fits_files4iter(directory, fwhm, dt, iout_or_bz,tau):
    """
    Find Teodor FLCT FITS files matching specific fwhm, dt, and keyword (Iout or Bz).

    Parameters:
    -----------
    directory : str
        Path to directory with FITS files.
    fwhm : int
        FWHM value to match in the filename.
    dt : int
        Time delta (dt) value to match.
    iout_or_bz : str
        Keyword in filename: 'Iout' or 'Bz' (case-insensitive).

    Returns:
    --------
    List[str] : Matching file names.
    """
    keyword = iout_or_bz.strip().lower()
    if keyword not in ['iout', 'bz']:
        raise ValueError("iout_or_bz must be 'Iout' or 'Bz'")

    keyword = keyword.capitalize()  # Make sure it matches 'Iout' or 'Bz' in filename

    if tau == 0.1:
        subdir = "TAU_0.1"
    elif tau == 1.0:
        subdir = "TAU_1.0"
    elif tau == 384:
        subdir = "Z_384"
    elif tau == 400:
        subdir = "Z_400"
    else:
        raise ValueError("tau0 must be one of 0.1, 1.0, 384, 400")

    matches = [
        subdir+'/'+f for f in os.listdir(directory+subdir)
        if f.startswith("Teodor_pyFLCT_vx_vy_")
        and f"fwhm_{fwhm}" in f
        and f"dt_{dt}" in f
        #and f"{keyword}_tracked.fits" in f
        and f.endswith(".fits")
    ]
    return matches


# Define function to extract vx and vy variables from FITS file
def extract_welsch_vxvy(file_path, fwhm=None, pixelsize=None):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        vx = data[0]
        vy = data[1]

    if fwhm is not None and pixelsize is not None:
        sigma = fwhm / 1.665 / pixelsize
        vx = gaussian_filter(vx, sigma, mode="wrap")
        vy = gaussian_filter(vy, sigma, mode="wrap")

    return vx, vy


# Define function to extract vx and vy variables from FITS file - same as welsch procedure
def extract_meetu_vxvy(file_path, fwhm=None, pixelsize=None):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        vx = data[0]
        vy = data[1]

    if fwhm is not None and pixelsize is not None:
        sigma = fwhm / 1.665 / pixelsize
        vx = gaussian_filter(vx, sigma, mode="wrap")
        vy = gaussian_filter(vy, sigma, mode="wrap")

    return vx, vy
# Define function to extract vx and vy variables from FITS file
def extract_teodor_vxvy(file_path,iter,dt,fwhm=None, pixelsize=None):
    with fits.open(file_path) as hdul:
        vx_cube = hdul[0].data  
        vy_cube = hdul[1].data  
    it=int(iter/dt/5-1)
    vx=vx_cube[it,:,:]
    vy=vy_cube[it,:,:]

    if fwhm is not None and pixelsize is not None:
        sigma = fwhm / 1.665 / pixelsize
        vx = gaussian_filter(vx, sigma, mode="wrap")
        vy = gaussian_filter(vy, sigma, mode="wrap")
    return vx, vy

# Define function to extract vx and vy variables from FITS file
def extract_jiayi_vxvyvz(file_path):
    with fits.open(file_path) as hdul:
        hdul.info()  # Shows the structure (optional)

        vx = hdul['VX'].data
        vy = hdul['VY'].data
        vz = hdul['VZ'].data

    return vx, vy, vz

#calculate curl E from Eh in V/cm and ds in km to derive G*km/s
def compute_precise_curl_Ez(Ex, Ey, dx):
    Ex=Ex*1e3
    Ey=Ey*1e3
    
    nx, ny = Ex.shape
    halfods = 0.5 / dx
    curl = np.zeros_like(Ex)

    curl[1:nx-1, 1:ny-1] = halfods * (
        Ey[2:nx, 1:ny-1] - Ey[0:nx-2, 1:ny-1] -
        Ex[1:nx-1, 2:ny] + Ex[1:nx-1, 0:ny-2]
    )

    for j in range(1, ny - 1):
        curl[0, j] = halfods * (-Ex[0, j+1] + Ex[0, j-1] + 4*Ey[1, j] - 3*Ey[0, j] - Ey[2, j])
        curl[-1, j] = halfods * (-Ex[-1, j+1] + Ex[-1, j-1] + 3*Ey[-1, j] + Ey[-3, j] - 4*Ey[-2, j])

    for i in range(1, nx - 1):
        curl[i, 0] = halfods * (Ey[i+1, 0] - Ey[i-1, 0] - 4*Ex[i, 1] + 3*Ex[i, 0] + Ex[i, 2])
        curl[i, -1] = halfods * (Ey[i+1, -1] - Ey[i-1, -1] - 3*Ex[i, -1] - Ex[i, -3] + 4*Ex[i, -2])

    curl[0, 0] = halfods * (4*Ey[1, 0] - 3*Ey[0, 0] - Ey[2, 0] - 4*Ex[0, 1] + 3*Ex[0, 0] + Ex[0, 2])
    curl[0, -1] = halfods * (4*Ey[1, -1] - 3*Ey[0, -1] - Ey[2, -1] - 3*Ex[0, -1] - Ex[0, -3] + 4*Ex[0, -2])
    curl[-1, 0] = halfods * (3*Ey[-1, 0] + Ey[-3, 0] - 4*Ey[-2, 0] - 4*Ex[-1, 1] + 3*Ex[-1, 0] + Ex[-1, 2])
    curl[-1, -1] = halfods * (3*Ey[-1, -1] + Ey[-3, -1] - 4*Ey[-2, -1] - 3*Ex[-1, -1] - Ex[-1, -3] + 4*Ex[-1, -2])

    return curl

#calculate muram variables in G, km/s and V/cm

# Re-import necessary modules after kernel reset
def load_muram_vars_with_inductivity(path, tau0, iter1, iter2, muram, issi,
                                     fwhm=None, pixelsize=None, dt_s=None,
                                     ds=None, average_over_time=False):
    """
    Loads MURaM data and computes E-fields, Poynting flux, curl(Ez), and dBz/dt.
    Allows optional spatial binning (ds) and optional averaging over time steps.

    Parameters:
    -----------
    path : str
        Path to MURaM data.
    tau0 : float or int
        Optical depth level or yz slice index.
    iter1, iter2 : int
        Iteration indices for time range.
    muram, issi : modules
        Objects with data loaders and utility functions.
    fwhm : float
        FWHM in km for optional Gaussian smoothing.
    pixelsize : float
        Pixel size in km.
    dt_s : float
        Time interval in seconds.
    ds : float
        Desired binning scale in km (must be >= pixelsize).
    average_over_time : bool
        If True, average over all time steps. If False, only use first and last step.

    Returns:
    --------
    dict of np.ndarray
        Processed field maps: B, V, E, Sz, curl_Ez, dBz_dt.
    """
    assert dt_s is not None, "dt_s (in seconds) must be specified"
    from issi import calc_e_field_from_v_and_b, compute_precise_curl_Ez, vertical_poynting_flux
    from scipy.ndimage import gaussian_filter, zoom

    def load_slice(i):
        if tau0 in [1.0, 0.1]:
            return muram.MuramTauSlice(path, i, tau0)
        elif tau0 in [384, 400]:
            return muram.MuramSlice(path, i, 'yz', '0' + str(tau0))
        else:
            raise ValueError("Invalid tau0")

    def downsample(arr, scale):
        if scale is None or scale == pixelsize:
            return arr
        factor = scale / pixelsize
        return zoom(arr, zoom=1 / factor, order=1)

    iter_list = list(range(iter1, iter2 + 1, 50)) if average_over_time else [iter1, iter2]
    n = len(iter_list)

    sum_vars = {
        'bx': 0, 'by': 0, 'bz': 0,
        'vx': 0, 'vy': 0, 'vz': 0,
        'ex': 0, 'ey': 0, 'ez': 0,
        'sz': 0,  'sz_emergence': 0, 'sz_shear': 0,
        'curl_Ez': 0, 'int': 0
    }

    first_bz = last_bz = None

    for idx, i in enumerate(iter_list):
        tau = load_slice(i)
        bx, by, bz = tau.By, tau.Bz, tau.Bx
        vx, vy, vz = tau.vz / 1e5, tau.vy / 1e5, tau.vx / 1e5  # to km/s
        ex, ey, ez = calc_e_field_from_v_and_b(vx, vy, vz, bx, by, bz)
        curl_Ez = compute_precise_curl_Ez(ex, ey, pixelsize)
        sz,sz_emergence,sz_shear = vertical_poynting_flux(vx, vy, vz, bx, by, bz)

        # Load intensity map
        intensity = muram.MuramIntensity(path, i)

        if fwhm and pixelsize:
            sigma = fwhm / 1.665 / pixelsize
            for arr in [bx, by, bz, vx, vy, vz, ex, ey, ez, curl_Ez, sz, sz_emergence,sz_shear,intensity]:
                gaussian_filter(arr, sigma, mode="wrap", output=arr)

        # Downsample
        bx, by, bz = [downsample(arr, ds) for arr in (bx, by, bz)]
        vx, vy, vz = [downsample(arr, ds) for arr in (vx, vy, vz)]
        ex, ey, ez = [downsample(arr, ds) for arr in (ex, ey, ez)]
        curl_Ez = downsample(curl_Ez, ds)
        sz = downsample(sz, ds)
        sz_emergence = downsample(sz_emergence, ds)
        sz_shear = downsample(sz_shear, ds)
        intensity = downsample(intensity, ds)

        # Store bz snapshots
        if idx == 0:
            first_bz = bz.copy()
        if idx == n - 1:
            last_bz = bz.copy()

        if average_over_time:
            sum_vars['bx'] += bx
            sum_vars['by'] += by
            sum_vars['bz'] += bz
            sum_vars['vx'] += vx
            sum_vars['vy'] += vy
            sum_vars['vz'] += vz
            sum_vars['ex'] += ex
            sum_vars['ey'] += ey
            sum_vars['ez'] += ez
            sum_vars['sz'] += sz
            sum_vars['sz_emergence'] += sz_emergence
            sum_vars['sz_shear'] += sz_shear
            sum_vars['curl_Ez'] += curl_Ez
            sum_vars['int'] += intensity
        elif idx == 0:
            sum_vars = {
                'bx': bx, 'by': by, 'bz': bz,
                'vx': vx, 'vy': vy, 'vz': vz,
                'ex': ex, 'ey': ey, 'ez': ez,
                'sz': sz,
                'sz_emergence':sz_emergence,'sz_shear':sz_shear,
                'curl_Ez': curl_Ez,
                'int': intensity
            }

    divisor = n if average_over_time else 1

    return {
        'b_muram_x': sum_vars['bx'] / divisor,
        'b_muram_y': sum_vars['by'] / divisor,
        'b_muram_z': sum_vars['bz'] / divisor,
        'v_muram_x': sum_vars['vx'] / divisor,
        'v_muram_y': sum_vars['vy'] / divisor,
        'v_muram_z': sum_vars['vz'] / divisor,
        'e_muram_x': sum_vars['ex'] / divisor,
        'e_muram_y': sum_vars['ey'] / divisor,
        'e_muram_z': sum_vars['ez'] / divisor,
        's_muram_z': sum_vars['sz'] / divisor,
        's_muram_z_emergence': sum_vars['sz_emergence'] / divisor,
        's_muram_z_shear': sum_vars['sz_shear'] / divisor,
        'curl_Ez': sum_vars['curl_Ez'] / divisor,
        'dBz_dt': (last_bz - first_bz) / dt_s,
        'intensity': sum_vars['int'] / divisor
    }

def plot_inductive_term_grid(path, directory,tau0, iter1, pixelsize_km, muram, issi, fwhm=200):
    """
    Plot 4x4 grid of curl(Ez) vs -dBz/dt for different dt and spatial binning (ds).
    """
    fig, axs = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    dt_list = [30, 60, 120, 360]
    ds_factors = [1, 4, 8, 16]

    for i, dt in enumerate(dt_list):
        iter2 = iter1 + 5 * dt
        for j, factor in enumerate(ds_factors):
            ds = factor * pixelsize_km
            print(i,j,ds,dt)
            result = issi.load_muram_vars_with_inductivity(
                path=path,
                tau0=tau0,
                iter1=iter1,
                iter2=iter2,
                muram=muram,
                issi=issi,
                fwhm=fwhm,
                pixelsize=pixelsize_km,
                dt_s=dt,
                ds=ds
            )

            x = result['curl_Ez'].flatten()
            y = -result['dBz_dt'].flatten()
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            ax = axs[i, j]
            lims = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
            hist = ax.hist2d(x, y, bins=100, range=[[-4,4], [-4,4]],
                             cmap='plasma', norm=LogNorm())[3]

            ax.plot(lims, lims, 'k--', linewidth=1)
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(lims, slope * np.array(lims) + intercept, color='orange', linestyle='--', linewidth=1.5,label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
            corr, _ = spearmanr(x, y)
            rmse = mean_squared_error(x, y, squared=False)

            ax.set_title(f"dt={dt}s, ds={ds}km\nCorr={corr:.2f}, RMSE={rmse:.2f}")
            ax.set_xlabel("curl(Ez) [G/s]")
            ax.set_ylabel("-dBz/dt [G/s]")

    plt.suptitle("Scatter plots: curl(Ez) vs -dBz/dt", fontsize=16)
    plt.savefig(directory+"inductive_term_grid_"+str(tau0)+".png", dpi=200)
    plt.show()

def decompose_electric_field(ex, ey, dx, dy):
    """
    Decompose a 2D electric field (ex, ey) into inductive (solenoidal) and
    non-inductive (irrotational) components using Helmholtz decomposition.

    Parameters:
    -----------
    ex, ey : 2D np.ndarray
        Horizontal components of the electric field (e.g., in V/cm).
    dx, dy : float
        Pixel size in x and y directions (in km).

    Returns:
    --------
    e_inductive_x, e_inductive_y : np.ndarray
        Inductive (solenoidal) electric field components.
    e_noninductive_x, e_noninductive_y : np.ndarray
        Non-inductive (irrotational) electric field components.
    """
    ny, nx = ex.shape

    # Fourier coordinates
    kx = fftfreq(nx, d=dx) * 2 * np.pi
    ky = fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='xy')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1  # avoid division by zero

    # Fourier transform of electric field components
    ex_hat = fft2(ex)
    ey_hat = fft2(ey)

    # Compute divergence and curl in Fourier space
    div_hat = kx * ex_hat + ky * ey_hat
    curl_hat = kx * ey_hat - ky * ex_hat

    # Scalar and vector potentials
    phi_hat = div_hat / k2
    psi_hat = curl_hat / k2

    # Reconstruct components
    e_noninductive_x = np.real(ifft2(kx * phi_hat))
    e_noninductive_y = np.real(ifft2(ky * phi_hat))

    e_inductive_x = np.real(ifft2(-ky * psi_hat))
    e_inductive_y = np.real(ifft2(kx * psi_hat))

    return e_inductive_x, e_inductive_y, e_noninductive_x, e_noninductive_y

def plot_inductive_terms(curl_Ez, dBz_dt, directory,fname,title_suffix='', bins=100):
    """
    2x2 layout:
    [curl_Ez]      [-dBz/dt]
    [residual]     [density scatter with metrics]
    [bins] is number of bins for ax.hist2d
    """
    
    residual = curl_Ez + dBz_dt
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

   # Set common color limits for top row for consistent comparison
    vmin = -20 #min(np.min(curl_Ez), np.min(-dBz_dt))
    vmax = 20  #max(np.max(curl_Ez), np.max(-dBz_dt))

    # Compute axis in km
    ny, nx = curl_Ez.shape
    #extent = [0, nx * pixelsize_km, 0, ny * pixelsize_km]

    # Top-left: curl(E)_z map
    im0 = axs[0, 0].imshow(curl_Ez, origin='lower', cmap='RdBu', aspect='auto',
                          vmin=vmin, vmax=vmax) #, extent=extent)
    axs[0, 0].set_title(f'curl(E)_h {title_suffix}')
    axs[0, 0].set_xlabel('X [km]')
    axs[0, 0].set_ylabel('Y [km]')
    plt.colorbar(im0, ax=axs[0, 0], label='G/s')

    # Top-right: -dBz/dt map
    im1 = axs[0, 1].imshow(-dBz_dt, origin='lower', cmap='RdBu', aspect='auto',
                          vmin=vmin, vmax=vmax) #, extent=extent)
    axs[0, 1].set_title(f'-dBz/dt {title_suffix}')
    axs[0, 1].set_xlabel('X [km]')
    axs[0, 1].set_ylabel('Y [km]')
    plt.colorbar(im1, ax=axs[0, 1], label='G/s')

    # Bottom-left: residual map
    im2 = axs[1, 0].imshow(residual, origin='lower', cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Residual: curl(E)_z + dBz/dt')
    axs[1, 0].set_xlabel('X [km]')
    axs[1, 0].set_ylabel('Y [km]')
    plt.colorbar(im2, ax=axs[1, 0], label='G/s')

    # Replace lower-right with density scatter
    ax = axs[1, 1]
    x = curl_Ez.flatten()
    y = (-dBz_dt).flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    lims = [vmin, vmax]
    
    h = ax.hist2d(x, y, bins=bins, range=[[lims[0], lims[1]], [lims[0], lims[1]]],
                  cmap='plasma', norm='log')[3]
    fig.colorbar(h, ax=ax, label='Counts')

    # Plot 1:1 and linear fit
    ax.plot(lims, lims, 'k--', linewidth=1)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(lims, slope * np.array(lims) + intercept, color='orange', linestyle='--', linewidth=1.5,
            label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    
    # Stats
    corr, _ = stats.spearmanr(x, y)
    rmse = mean_squared_error(x, y, squared=False)

    ax.set_xlabel('curl(E)_z')
    ax.set_ylabel('-dBz/dt')
    ax.set_title(f'Density Plot\nCorr={corr:.2f}, RMSE={rmse:.2f}')
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    figname = directory+fname   
    plt.savefig(figname, dpi=150)
    plt.show()

def plot_inductive_terms_with_psd(curl_Ez, dBz_dt,directory,fname, title_suffix='', bins=100, ds0=16):
    """
    Compare curl_Ez and -dBz/dt spatially and spectrally:
    [curl_Ez]        [-dBz/dt]
    [Spatial PSD]    [Density scatter with metrics]

    Parameters:
    -----------
    curl_Ez : 2D array
        Vertical component of curl(E) in G/s.
        Bins is number of bins for ax.hist2d
    dBz_dt : 2D array
        Time derivative of Bz in G/s.
    ds0 : float
        Pixel size in km.
    """
    residual = curl_Ez + dBz_dt
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    vmin = -20
    vmax = 20
    ny, nx = curl_Ez.shape

    # Top-left: curl(E)_z map
    im0 = axs[0, 0].imshow(curl_Ez, origin='lower', cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title(f'curl(E)_h {title_suffix}')
    axs[0, 0].set_xlabel('X [km]')
    axs[0, 0].set_ylabel('Y [km]')
    plt.colorbar(im0, ax=axs[0, 0], label='G/s')

    # Top-right: -dBz/dt map
    im1 = axs[0, 1].imshow(-dBz_dt, origin='lower', cmap='RdBu', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f'-dBz/dt {title_suffix}')
    axs[0, 1].set_xlabel('X [km]')
    axs[0, 1].set_ylabel('Y [km]')
    plt.colorbar(im1, ax=axs[0, 1], label='G/s')

    # Bottom-left: spatial PSD plot
    def compute_psd(field):
        f = np.fft.fft2(field)
        fshift = np.fft.fftshift(f)
        psd2d = np.abs(fshift)**2 * (ds0**2)  # factor to get units in G²·km²/s²
        return np.fft.fftshift(psd2d)

    freqs_x = np.fft.fftfreq(nx, d=ds0)
    freqs_y = np.fft.fftfreq(ny, d=ds0)
    kx, ky = np.meshgrid(freqs_x, freqs_y)
    k = np.sqrt(kx**2 + ky**2)
    k = np.fft.fftshift(k)

    psd_curl = compute_psd(curl_Ez)
    psd_dbz = compute_psd(dBz_dt)

    k_flat = k.flatten()
    psd_curl_flat = psd_curl.flatten()
    psd_dbz_flat = psd_dbz.flatten()

    # Bin radially
    k_bins = np.linspace(1e-4, np.max(k), 50)
    k_center = 0.5 * (k_bins[:-1] + k_bins[1:])
    psd_curl_binned = np.zeros_like(k_center)
    psd_dbz_binned = np.zeros_like(k_center)

    for i in range(len(k_center)):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
        psd_curl_binned[i] = np.mean(psd_curl_flat[mask]) if np.any(mask) else np.nan
        psd_dbz_binned[i] = np.mean(psd_dbz_flat[mask]) if np.any(mask) else np.nan

    # Normalize PSD curves
    psd_curl_binned /= np.nanmax(psd_curl_binned)
    psd_dbz_binned /= np.nanmax(psd_dbz_binned)

    ax_psd = axs[1, 0]
    ax_psd.loglog(1 / k_center, psd_curl_binned, label='curl(E)_z')
    ax_psd.loglog(1 / k_center, psd_dbz_binned, label='-dBz/dt')
    ax_psd.set_xlabel('Spatial scale [km]')
    ax_psd.set_ylabel('Normalized PSD [G²·km²/s²]')
    ax_psd.set_title('Spatial Power Spectral Density')
    ax_psd.grid(True, which='both')
    ax_psd.legend()

    # Bottom-right: density scatter plot
    ax = axs[1, 1]
    x = curl_Ez.flatten()
    y = (-dBz_dt).flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    #lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    
    lims = [vmin, vmax]
    
    h = ax.hist2d(x, y, bins=bins, range=[[lims[0], lims[1]], [lims[0], lims[1]]],
                  cmap='plasma', norm='log')[3]
    plt.colorbar(h, ax=ax, label='Counts')

    ax.plot(lims, lims, 'k--', linewidth=1)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(lims, slope * np.array(lims) + intercept, color='orange', linestyle='--',
            linewidth=1.5, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

    corr, _ = stats.spearmanr(x, y)
    rmse = mean_squared_error(x, y, squared=False)

    ax.set_xlabel('curl(E)_z [G/s]')
    ax.set_ylabel('-dBz/dt [G/s]')
    ax.set_title(f'Density Plot\nCorr={corr:.2f}, RMSE={rmse:.2f}')
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    figname = directory+fname   
    plt.savefig(figname, dpi=150)
    plt.show()

def plot_residual_histogram(curl_Ez, dBz_dt):
    residual = curl_Ez + dBz_dt
    plt.figure(figsize=(6, 4))
    plt.hist(residual.flatten(), bins=100, color='gray', alpha=0.8)
    plt.title('Histogram of curl(E)_z + dBz/dt')
    plt.xlabel('Residual (G/s)')
    plt.ylabel('Pixel count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def flatten_valid(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None):
    """Flatten and mask out invalid data and optionally apply an additional mask."""
    base_mask = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        base_mask &= mask
    return a[base_mask].flatten(), b[base_mask].flatten()

def to_scalar_and_round(val, digits=3):
    try:
        if isinstance(val, np.ndarray):
            val = val.item()  # Convert 0-D array to scalar
        if val is None or np.isnan(val):
            return np.nan
        return round(float(val), digits)
    except Exception:
        return np.nan
def flows2metrics_and_save_dataframe(
    directory,
    fname,
    fwhm,
    dt,
    iter1,
    iter2,
    tau_levels,
    i,
    method,
    v_muram_x,
    v_muram_y,
    v_muram_z,
    s_muram_z,s_muram_z_emergence,s_muram_z_shear,
    int_muram,
    v_model_x,
    v_model_y,
    v_model_z,
    s_model_z,s_model_z_emergence,s_model_z_shear,
    b_muram_x,
    b_muram_y,
    b_muram_z,
    e_model_x=None,
    e_model_y=None,
    e_model_z=None,
    bthr=150,
    ithr=3.2e8
):
    from issi import calc_e_field_from_v_and_b, flatten_valid, Metrics,to_scalar_and_round

    tau = tau_levels[i]
    records = []

    # Derived fields
    e_muram_x, e_muram_y, e_muram_z = calc_e_field_from_v_and_b(v_muram_x, v_muram_y, v_muram_z, b_muram_x, b_muram_y, b_muram_z)
    if e_model_x is None or e_model_y is None or e_model_z is None:
        e_model_x, e_model_y, e_model_z = calc_e_field_from_v_and_b(v_model_x, v_model_y, v_model_z, b_muram_x, b_muram_y, b_muram_z)

    field_sets = {
        'v': (v_muram_x, v_muram_y, v_muram_z, v_model_x, v_model_y, v_model_z),
        'e': (e_muram_x, e_muram_y, e_muram_z, e_model_x, e_model_y, e_model_z),
        'Sz': (s_muram_z, s_model_z),
        'Sz_emergence': (s_muram_z_emergence, s_model_z_emergence),
        'Sz_shear': (s_muram_z_shear, s_model_z_shear)
    }


    bmag = np.sqrt(b_muram_x**2 + b_muram_y**2 + b_muram_z**2)

    masks = {
        'all': np.ones_like(bmag, dtype=bool),
        'B_high': bmag >= bthr,
        'B_low': bmag < bthr,
        'I_low': int_muram < ithr,
        'I_high': int_muram >= ithr
    }

    for region, mask in masks.items():
        for field_type, data in field_sets.items():
            if field_type in ['v', 'e']:
                v1x, v1y, v1z, v2x, v2y, v2z = data  #def __init__(self, v1x, v2x, v1y=None, v2y=None, v1z=None, v2z=None, mask=None):
                metrics = Metrics(v1x, v2x, v1y, v2y, v1z, v2z, mask=mask)
                vmag1=np.sqrt(v1x*v1x+v1y*v1y+v1z*v1z)
                vmag2=np.sqrt(v2x*v2x+v2y*v2y+v2z*v2z)
                muram_mean = to_scalar_and_round(np.nanmean(vmag1[mask]),3)
                model_mean = to_scalar_and_round(np.nanmean(vmag2[mask]),3)
                muram_abs_mean = to_scalar_and_round(np.nanmean(np.abs(vmag1[mask])),3)
                model_abs_mean = to_scalar_and_round(np.nanmean(np.abs(vmag2[mask])),3)
                #print("v1x stats", np.nanmin(v1x), np.nanmax(v1x), np.isnan(v1x).sum(), np.isinf(v1x).sum())
                record = {
                    'region': region,
                    'field': field_type,
                    'fwhm': fwhm,
                    'dt': dt,
                    'iter1': iter1,
                    'iter2': iter2,
                    'tau': tau,
                    'method': method,
                    'vector_corr': to_scalar_and_round(metrics.vector_correlation_coefficient(),3),   
                    'cosine_sim': to_scalar_and_round(metrics.cosine_similarity_index(),3),
                    'norm_error': to_scalar_and_round(metrics.normalized_error(),3),
                    'mean_abs_rel_error': to_scalar_and_round(metrics.mean_absolute_relative_error(),3),
                    'median_abs_rel_error': to_scalar_and_round(metrics.median_absolute_relative_error(),3),
                    'energy_ratio': to_scalar_and_round(metrics.energy_ratio(),3),
                    'muram_mean': muram_mean,
                    'model_mean': model_mean,
                    'muram_abs_mean': muram_abs_mean,
                    'model_abs_mean': model_abs_mean,
                }
                for label, (v1, v2) in zip(['x', 'y', 'z'], [(v1x, v2x), (v1y, v2y), (v1z, v2z)]):
                    x_flat, y_flat = flatten_valid(v1, v2, mask)
                    if len(x_flat) > 0:
                        slope, intercept = np.polyfit(x_flat, y_flat, 1)
                        corr, _ = stats.spearmanr(x_flat, y_flat)
                        rmse = mean_squared_error(x_flat, y_flat, squared=False)
                        record.update({
                            f'{label}_slope': to_scalar_and_round(slope,3),
                            f'{label}_intercept': to_scalar_and_round(intercept,3),
                            f'{label}_spearmanr': to_scalar_and_round(corr,3),
                            f'{label}_rmse': to_scalar_and_round(rmse,3)
                        })
                records.append(record)
            else:
                v1x, v2x = data
                metrics = Metrics(v1x, v2x, mask=mask)
                x_flat, y_flat = flatten_valid(v1x, v2x, mask)
                slope, intercept, corr, rmse = (np.nan,)*4
                #
                muram_mean = to_scalar_and_round(np.nanmean(v1x[mask]),3)
                model_mean = to_scalar_and_round(np.nanmean(v2x[mask]),3)
                muram_abs_mean = to_scalar_and_round(np.nanmean(np.abs(v1x[mask])),3)
                model_abs_mean = to_scalar_and_round(np.nanmean(np.abs(v2x[mask])),3)

                if len(x_flat) > 0:
                    slope, intercept = np.polyfit(x_flat, y_flat, 1)
                    corr, _ = stats.spearmanr(x_flat, y_flat)
                    rmse = mean_squared_error(x_flat, y_flat, squared=False)
                records.append({
                    'region': region,
                    'field': field_type,
                    'fwhm': fwhm,
                    'dt': dt,
                    'iter1': iter1,
                    'iter2': iter2,
                    'tau': tau,
                    'method': method,
                    'vector_corr': to_scalar_and_round(metrics.vector_correlation_coefficient(),3),
                    'cosine_sim': np.nan,
                    'norm_error': to_scalar_and_round(metrics.normalized_error(),3),
                    'mean_abs_rel_error': to_scalar_and_round(metrics.mean_absolute_relative_error(),3),
                    'median_abs_rel_error':to_scalar_and_round(metrics.median_absolute_relative_error(),3),
                    'energy_ratio': to_scalar_and_round(metrics.energy_ratio(),3),
                    'x_slope': to_scalar_and_round(slope,3),
                    'x_intercept': to_scalar_and_round(intercept,3),
                    'x_spearmanr': to_scalar_and_round(corr,3),
                    'x_rmse': to_scalar_and_round(rmse,3),
                    'muram_mean': muram_mean,
                    'model_mean': model_mean,
                    'muram_abs_mean': muram_abs_mean,
                    'model_abs_mean': model_abs_mean,
                })

    df = pd.DataFrame(records)
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, f"{fname}_metrics.csv"), index=False)
    #import ace_tools as tools; tools.display_dataframe_to_user(name="Field Comparison Metrics", dataframe=df)
    return df
def calc_e_field_from_v_and_b(vx, vy, vz, bx, by, bz):
    """
    Compute electric field components from v × B in V/cm units.

    Parameters:
    -----------
    vx, vy, vz : 2D arrays
        Velocity components in km/s.
    bx, by, bz : 2D arrays
        Magnetic field components in Gauss.

    Returns:
    --------
    ex, ey, ez : 2D arrays
        Electric field components in V/cm.
    """
    #  velocity in km/s 


    ex = -(vy * bz - vz * by) * 1e-3
    ey = -(vz * bx - vx * bz) * 1e-3
    ez = -(vx * by - vy * bx) * 1e-3

    return ex, ey, ez

def plot_poynting_decomposition_comparison(directory, fname,
                                           e_muram_x, e_muram_y,
                                           e_model_x, e_model_y,
                                           b_model_x, b_model_y,
                                           s_muram_z, s_model_z,
                                           dx=16, dy=16):
    """
    Plot a 3x3 panel figure comparing inductive, non-inductive, and total vertical Poynting flux (Sz)
    for MURaM and model electric fields using Helmholtz decomposition, assuming Sz is precomputed.
    """

    # Decompose electric fields - in V/cm: modeled "_m" and reconstructred "e"
    eix_m, eiy_m, enx_m, eny_m = decompose_electric_field(e_muram_x, e_muram_y, dx, dy)
    eix_r, eiy_r, enx_r, eny_r = decompose_electric_field(e_model_x, e_model_y, dx, dy)

    # Recalculate inductive/noninductive components of Sz
    def calc_sz(ex, ey, bx, by):
        return (bx * ey - by * ex)* 1e8 / (4 * np.pi)

    sz_i_m = calc_sz(eix_m, eiy_m, b_model_x, b_model_y)
    sz_n_m = calc_sz(enx_m, eny_m, b_model_x, b_model_y)

    sz_i_r = calc_sz(eix_r, eiy_r, b_model_x, b_model_y)
    sz_n_r = calc_sz(enx_r, eny_r, b_model_x, b_model_y)

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    titles = ['Inductive', 'Non-inductive', 'Total']
    muram_fluxes = [sz_i_m, sz_n_m, s_muram_z]
    model_fluxes = [sz_i_r, sz_n_r, s_model_z]

    for j in range(3):
        #vmin_j, vmax_j = np.nanpercentile(muram_fluxes[j], [1, 99])
        lims_symmetric = np.nanpercentile(muram_fluxes[j], 90)
        im = axs[0, j].imshow(muram_fluxes[j], origin='lower', cmap='coolwarm_r', vmin=-lims_symmetric, vmax=lims_symmetric)
        axs[0, j].set_title(f'MURaM {titles[j]}')
        fig.colorbar(im, ax=axs[0, j], label='$S_z$ [erg s$^{-1}$ cm$^{-2}$]')

    for j in range(3):
        lims_symmetric = np.nanpercentile(model_fluxes[j], 90)
        #vmin_j, vmax_j = np.nanpercentile(model_fluxes[j], [1, 99])
        im = axs[1, j].imshow(model_fluxes[j], origin='lower', cmap='coolwarm_r', vmin=-lims_symmetric, vmax=lims_symmetric)
        axs[1, j].set_title(f'Model {titles[j]}')
        fig.colorbar(im, ax=axs[1, j], label='$S_z$ [erg s$^{-1}$ cm$^{-2}$]')

    for j in range(3):
        x = muram_fluxes[j].flatten()
        y = model_fluxes[j].flatten()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) == 0 or len(y) == 0:
            print( 'No valid data')
            continue
        lims_symmetric = np.nanpercentile(np.concatenate([x, y]), 90)
        #lims = np.percentile(np.concatenate([x, y]), [1, 99])
        #add_density_subplot(x, y , 9, 'Sz Comparison [1e5]', lims=[-szrange,szrange])
        ax = axs[2, j]
        #h = ax.hist2d(x, y, bins=100, norm='log', range=[-lims_symmetric, lims_symmetric], cmap='plasma')[3]
        #fig.colorbar(h, ax=ax, label='Counts')
        add_density_subplot(x, y , 7+j, titles[j], lims=[-lims_symmetric,lims_symmetric])
        #slope, intercept = np.polyfit(x, y, 1)
        #corr, _ = spearmanr(x, y)
        #rmse = mean_squared_error(x, y, squared=False)
        #ax.plot(lims, lims, 'k--')
        #ax.plot(lims, slope * np.array(lims) + intercept, 'orange', linestyle='--')
        #ax.set_title(f'Scatter: {titles[j]}\nCorr={corr:.2f}, RMSE={rmse:.2f}')
        #ax.set_xlabel('MURaM $S_z$')
        #ax.set_ylabel('Model $S_z$')

    plt.tight_layout()
    outname = os.path.join(directory, f"sz_decomp_{fname}.pdf")
    plt.savefig(outname)
    plt.show()
    print(f"✅ Saved: {outname}")
    return outname,muram_fluxes,model_fluxes

def plot_and_metrics_muram_comparison(
    directory,
    fname,
    fwhm,
    dt,
    iter1,
    iter2,
    tau_levels,
    i,
    method,
    v_muram_x,
    v_muram_y,
    v_muram_z,
    s_muram_z,
    s_muram_z_emergence,
    s_muram_z_shear,
    int_muram,
    v_model_x,
    v_model_y,
    v_model_z,
    s_model_z,
    s_model_z_emergence,
    s_model_z_shear,
    b_muram_x,
    b_muram_y,
    b_muram_z,
    field_type="velocity",
    e_model_x=None,
    e_model_y=None,
    e_model_z=None,
    bthr=None,
    ithr=None,
    velrange=None,
    szrange=None
):
    if tau_levels[i] == 0.1:
        strlev = "Tau_0.1"
        szrange=1e8
        if velrange is None:
            velrange=5
    elif tau_levels[i] == 1.0:
        strlev = "Tau_1.0"
        szrange=5e8
        if velrange is None:
            velrange=8
    elif tau_levels[i] == 384:
        strlev = "Yz_0384"
        szrange=5e8
        if velrange is None:
            velrange=8
    elif tau_levels[i] == 400:
        strlev = "Yz_0400"
        szrange=1e8
        if velrange is None:
            velrange=5
    else:
        raise ValueError("tau0 must be one of 0.1, 1.0, 384, 400")

    erange=0.5

    figname = os.path.join(directory, f"cmp_muram_{fname}.pdf")
    plt.figure(figsize=(18, 15))

    if field_type == "electric":
        # === Compute E_muram = -v x B in V/cm===
        e_muram_x = (v_muram_y * b_muram_z - v_muram_z * b_muram_y)*1e-3
        e_muram_y = (v_muram_z * b_muram_x - v_muram_x * b_muram_z)*1e-3
        e_muram_z = (v_muram_x * b_muram_y - v_muram_y * b_muram_x)*1e-3

        # === Use e_model if provided, else compute from v_model × B in V/cm ===
        if e_model_x is None or e_model_y is None or e_model_z is None:
            e_model_x = (v_model_y * b_muram_z - v_model_z * b_muram_y)*1e-3
            e_model_y = (v_model_z * b_muram_x - v_model_x * b_muram_z)*1e-3
            e_model_z = (v_model_x * b_muram_y - v_model_y * b_muram_x)*1e-3

        # === Plotting ===
 
        plt.subplot(3, 3, 1)
        plt.imshow(e_muram_x, cmap='coolwarm_r', vmin=-erange, vmax=erange, origin='lower')
        plt.colorbar(label='Ex [V/cm]')
        plt.title(f"MURaM Ex: {strlev}")

        plt.subplot(3, 3, 2)
        plt.imshow(e_muram_y, cmap='coolwarm_r',  vmin=-erange, vmax=erange, origin='lower')
        plt.colorbar(label='Ey [V/cm]')
        plt.title(f"MURaM Ey: {iter1}-{iter2}")

        plt.subplot(3, 3, 3)
        plt.imshow(s_muram_z, cmap='coolwarm_r', vmin=-szrange, vmax=szrange, origin='lower')
        plt.colorbar(label='Sz [ergs/cm²/s]')
        plt.title("MURaM Sz (dt0 = 10s)")

        plt.subplot(3, 3, 4)
        plt.imshow(e_model_x, cmap='coolwarm_r', vmin=-erange, vmax=erange, origin='lower')
        plt.colorbar(label='Ex_inv [V/cm]')
        plt.title(f"Model Ex: {strlev}, FWHM={fwhm}, Dt={dt}s")

        plt.subplot(3, 3, 5)
        plt.imshow(e_model_y, cmap='coolwarm_r', vmin=-erange, vmax=erange, origin='lower')
        plt.colorbar(label='Ey_inv [V/cm]')
        plt.title(f"Model Ey (iter1 = {iter1})")

        plt.subplot(3, 3, 6)
        plt.imshow(s_model_z, cmap='coolwarm_r', vmin=-szrange, vmax=szrange, origin='lower')
        plt.colorbar(label='Sz_inv [ergs/cm²/s]')
        plt.title(f"iter2 = {iter2}")

        add_density_subplot(e_muram_x, e_model_x, 7, 'Ex Comparison', lims=[-erange, erange])
        add_density_subplot(e_muram_y, e_model_y, 8, 'Ey Comparison', lims=[-erange, erange])
        add_density_subplot(s_muram_z, s_model_z , 9, 'Sz Comparison [1e5]', lims=[-szrange,szrange])

    elif field_type == "velocity":
        # === Recover velocity from E if needed ===
        if (v_model_x is None or v_model_y is None or v_model_z is None) and \
           (e_model_x is not None and e_model_y is not None and e_model_z is not None):

            B2 = b_muram_x**2 + b_muram_y**2 + b_muram_z**2
            eps = 1e-10  # avoid division by zero

            # v = - (E × B) / |B|²
            v_model_x = -(e_model_y * b_muram_z - e_model_z * b_muram_y) *1e3/ (B2 + eps)
            v_model_y = -(e_model_z * b_muram_x - e_model_x * b_muram_z) *1e3/ (B2 + eps)
            v_model_z = -(e_model_x * b_muram_y - e_model_y * b_muram_x) *1e3/ (B2 + eps)

        # === Plotting ===
        plt.subplot(3, 3, 1)
        plt.imshow(v_muram_x, cmap='coolwarm_r', vmin=-velrange, vmax=velrange, origin='lower')
        plt.colorbar(label='Vx [km/s]')
        plt.title(f"MURaM Vx: {strlev}")

        plt.subplot(3, 3, 2)
        plt.imshow(v_muram_y, cmap='coolwarm_r', vmin=-velrange, vmax=velrange, origin='lower')
        plt.colorbar(label='Vy [km/s]')
        plt.title(f"MURaM Vy: {iter1}-{iter2}")

        plt.subplot(3, 3, 3)
        plt.imshow(s_muram_z, cmap='coolwarm_r', vmin=-szrange, vmax=szrange, origin='lower')
        plt.colorbar(label='Sz [ergs/cm²/s]')
        plt.title("MURaM Sz (dt0 = 10s)")

        plt.subplot(3, 3, 4)
        plt.imshow(v_model_x, cmap='coolwarm_r', vmin=-velrange, vmax=velrange, origin='lower')
        plt.colorbar(label='Vx_inv [km/s]')
        plt.title(f"Inverted Vx: {strlev}, FWHM={fwhm}, Dt={dt}s")

        plt.subplot(3, 3, 5)
        plt.imshow(v_model_y, cmap='coolwarm_r', vmin=-velrange, vmax=velrange, origin='lower')
        plt.colorbar(label='Vy_inv [km/s]')
        plt.title(f"Inverted Vy: {iter1}-{iter2}")

        plt.subplot(3, 3, 6)
        plt.imshow(s_model_z, cmap='coolwarm_r', vmin=-szrange, vmax=szrange, origin='lower')
        plt.colorbar(label='Sz_inv [ergs/cm²/s]')
        plt.title("Inverted Sz (dt0 = 10s)")

        add_density_subplot(v_muram_x, v_model_x, 7, 'Vx Comparison', lims=[-velrange, velrange])
        add_density_subplot(v_muram_y, v_model_y, 8, 'Vy Comparison', lims=[-velrange, velrange])
        add_density_subplot(s_muram_z , s_model_z, 9, 'Sz Comparison', lims=[-szrange,szrange])

#
    else:
        raise ValueError("field_type must be 'velocity' or 'electric'")

    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()

    # add an extra figure comparing histograms of the magnitude of the MUram and inferred velocity |V| and poynting fluxes Sz:
    # Plot histograms and save to separate figure
    plot_histograms_of_magnitudes(
        directory, fname,
        v_muram_x, v_muram_y, v_muram_z,
        v_model_x, v_model_y, v_model_z,
        s_muram_z, s_model_z,
        field_type=field_type
    )

    return flows2metrics_and_save_dataframe(
        directory, fname, fwhm, dt, iter1, iter2, tau_levels, i,method,
        v_muram_x, v_muram_y, v_muram_z, s_muram_z,s_muram_z_emergence,s_muram_z_shear,int_muram,
        v_model_x, v_model_y, v_model_z, s_model_z,s_model_z_emergence,s_model_z_shear,
        b_muram_x, b_muram_y, b_muram_z,
        e_model_x, e_model_y, e_model_z,bthr=bthr,ithr=ithr)

def plot_histograms_of_magnitudes(directory, fname,
                                   v_muram_x, v_muram_y, v_muram_z,
                                   v_model_x, v_model_y, v_model_z,
                                   s_muram_z, s_model_z,
                                   field_type="velocity"):
    """
    Plot histograms comparing magnitudes |V| or |E| and Poynting flux S_z for MURaM and model data.
    Automatically cleans data to avoid NaN/Inf issues.
    """
    import matplotlib.pyplot as plt
    import os

    # Compute magnitude
    mag_muram = np.sqrt(v_muram_x**2 + v_muram_y**2 + v_muram_z**2)
    mag_model = np.sqrt(v_model_x**2 + v_model_y**2 + v_model_z**2)
    label = "|V| [km/s]" if field_type == "velocity" else "|E| [V/cm]"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: |V| or |E| histogram ---
    bins_mag = np.histogram_bin_edges(
        safe_concatenate_for_histogram(mag_muram, mag_model), bins='auto'
    )
    axs[0].plot(bins_mag[:-1], cleaned_histogram_data(mag_muram, bins_mag), color='blue', label='MURaM')
    axs[0].plot(bins_mag[:-1], cleaned_histogram_data(mag_model, bins_mag), color='orange', label='Model')
    axs[0].set_title(f'Histogram of {label}')
    axs[0].set_xlabel(label)
    axs[0].set_ylabel('Counts')
    axs[0].set_xlim(0, np.nanpercentile(np.concatenate([mag_muram, mag_model]), 99))
    axs[0].legend()

    # --- Right panel: Sz histogram ---
    bins_sz = np.histogram_bin_edges(
        safe_concatenate_for_histogram(s_muram_z, s_model_z), bins='auto'
    )
    axs[1].plot(bins_sz[:-1], cleaned_histogram_data(s_muram_z, bins_sz), color='blue', label='MURaM')
    axs[1].plot(bins_sz[:-1], cleaned_histogram_data(s_model_z, bins_sz), color='orange', label='Model')
    axs[1].set_title('Histogram of $S_z$ [ergs/cm²/s]')
    axs[1].set_xlabel('$S_z$ [ergs/cm²/s]')
    axs[1].set_ylabel('Counts')
    sz_values = np.concatenate([s_muram_z.flatten(), s_model_z.flatten()])
    sz_max = np.nanpercentile(np.abs(sz_values), 90)
    axs[1].set_xlim(-sz_max, sz_max)
    axs[1].legend()


    plt.tight_layout()
    outpath = os.path.join(directory, f"hist_muram_{fname}.pdf")
    plt.savefig(outpath)
    plt.close(fig)

    print(f"📊 Histogram saved to {outpath}")
    return outpath

def safe_concatenate_for_histogram(a, b):
    """
    Safely concatenate two arrays for histogram bin calculation by removing NaNs and Infs.
    """
    a_clean = a[np.isfinite(a)]
    b_clean = b[np.isfinite(b)]
    return np.concatenate([a_clean, b_clean])

def cleaned_histogram_data(data, bins):
    """
    Compute histogram from cleaned data (no NaNs or Infs).
    """
    data_clean = data[np.isfinite(data)]
    hist, _ = np.histogram(data_clean, bins=bins)
    return hist


import matplotlib.pyplot as plt
import numpy as np
import os

def plot_masks_and_histograms(intensity, bmag, ithr, bthr, fname, directory='.'):
    """
    Plot a 3x2 panel showing B field magnitude and intensity maps, masks, and histograms.

    Parameters:
    -----------
    intensity : 2D np.ndarray
        Intensity map.
    bmag : 2D np.ndarray
        Magnetic field magnitude map.
    ithr : float
        Intensity threshold.
    bthr : float
        Magnetic field threshold in Gauss.
    fname : str
        Base filename for saving.
    directory : str
        Output directory for saving the figure.
    """
    # Create masks
    intmask = (intensity < ithr).astype(int)
    bmask = (bmag >= bthr).astype(int)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # --- Row 1: Magnetic Field ---
    # |B| map
    im_b = axs[0, 0].imshow(bmag, cmap='magma', origin='lower', vmin=0, vmax=300)
    axs[0, 0].set_title('|B| Map [G]')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    fig.colorbar(im_b, ax=axs[0, 0], fraction=0.046, pad=0.04)

    # B mask
    imm = axs[0, 1].imshow(bmask, cmap='gray', origin='lower')
    axs[0, 1].set_title(f'B Mask (|B| ≥ {bthr} G)')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Y')
    fig.colorbar(imm, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # |B| histogram
    axs[0, 2].hist(bmag.flatten(), bins=100, color='steelblue')
    axs[0, 2].axvline(bthr, color='red', linestyle='--', label=f'Threshold = {bthr} G')
    axs[0, 2].set_title('|B| Histogram')
    axs[0, 2].set_xlabel('|B| [G]')
    axs[0, 2].set_ylabel('Counts')
    axs[0, 2].set_xlim([0, 800])
    axs[0, 2].legend()

    # --- Row 2: Intensity ---
    # Intensity map
    im_i = axs[1, 0].imshow(intensity, cmap='inferno', origin='lower', vmin=2e8, vmax=6e8)
    axs[1, 0].set_title('Intensity Map')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Y')
    fig.colorbar(im_i, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # Intensity mask
    imk = axs[1, 1].imshow(intmask, cmap='gray', origin='lower')
    axs[1, 1].set_title(f'Intensity Mask (I < {ithr:.1e})')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    fig.colorbar(imk, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # Intensity histogram
    axs[1, 2].hist(intensity.flatten(), bins=100, color='orange')
    axs[1, 2].axvline(ithr, color='red', linestyle='--', label=f'Threshold = {ithr:.1e}')
    axs[1, 2].set_title('Intensity Histogram')
    axs[1, 2].set_xlabel('Intensity')
    axs[1, 2].set_ylabel('Counts')
    axs[1, 2].set_xlim([2e8, 7e8])
    axs[1, 2].legend()

    plt.tight_layout()
    figname = directory+fname   #os.path.join(directory, f"mask_{fname}.pdf")
    plt.savefig(figname, dpi=150)
    plt.show()
    return figname


def create_strong_field_mask(bx, by, bz, bthr):
    """
    Create a binary mask where the magnetic field magnitude exceeds a threshold.

    Parameters:
    -----------
    bx, by, bz : np.ndarray
        Components of the magnetic field in Gauss.
    bthr : float
        Threshold magnetic field magnitude in Gauss.

    Returns:
    --------
    mask : np.ndarray
        Binary mask with 1 where |B| >= bthr, and 0 elsewhere.
    """
    # Compute magnitude of magnetic field
    B_mag = np.sqrt(bx**2 + by**2 + bz**2)

    # Create mask
    mask = (B_mag >= bthr).astype(np.uint8)

    return mask


def pad_array(unpadded_array, xpadding_width, ypadding_width, pad_value=0):
    """
    Pads a 2D array with specified padding widths on each side.

    Parameters:
    -----------
    unpadded_array : np.ndarray
        2D input array to be padded.
    xpadding_width : int
        Number of columns to add on the left and right sides.
    ypadding_width : int
        Number of rows to add on the top and bottom sides.
    pad_value : scalar, optional
        Value to use for padding. Default is 0.

    Returns:
    --------
    padded_array : np.ndarray
        The padded 2D array.
    """
    return np.pad(
        unpadded_array,
        pad_width=((ypadding_width, ypadding_width), (xpadding_width, xpadding_width)),
        mode='constant',
        constant_values=pad_value
    )


def create_intensity_mask(intensity, ithr, mode='above'):
    """
    Create a binary mask for intensity thresholds.

    Parameters:
    -----------
    intensity : np.ndarray
        2D intensity map.
    ithr : float
        Intensity threshold value.
    mode : str, optional
        Thresholding mode: 'above' for intensity >= ithr,
                           'below' for intensity < ithr.
                           Default is 'above'.

    Returns:
    --------
    mask : np.ndarray
        Binary mask (1s and 0s) based on the specified threshold.
    """
    if mode == 'above':
        mask = (intensity >= ithr).astype(np.uint8)
    elif mode == 'below':
        mask = (intensity < ithr).astype(np.uint8)
    else:
        raise ValueError("mode must be 'above' or 'below'")

    return mask

def read_all_csvs_in_directory(directory):
    """
    Read all CSV files in a directory, add filename and method info, 
    and return a concatenated DataFrame along with summary stats.

    Parameters:
    -----------
    directory : str
        Path to the directory containing CSV files.

    Returns:
    --------
    df_all : pd.DataFrame
        Concatenated DataFrame from all CSVs, including 'fname' and 'method' columns.

    summary : pd.DataFrame
        Grouped summary (mean ± std) of key metrics by field.
    """
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    all_dfs = []

    for file in csv_files:
        df = pd.read_csv(os.path.join(directory, file))
        df['fname'] = os.path.splitext(file)[0]  # Add base filename
        df['method'] = df['fname'].str.extract(r'^([a-zA-Z0-9_]+)_')  # Extract method prefix
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Grouped mean ± std summary by field
    summary = df_all.groupby('field')[
        ['vector_corr', 'cosine_sim', 'norm_error',
        'mean_abs_rel_error', 'median_abs_rel_error', 'energy_ratio']].agg(['mean', 'std'])

    return df_all, summary

# Re-define the safe summarization function since the previous definition was not executed
def summarize_method_performance_by_tau_safe(df_all):
    df_all['tau'] = df_all['tau'].astype(float)
    df_all['method'] = df_all['method'].astype(str)

    # Define all expected metrics
    expected_metrics = [
        'vector_corr', 'cosine_sim', 'norm_error', 'mean_abs_rel_error',
        'energy_ratio', 'muram_mean', 'model_mean', 'muram_abs_mean', 'model_abs_mean'
    ]
    available_metrics = [col for col in expected_metrics if col in df_all.columns]

    summary_df = (
        df_all.groupby(['method', 'tau', 'dt', 'fwhm', 'field'])[available_metrics]
        .mean(numeric_only=True).reset_index()
    )

    # Add performance rank (based on vector_corr)
    if 'vector_corr' in summary_df.columns:
        summary_df['rank'] = summary_df.groupby(['tau', 'field'])['vector_corr'].rank(ascending=False)

    return summary_df


def summarize_method_performance_by_tau(df_all):
    """
    Summarize performance of each method for different tau levels, dt, and fwhm.
    
    Parameters:
    -----------
    df_all : pd.DataFrame
        Combined metrics from all sources with 'method', 'tau', 'fwhm', 'dt', and 'field'.

    Returns:
    --------
    summary_df : pd.DataFrame
        Grouped average performance and ranking.
    """
    # Ensure correct types
    df_all['tau'] = df_all['tau'].astype(float)
    df_all['method'] = df_all['method'].astype(str)

    # Group and compute mean performance for each method/tau/dt/fwhm/field combo
    summary_df = (
        df_all.groupby(['method', 'tau', 'dt', 'fwhm', 'field'])[
            ['vector_corr', 'cosine_sim', 'norm_error', 'mean_abs_rel_error', 'energy_ratio','muram_mean','model_mean','muram_abs_mean','model_abs_mean']
        ].mean().reset_index()
    )

    # Add rankings (higher vector_corr is better)
    summary_df['rank'] = summary_df.groupby(['tau', 'field'])['vector_corr'].rank(ascending=False)

    return summary_df

def plot_sz_model_vs_muram(
    s_total_model, s_emerg_model, s_shear_model,
    s_total_muram, s_emerg_muram, s_shear_muram,directory,fname,
    mask=None, extent=None, vmin=None, vmax=None
):
    """
    Compare model and MURaM Sz components with scatter plots and unified color/scatter limits.

    Parameters:
    -----------
    s_total_model, s_emerg_model, s_shear_model : 2D arrays
        Model-derived Sz components
    s_total_muram, s_emerg_muram, s_shear_muram : 2D arrays
        MURaM Sz components
    mask : 2D bool array, optional
        Region mask
    extent : list or tuple, optional
        Plot extent [xmin, xmax, ymin, ymax]
    vmin, vmax : float, optional
        Color scale range for all image panels and scatter axes
    """
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    figname = os.path.join(directory, f"sz_shear_emerg_comparison_{fname}.pdf")

    # --- Helper function for image plot ---
    def imshow_field(ax, data, title):
        im = ax.imshow(data, origin='lower', cmap='RdBu', extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('X-axis [pix]')
        ax.set_ylabel('Y-axis [pix]')
        plt.colorbar(im, ax=ax)

    # --- Top row: Model ---
    imshow_field(axs[0, 0], s_total_model, 'Model: $S_z$ total')
    imshow_field(axs[0, 1], s_emerg_model, 'Model: $S_z$ emergence')
    imshow_field(axs[0, 2], s_shear_model, 'Model: $S_z$ shear')

    # --- Middle row: MURaM ---
    imshow_field(axs[1, 0], s_total_muram, 'MURaM: $S_z$ total')
    imshow_field(axs[1, 1], s_emerg_muram, 'MURaM: $S_z$ emergence')
    imshow_field(axs[1, 2], s_shear_muram, 'MURaM: $S_z$ shear')

    # --- Bottom row: Scatter plots ---

    def flatten(a, b):
        a_flat = a.ravel()
        b_flat = b.ravel()
        if mask is not None:
            m_flat = mask.ravel()
            a_flat = a_flat[m_flat]
            b_flat = b_flat[m_flat]
        return a_flat, b_flat

    fields = [
        ('Total $S_z$', s_total_model, s_total_muram),
        ('Emergence $S_z$', s_emerg_model, s_emerg_muram),
        ('Shear $S_z$', s_shear_model, s_shear_muram)
    ]

    for j, (label, m_model, m_muram) in enumerate(fields):
        x, y = flatten(m_model, m_muram)
        ax = axs[2, j]
        ax.scatter(x, y, s=1, alpha=0.3)
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', label='y = x')
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel('Model')
        ax.set_ylabel('MURaM')
        ax.set_title(f'Scatter: {label}')
        ax.grid(True)
        ax.legend()

    for ax in axs.flat:
        ax.set_xlabel('X [Mm]')
        ax.set_ylabel('Y [Mm]')

    plt.savefig(figname, dpi=200)
    plt.show()
    print("issi.plot_sz_model_vs_muram: sz plot saved in: "+figname)

def plot_sz_component_diagnostics(s_total, s_emerg, s_shear, tau0=None,iter1=None,mask=None, extent=None, vmin=None, vmax=None,directory=None):
    """
    Plots 2x3 panel with Sz components and scatter comparison.
    
    Parameters:
    -----------
    s_total : 2D array
        Total Sz from E × B
    s_emerg, s_shear : 2D arrays
        Emergence and shear components
    mask : 2D array of bool, optional
        If given, restrict scatter/residual plots
    extent : list or tuple, optional
        Plot extent [xmin, xmax, ymin, ymax] in Mm or pixels
    vmin, vmax : float, optional
        Common color scale limits for Sz maps
    """
    s_sum = s_emerg + s_shear
    residual = s_total - s_sum

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    mtit='MURaM at i='+str(iter1)+', tau='+str(tau0)+', '
    # Plot maps
    im0 = axs[0, 0].imshow(s_total, origin='lower', cmap='RdBu', extent=extent, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title(mtit+': $S_z$ total')
    plt.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(s_emerg, origin='lower', cmap='RdBu', extent=extent, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(mtit+'$S_z$ emergence')
    plt.colorbar(im1, ax=axs[0, 1])

    im2 = axs[0, 2].imshow(s_shear, origin='lower', cmap='RdBu', extent=extent, vmin=vmin, vmax=vmax)
    axs[0, 2].set_title(mtit+'$S_z$ shear')
    plt.colorbar(im2, ax=axs[0, 2])

    #im3 = axs[1, 0].imshow(s_sum, origin='lower', cmap='RdBu', extent=extent, vmin=vmin, vmax=vmax)
    #axs[1, 0].set_title('Sum: emergence + shear')
    #plt.colorbar(im3, ax=axs[1, 0])

    #----4
    # Flatten arrays with optional mask
    x = s_total.ravel()
    y = s_sum.ravel()
    #res = residual.ravel()

    if mask is not None:
        flat_mask = mask.ravel()
        x = x[flat_mask]
        y = y[flat_mask]
        #res = res[flat_mask]

    # Scatter plot
    axs[1, 0].scatter(x, y, s=1, alpha=0.3)
    axs[1, 0].plot([x.min(), x.max()], [x.min(), x.max()], 'k--', label='y = x')
    axs[1, 0].set_xlabel('Sz_total')
    axs[1, 0].set_ylabel('Sz_emergence + Sz_shear')
    axs[1, 0].set_title(mtit+'Scatter: sum vs. total')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Residual plot
    #im5 = axs[1, 2].imshow(residual, origin='lower', cmap='bwr', extent=extent)
    #axs[1, 2].set_title('Residual: Sz_total - (emerg + shear)')
    #plt.colorbar(im5, ax=axs[1, 2])

    #for ax in axs.flat:
    #    ax.set_xlabel('X')
    #    ax.set_ylabel('Y')
    #----5
    # Flatten arrays with optional mask
    x = s_total.ravel()
    y = s_emerg.ravel()
    #res = residual.ravel()

    if mask is not None:
        flat_mask = mask.ravel()
        x = x[flat_mask]
        y = y[flat_mask]
        #res = res[flat_mask]

    # Scatter plot
    axs[1, 1].scatter(x, y, s=1, alpha=0.3)
    axs[1, 1].plot([x.min(), x.max()], [x.min(), x.max()], 'k--', label='y = x')
    axs[1, 1].set_xlabel('Sz_total')
    axs[1, 1].set_ylabel('Sz_emergence')
    axs[1, 1].set_title(mtit+'Scatter: emergence vs. total')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    #----6
        # Flatten arrays with optional mask
    x = s_total.ravel()
    y = s_shear.ravel()
    #res = residual.ravel()

    if mask is not None:
        flat_mask = mask.ravel()
        x = x[flat_mask]
        y = y[flat_mask]
        #res = res[flat_mask]

    # Scatter plot
    axs[1, 2].scatter(x, y, s=1, alpha=0.3)
    axs[1, 2].plot([x.min(), x.max()], [x.min(), x.max()], 'k--', label='y = x')
    axs[1, 2].set_xlabel('Sz_total')
    axs[1, 2].set_ylabel('Sz_shear')
    axs[1, 2].set_title(mtit+'Scatter: shear vs. total')
    axs[1, 2].legend()
    axs[1, 2].grid(True)
     
    plt.tight_layout()
    fname=directory+"sz_muram_terms_"+str(tau0)+'.'+str(iter1)+".png"
    plt.savefig(fname, dpi=200)
    plt.show()
    print("Sz plot saved in: "+fname)

def norm(vx, vy, vz) -> np.ndarray:
    """ Compute the norm of the vector field.

        Parameters:
        ----------
        vx, vy, vz: np.ndarray. Components of the vector field.

        Returns:
        --------
        norm: np.ndarray. Norm of the vector.
    """
    return np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def spearman_correlation_coefficient(v1: np.ndarray, v2: np.ndarray, axis: int = None) -> np.ndarray:
    """
        Compute the Spearman correlation coefficient between two images.

        Parameters:
        ----------
        v1, v2: np.ndarray. Input images.
        axis: Union[int, tuple]. Axis along which to compute the metric (i.e., samples/time axis).

        Returns:
        --------
        spearman_corr: np.ndarray. Spearman correlation coefficient between v1 and v2.
    """

    # Reshape based on axis
    if axis is not None:

        # Extract shape
        shape = v1.shape
        # Move the specified axis to the front and reshape
        v1 = np.moveaxis(v1, axis, 0).reshape(shape[axis], -1)
        v2 = np.moveaxis(v2, axis, 0).reshape(shape[axis], -1)

        # Compute the Spearman correlation coefficient for each sample
        sc = []
        for i in range(shape[axis]):
            sc.append(stats.spearmanr(v1[i], v2[i]))

        return np.stack(sc)

    else:
        # Spearman correlation coefficient
        sc, _ = stats.spearmanr(v1.flatten(), v2.flatten())

        return sc


def vertical_poynting_flux(
        vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
        bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
        ex: np.ndarray = None, ey: np.ndarray = None) -> dict:
        """
        Compute total, emergence, and shear vertical Poynting flux components [erg/cm²/s].

        Parameters:
        -----------
        vx, vy, vz : np.ndarray
        Velocity components [km/s]
        bx, by, bz : np.ndarray
        Magnetic field components [G]
        ex, ey : np.ndarray, optional
        Electric field components [V/cm]. If not provided, they are computed from v and B.

        Returns:
        --------
        dict with:
        'sz_total'     : total vertical Poynting flux [erg/cm²/s]
        'sz_emergence' : emergence component [erg/cm²/s]
        'sz_shear'     : shear component [erg/cm²/s]
        """

        four_pi = 4 * np.pi

        # Compute Ex, Ey from v and B if not provided
        if ex is None or ey is None:
            if vx is None or vy is None or vz is None:
                raise ValueError("If ex and ey are not provided, vx, vy, and vz must be given.")
            ex, ey, ez = calc_e_field_from_v_and_b(vx, vy, vz, bx, by, bz)

        # Total Sz from E × B
        sz_total = 1e8 * (ex * by - ey * bx) / four_pi  # [erg/cm²/s]

        # Bh^2 = Bx^2 + By^2
        bh2 = bx**2 + by**2

        # Vh ⋅ Bh = vx * bx + vy * by
        vdotb_h = vx * bx + vy * by

        # Emergence term: Bh^2 * vz / (4π)
        sz_emergence = 1e5 * bh2 * vz / four_pi

        # Shear term: - (Vh ⋅ Bh) * Bz / (4π)
        sz_shear = -1e5 * vdotb_h * bz / four_pi

        return sz_total, sz_emergence, sz_shear

from typing import Optional

class Metrics:
    def __init__(self, v1x, v2x, v1y=None, v2y=None, v1z=None, v2z=None, mask=None):
        self.mask = mask

        if mask is not None:
            self.v1x, self.v2x = v1x[mask], v2x[mask]
            self.v1y = v1y[mask] if v1y is not None else np.zeros_like(self.v1x)
            self.v2y = v2y[mask] if v2y is not None else np.zeros_like(self.v2x)
            self.v1z = v1z[mask] if v1z is not None else np.zeros_like(self.v1x)
            self.v2z = v2z[mask] if v2z is not None else np.zeros_like(self.v2x)
        else:
            self.v1x, self.v2x = v1x, v2x
            self.v1y = v1y if v1y is not None else np.zeros_like(v1x)
            self.v2y = v2y if v2y is not None else np.zeros_like(v2x)
            self.v1z = v1z if v1z is not None else np.zeros_like(v1x)
            self.v2z = v2z if v2z is not None else np.zeros_like(v2x)

        self.n_valid = len(self.v1x)

    def count_valid(self):
        return self.n_valid
    
    def squared_error(self) -> np.ndarray:
        """ Compute the squared error between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Squared error between v1 and v2.
        """
        return (self.v1x - self.v2x) ** 2 + (self.v1y - self.v2y) ** 2 + (self.v1z - self.v2z) ** 2

    def absolute_error(self) -> np.ndarray:
        """ Compute the absolute error between two vector fields.

            Returns:
            --------
            metric: np.ndarray. Absolute error between v1 and v2.
        """
        return np.sqrt(self.squared_error())

    def absolute_relative_error(self) -> np.ndarray:
        """ Compute the absolute relative error between two vector fields.
            Note that v1 is the reference field.

            Returns:
            --------
            metric: np.ndarray. Absolute relative error between v1 and v2.
        """
        return self.absolute_error() / norm(self.v1x, self.v1y, self.v1z)

    def cosine_similarity(self) -> np.ndarray:
        """ Compute the cosine similarity between two vector fields. """
        dot_prod = self.v1x * self.v2x + self.v1y * self.v2y + self.v1z * self.v2z
        norm1 = norm(self.v1x, self.v1y, self.v1z)
        norm2 = norm(self.v2x, self.v2y, self.v2z)
        denom = norm1 * norm2

        # Avoid division by zero and filter invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine_sim = np.where(denom > 0, dot_prod / denom, np.nan)

        return cosine_sim

    def cosine_similarity_index(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the cosine similarity index between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Cosine similarity index between v1 and v2.
        """
        cos_sim = self.cosine_similarity()
        return np.nanmean(cos_sim, axis=axis)  # nanmean skips NaNs


    def mean_squared_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean squared error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean squared error between v1 and v2.
        """
        return np.mean(self.squared_error(), axis=axis)

    def mean_absolute_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean absolute error between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Mean absolute error between v1 and v2.
        """
        return np.mean(self.absolute_error(), axis=axis)

    def mean_absolute_relative_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the mean absolute relative error between two vector fields.
        Avoids division by zero using safe masking.

        Parameters:
        ----------
        axis: int or tuple, optional. Defaults to None. Axis along which to compute the metric.

        Returns:
        --------
        metric: np.ndarray or float. Mean absolute relative error.
        """
    
        abs_err = self.absolute_error()
        norm_ref = norm(self.v1x, self.v1y, self.v1z)

        # Mask regions where the norm is too small to avoid division by zero
        valid_mask = norm_ref > 1e-10

        # Only compute relative error where valid
        relative_error = np.full_like(norm_ref, np.nan)
        relative_error[valid_mask] = abs_err[valid_mask] / norm_ref[valid_mask]

        # Compute the mean excluding NaNs
        return np.nanmean(relative_error, axis=axis)


    def median_absolute_relative_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the median absolute relative error between two vector fields.
            Avoids division by zero using safe masking.

            Parameters:
            ----------
            axis: int or tuple, optional. Defaults to None. Axis along which to compute the metric.

            Returns:
            --------
            metric: np.ndarray or float. Median absolute relative error.
        """
        abs_err = self.absolute_error()
        norm_ref = norm(self.v1x, self.v1y, self.v1z)

        # Mask regions where the norm is very small to avoid division by zero
        valid_mask = norm_ref > 1e-10
        relative_error = np.full_like(norm_ref, np.nan)
        relative_error[valid_mask] = abs_err[valid_mask] / norm_ref[valid_mask]

        # Use np.nanmedian to skip invalid (nan) entries
        return np.nanmedian(relative_error, axis=axis)

    def normalized_error(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the normalized error between two vector fields.
        Uses the definition from Schrijver et al. (2006), avoiding division by zero.

        Parameters:
        ----------
        axis: int or tuple, optional. Defaults to None. Axis along which to compute the metric.

        Returns:
        --------
        metric: np.ndarray or float. Normalized error between v1 and v2.
        """

        abs_err = self.absolute_error()
        norm_ref = norm(self.v1x, self.v1y, self.v1z)

        # Create a mask where the norm is > 0 to avoid division by zero
        valid_mask = norm_ref > 1e-10  # or your preferred small threshold
        abs_err_masked = np.where(valid_mask, abs_err, 0.0)
        norm_masked = np.where(valid_mask, norm_ref, 0.0)

        # Compute the metric safely
        numerator = np.sum(abs_err_masked, axis=axis)
        denominator = np.sum(norm_masked, axis=axis)

        # Final division with safeguard
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result = np.nan_to_num(result, nan=np.nan)  # Keep NaN if denominator was 0

        return result

    def vector_correlation_coefficient(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """ Compute the vector correlation coefficient between two vector fields.

            Parameters:
            ----------
            axis: tuple, optional. Defaults to None. Axis along which to compute the metric (i.e., spatial axes).

            Returns:
            --------
            metric: np.ndarray. Vector correlation coefficient between v1 and v2.
        """
        # Dot product
        dot_prod = self.v1x * self.v2x + self.v1y * self.v2y + self.v1z * self.v2z

        # Norms
        norm1_sq = self.v1x**2 + self.v1y**2 + self.v1z**2
        norm2_sq = self.v2x**2 + self.v2y**2 + self.v2z**2

        # Sum over axis or keep as is
        dot_sum = np.nansum(dot_prod, axis=axis)
        norm1_sum = np.nansum(norm1_sq, axis=axis)
        norm2_sum = np.nansum(norm2_sq, axis=axis)

        # Denominator
        denominator = np.sqrt(norm1_sum * norm2_sum)

        # Avoid division by zero or invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            coef = np.where(denominator > 0, dot_sum / denominator, np.nan)

        return coef

    def energy_ratio(self, axis: Union[int, tuple] = None) -> floating[Any]:
        """Compute the energy ratio between two vector fields.

        Parameters:
        ----------
        axis: tuple or int, optional. Axis along which to compute the metric.

        Returns:
        --------
        metric: np.ndarray or float. Energy ratio between v2 and v1.
        """
        energy_v2 = self.v2x**2 + self.v2y**2 + self.v2z**2
        energy_v1 = self.v1x**2 + self.v1y**2 + self.v1z**2

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(np.sum(energy_v2, axis=axis), np.sum(energy_v1, axis=axis))
            ratio = np.nan_to_num(ratio, nan=np.nan, posinf=np.nan, neginf=np.nan)

        return ratio
    
if __name__ == '__main__':
    """ Compute metrics between two vector fields.

        Parameters
        ----------
        None.

        Returns
        -------
        metrics: dict. Dictionary containing the computed metrics.
    """

    # Example vector fields with random values
    # Dimensions are (samples, height, width)
    sample_axis = 0
    spatial_axes = (1, 2)
    n_samples = 10  # Could be interpreted as time
    nx = ny = 48
    b_muram_x, b_muram_y, b_muram_z = [np.random.rand(n_samples, ny, nx) for _ in range(3)]
    v_muram_x, v_muram_y, v_muram_z = [np.random.rand(n_samples, ny, nx) for _ in range(3)]
    v_model_x, v_model_y, v_model_z = [np.random.rand(n_samples, ny, nx) for _ in range(3)]

    # Compute the vertical Poynting flux
    s_muram_z, s_muram_z_emergence, s_muram_z_shear = vertical_poynting_flux(v_muram_x, v_muram_y, v_muram_z,
                                       b_muram_x, b_muram_y, b_muram_z)
    s_model_z, s_model_z_emergence, s_model_z_shear = vertical_poynting_flux(v_model_x, v_model_y, v_model_z,
                                       b_muram_x, b_muram_y, b_muram_z)

    # ---------------------------------------------------------------------
    # Examples of computations accounting for all vector components at once
    # ---------------------------------------------------------------------

    # Initialize the Metrics class:
    # v1 is the reference field.
    # v2 is the model field.
    # Note that the z component is optional.
    metrics = Metrics(v1x=v_muram_x, v2x=v_model_x, v1y=v_muram_y, v2y=v_model_y)

    # Compute metrics over the entire data at once
    metrics_for_v_over_all_samples = {
        'vector_correlation_coefficient': metrics.vector_correlation_coefficient(),
        'cosine_similarity_index': metrics.cosine_similarity_index(),
        'normalized_error': metrics.normalized_error(),
        'mean_absolute_relative_error': metrics.mean_absolute_relative_error(),
        'median_absolute_relative_error': metrics.median_absolute_relative_error(),
        'energy_ratio': metrics.energy_ratio()
    }

    # Compute metrics over the entire data at once
    metrics_for_v_per_sample = {
        'vector_correlation_coefficient': metrics.vector_correlation_coefficient(axis=spatial_axes),
        'cosine_similarity_index': metrics.cosine_similarity_index(axis=spatial_axes),
        'normalized_error': metrics.normalized_error(axis=spatial_axes),
        'mean_absolute_relative_error': metrics.mean_absolute_relative_error(axis=spatial_axes),
        'median_absolute_relative_error': metrics.median_absolute_relative_error(axis=spatial_axes),
        'energy_ratio': metrics.energy_ratio(axis=spatial_axes)
    }


    # ---------------------------------------------------------------------
    # Examples of computations for a single vector component (e.g., vx)
    # ---------------------------------------------------------------------

    # Initialize the Metrics class:
    # v1 is the reference field.
    # v2 is the model field.
    # Note that you can just compare images by assigning them to v1x and v2x.
    metrics_vx = Metrics(v1x=v_muram_x, v2x=v_model_x)

    # Compute metrics over the entire data at once
    metrics_for_vx_over_all_samples  = {
        'vector_correlation_coefficient': metrics_vx.vector_correlation_coefficient(),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x),
        'normalized_error': metrics_vx.normalized_error(),
        'mean_absolute_relative_error': metrics_vx.mean_absolute_relative_error(),
        'median_absolute_relative_error': metrics_vx.median_absolute_relative_error(),
        'energy_ratio': metrics_vx.energy_ratio()
    }

    # Compute metrics over the entire data at once
    metrics_for_vx_per_sample = {
        'vector_correlation_coefficient': metrics_vx.vector_correlation_coefficient(axis=spatial_axes),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x, axis=sample_axis),
        'normalized_error': metrics_vx.normalized_error(axis=spatial_axes),
        'mean_absolute_relative_error': metrics_vx.mean_absolute_relative_error(axis=spatial_axes),
        'median_absolute_relative_error': metrics_vx.median_absolute_relative_error(axis=spatial_axes),
        'energy_ratio': metrics_vx.energy_ratio(axis=spatial_axes)
    }

    # ---------------------------------------------------------------------
    # Examples of computations for a single vector component (e.g., Sz)
    # ---------------------------------------------------------------------

    # Initialize the Metrics class:
    # v1 is the reference field.
    # v2 is the model field.
    # Note that you can just compare images by assigning them to v1x and v2x.
    metrics_sz = Metrics(v1x=s_muram_z, v2x=s_model_z)
    metrics_sz_emergence = Metrics(v1x=s_muram_z_emergence, v2x=s_model_z_emergence)
    metrics_sz_shear= Metrics(v1x=s_muram_z_shear, v2x=s_model_z_shear)
    
    # Compute metrics over the entire data at once
    metrics_for_sz_over_all_samples = {
        'vector_correlation_coefficient': metrics_sz.vector_correlation_coefficient(),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x),
        'normalized_error': metrics_sz.normalized_error(),
        'mean_absolute_relative_error': metrics_sz.mean_absolute_relative_error(),
        'median_absolute_relative_error': metrics_sz.median_absolute_relative_error(),
        'energy_ratio': metrics_sz.energy_ratio()
    }

    # Compute metrics over the entire data at once
    metrics_for_sz_per_sample = {
        'vector_correlation_coefficient': metrics_sz.vector_correlation_coefficient(axis=spatial_axes),
        'spearman_correlation_coefficient': spearman_correlation_coefficient(v_muram_x, v_model_x, axis=sample_axis),
        'normalized_error': metrics_sz.normalized_error(axis=spatial_axes),
        'mean_absolute_relative_error': metrics_sz.mean_absolute_relative_error(axis=spatial_axes),
        'median_absolute_relative_error': metrics_sz.median_absolute_relative_error(axis=spatial_axes),
        'energy_ratio': metrics_sz.energy_ratio(axis=spatial_axes)
    }

def debug_histogram_nan_issue(mag_muram, mag_model):
    """
    Debug helper to check for NaNs or invalid values before histogramming.
    """
    issues = {}

    for name, arr in [("mag_muram", mag_muram), ("mag_model", mag_model)]:
        issues[name] = {
            "has_nan": np.isnan(arr).any(),
            "has_inf": np.isinf(arr).any(),
            "min": np.nanmin(arr),
            "max": np.nanmax(arr),
            "shape": arr.shape,
        }

    return issues
