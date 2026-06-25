
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from scipy.special import gamma

def generate_correlated_field(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb, method='phir_randn',correlate_method="exp"):
    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz) 
        phi0_k = fftn(phi0_r)                 
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    d = 3  #
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    Eb_r = ifftn(Eb_k).real
    
    Eb_final = ((Eb_r - np.mean(Eb_r)) / np.std(Eb_r)) * sigma_Eb + mean_Eb
    
    return Eb_final

def generate_correlated_field_truncated(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb,truncated_value , method='phir_randn',correlate_method="exp"):
    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz)  
        phi0_k = fftn(phi0_r)                
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    d = 3  
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    Eb_r = ifftn(Eb_k).real
    
    Eb_r = np.abs(Eb_r)  

    Eb_final = (Eb_r - np.mean(Eb_r)) / np.std(Eb_r) * sigma_Eb + mean_Eb
    Eb_final[Eb_final > truncated_value] = mean_Eb
    
    return Eb_final

def generate_correlated_field_half_guassian(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb, method='phir_randn',correlate_method="exp"):

    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz)  
        phi0_k = fftn(phi0_r)                 
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    d = 3  
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    Eb_r = ifftn(Eb_k).real
    
    Eb_r = np.abs(Eb_r)  

    Eb_final = (Eb_r - np.mean(Eb_r)) / np.std(Eb_r) * sigma_Eb + mean_Eb
    Eb_final[Eb_final < mean_Eb] = mean_Eb
    
    return Eb_final

def generate_correlated_field_truncated_gaussian(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb, method='phir_randn',correlate_method="exp"):
    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz) 
        phi0_k = fftn(phi0_r)                 
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    d = 3  
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    Eb_r = ifftn(Eb_k).real
    
    Eb_r[Eb_r < 0] = 0 

    Eb_final = (Eb_r - np.mean(Eb_r)) / np.std(Eb_r) * sigma_Eb + mean_Eb
    
    return Eb_final

def plot_field_slices(field, cmap='viridis'):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    mid_x, mid_y, mid_z = [s//2 for s in field.shape]
    
    axes[0].imshow(field[mid_x, :, :], cmap=cmap)
    axes[0].set_title('XY Plane (Z=mid)')
    
    axes[1].imshow(field[:, mid_y, :], cmap=cmap)
    axes[1].set_title('XZ Plane (Y=mid)')
    
    axes[2].imshow(field[:, :, mid_z], cmap=cmap)
    axes[2].set_title('YZ Plane (X=mid)')
    
    plt.colorbar(axes[0].images[0], ax=axes, fraction=0.02)
    plt.tight_layout()
    plt.show()

def plot_3d_isosurface(field, levels=7):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
    
    min_val, max_val = np.min(field), np.max(field)
    level_values = np.linspace(min_val, max_val, levels+2)[1:-1]
    
    for level in level_values:
        verts, faces, _, _ = marching_cubes(field, level)
        ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                       alpha=0.3, linewidth=0.5, antialiased=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Isosurface Visualization')
    plt.show()


def plot_volume(field, cmap='viridis', alpha=0.3):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    x,y,z = np.indices(field.shape)
    
    threshold = np.percentile(field, 80)
    mask = field > threshold
    
    ax.scatter(x[mask], y[mask], z[mask], 
               c=field[mask], cmap=cmap, 
               alpha=alpha, marker='o', 
               label='Field Intensity')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    plt.title('3D Random Field Volume Rendering')
    plt.tight_layout()

