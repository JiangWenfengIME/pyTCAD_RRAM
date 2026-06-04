
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from scipy.special import gamma

def generate_correlated_field(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb, method='phir_randn',correlate_method="exp"):
    """
    生成具有高斯相关性的3D随机场
    参数:
        Nx, Ny, Nz: 网格尺寸
        lambda_: 相关长度
        sigma_Eb: 目标标准差
        mean_Eb: 目标均值
        method: 'gaussian'或'exponential'生成方式
    返回:
        Eb_final: 归一化后的空间相关场
    """
    # 步骤1: 生成初始随机场
    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz)  # 实空间高斯噪声
        phi0_k = fftn(phi0_r)                 # 傅里叶变换到频域
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    # 步骤2: 生成k值网格
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    # 步骤3: 应用高斯功率谱P(k)
    d = 3  # 维度
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    # 步骤4: 逆变换到实空间
    Eb_r = ifftn(Eb_k).real
    
    # 步骤5: 归一化
    # Eb_final = (Eb_r / np.std(Eb_r)) * sigma_Eb + mean_Eb

    Eb_final = ((Eb_r - np.mean(Eb_r)) / np.std(Eb_r)) * sigma_Eb + mean_Eb
    
    return Eb_final

def generate_correlated_field_truncated(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb,truncated_value , method='phir_randn',correlate_method="exp"):
    """
    生成具有高斯相关性的3D随机场
    参数:
        Nx, Ny, Nz: 网格尺寸
        lambda_: 相关长度
        sigma_Eb: 目标标准差
        mean_Eb: 目标均值
        method: 'gaussian'或'exponential'生成方式
    返回:
        Eb_final: 归一化后的空间相关场
    """
    # 步骤1: 生成初始随机场
    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz)  # 实空间高斯噪声
        phi0_k = fftn(phi0_r)                 # 傅里叶变换到频域
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    # 步骤2: 生成k值网格
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    # 步骤3: 应用高斯功率谱P(k)
    d = 3  # 维度
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    # # 步骤4: 逆变换到实空间
    # Eb_r = ifftn(Eb_k).real
    
    # # 步骤5: 归一化
    # Eb_final = (Eb_r / np.std(Eb_r)) * sigma_Eb + mean_Eb

    # 步骤4: 逆变换到实空间
    Eb_r = ifftn(Eb_k).real
    
    # 半高斯化（关键一步）
    Eb_r = np.abs(Eb_r)   # 或者 Eb_r[Eb_r < 0] = 0  (如果你想做截断高斯)

    # 步骤5: 归一化
    Eb_final = (Eb_r - np.mean(Eb_r)) / np.std(Eb_r) * sigma_Eb + mean_Eb
    Eb_final[Eb_final > truncated_value] = mean_Eb
    
    return Eb_final

def generate_correlated_field_half_guassian(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb, method='phir_randn',correlate_method="exp"):
    """
    生成具有高斯相关性的3D随机场
    参数:
        Nx, Ny, Nz: 网格尺寸
        lambda_: 相关长度
        sigma_Eb: 目标标准差
        mean_Eb: 目标均值
        method: 'gaussian'或'exponential'生成方式
    返回:
        Eb_final: 归一化后的空间相关场
    """
    # 步骤1: 生成初始随机场
    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz)  # 实空间高斯噪声
        phi0_k = fftn(phi0_r)                 # 傅里叶变换到频域
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    # 步骤2: 生成k值网格
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    # 步骤3: 应用高斯功率谱P(k)
    d = 3  # 维度
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    # # 步骤4: 逆变换到实空间
    # Eb_r = ifftn(Eb_k).real
    
    # # 步骤5: 归一化
    # Eb_final = (Eb_r / np.std(Eb_r)) * sigma_Eb + mean_Eb

    # 步骤4: 逆变换到实空间
    Eb_r = ifftn(Eb_k).real
    
    # 半高斯化（关键一步）
    Eb_r = np.abs(Eb_r)   # 或者 Eb_r[Eb_r < 0] = 0  (如果你想做截断高斯)

    # 步骤5: 归一化
    Eb_final = (Eb_r - np.mean(Eb_r)) / np.std(Eb_r) * sigma_Eb + mean_Eb
    Eb_final[Eb_final < mean_Eb] = mean_Eb
    
    return Eb_final

def generate_correlated_field_truncated_gaussian(Lx, Ly, Lz, spac, lambda_, sigma_Eb, mean_Eb, method='phir_randn',correlate_method="exp"):
    """
    生成具有高斯相关性的3D随机场
    参数:
        Nx, Ny, Nz: 网格尺寸
        lambda_: 相关长度
        sigma_Eb: 目标标准差
        mean_Eb: 目标均值
        method: 'gaussian'或'exponential'生成方式
    返回:
        Eb_final: 归一化后的空间相关场
    """
    # 步骤1: 生成初始随机场
    Nx=int(np.ceil(Lx/spac))
    Ny=int(np.ceil(Ly/spac))
    Nz=int(np.ceil(Lz/spac))
    lambda_=lambda_/spac
    if method == 'phir_randn':
        phi0_r = np.random.randn(Nx, Ny, Nz)  # 实空间高斯噪声
        phi0_k = fftn(phi0_r)                 # 傅里叶变换到频域
    else:  # fft_phik
        phi0_k = np.exp(2 * np.pi * np.random.rand(Nx, Ny, Nz))
    
    # 步骤2: 生成k值网格
    kx = 2 * np.pi * fftfreq(Nx)
    ky = 2 * np.pi * fftfreq(Ny)
    kz = 2 * np.pi * fftfreq(Nz)
    k_grid = np.sqrt(kx[:,None,None]**2 + ky[None,:,None]**2 + kz[None,None,:]**2)
    
    # 步骤3: 应用高斯功率谱P(k)
    d = 3  # 维度
    if correlate_method == "gaussian":
        P_k = np.sqrt((lambda_**d * 2*np.pi)**2) * np.exp(-(k_grid*lambda_)**2 / 2)
    elif correlate_method == "exp":
        numerator = gamma((d + 1)/2) / (np.pi ** ((d + 1)/2))
        term1 = (2 * np.pi) ** d * (1/lambda_)
        denominator = ((1/lambda_)**2 + k_grid**2) ** ((d + 1)/2)
        P_k = numerator * term1 / denominator
    Eb_k = phi0_k * np.sqrt(P_k)
    
    # 步骤4: 逆变换到实空间
    Eb_r = ifftn(Eb_k).real
    
    # 截断高斯
    Eb_r[Eb_r < 0] = 0 

    # 步骤5: 归一化
    Eb_final = (Eb_r - np.mean(Eb_r)) / np.std(Eb_r) * sigma_Eb + mean_Eb
    
    return Eb_final

def plot_field_slices(field, cmap='viridis'):
    """绘制3D场的三个正交切片"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 取中间切片
    mid_x, mid_y, mid_z = [s//2 for s in field.shape]
    
    # XY平面切片
    axes[0].imshow(field[mid_x, :, :], cmap=cmap)
    axes[0].set_title('XY Plane (Z=mid)')
    
    # XZ平面切片
    axes[1].imshow(field[:, mid_y, :], cmap=cmap)
    axes[1].set_title('XZ Plane (Y=mid)')
    
    # YZ平面切片
    axes[2].imshow(field[:, :, mid_z], cmap=cmap)
    axes[2].set_title('YZ Plane (X=mid)')
    
    plt.colorbar(axes[0].images[0], ax=axes, fraction=0.02)
    plt.tight_layout()
    plt.show()

# def plot_3d_isosurface(field, n_levels=4):
#     """绘制3D等值面视图"""
#     fig = plt.figure(figsize=(10,8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     levels = np.linspace(field.min(), field.max(), n_levels+2)[1:-1]
#     colors = plt.cm.viridis(np.linspace(0,1,len(levels)))
    
#     for level, color in zip(levels, colors):
#         verts, faces, _, _ = marching_cubes(field, level)
#         mesh = ax.plot_trisurf(verts[:,0], verts[:,1], faces, verts[:,2],
#                               color=color, alpha=0.6, label=f'Level={level:.2f}')
    
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     ax.set_title('3D Isosurface Visualization')
#     ax.legend(title='Isovalues', bbox_to_anchor=(1.05,1), loc='upper left')
#     plt.tight_layout()

def plot_3d_isosurface(field, levels=7):
    """绘制3D等值面"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建网格
    x, y, z = np.mgrid[:field.shape[0], :field.shape[1], :field.shape[2]]
    
    # 计算等值面层级
    min_val, max_val = np.min(field), np.max(field)
    level_values = np.linspace(min_val, max_val, levels+2)[1:-1]
    
    # 绘制每个等值面
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
    """3D体积渲染"""
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建坐标网格
    x,y,z = np.indices(field.shape)
    
    # 阈值过滤显示范围
    threshold = np.percentile(field, 80)
    mask = field > threshold
    
    # 绘制体素
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
# # 示例使用
# if __name__ == "__main__":
#     Nx, Ny, Nz = 64, 64, 64
#     lambda_ = 5.0    # 相关长度
#     sigma_Eb = 2.0   # 目标标准差
#     mean_Eb = 10.0   # 目标均值
    
#     field = generate_correlated_field(Nx, Ny, Nz, lambda_, sigma_Eb, mean_Eb)
#     print(f"生成场形状: {field.shape}, 均值: {np.mean(field):.2f}, 标准差: {np.std(field):.2f}")
