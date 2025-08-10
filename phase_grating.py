import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as colors
from numpy.fft import ifftshift, fftshift, fft2, fft, ifft2
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.rcParams.update({'font.size': 24})
import utils
import ovmm

def imshow(mat, ax):
    im =ax.imshow(mat, aspect="auto", origin="upper")
    plt.colorbar(im)

def sinc(x):
    x = np.asarray(x)  # convert input to numpy array if it's not already
    # Use np.where to handle the zero case element-wise
    return np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))

def inverse_sinc(target):
    """Approximate the inverse sinc function over [0, 1] using interpolation."""
    x_vals = np.linspace(0, 1.0, 5000)  # avoid singularity at 0
    sinc_vals = sinc(x_vals)
    return np.interp(target, sinc_vals[::-1], x_vals[::-1])
def downsample(matrix, lpw):
    target_shape = np.asarray(matrix.shape)//lpw
    mat = np.zeros(target_shape)
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            mat[i,j] = np.mean(np.abs(matrix[i*lpw:(i+1)*lpw, j*lpw:(j+1)*lpw]), axis=None)
    return mat
# def generate_iris(r, order, grating_period=1, shape=(512, 512)):
#     cy, cx = np.array(shape) // 2 + np.array(order)*shape[0]/grating_period
#     Y, X = np.ogrid[:shape[0], :shape[1]]
#     return (X - cx)**2 + (Y - cy)**2 <= r**2

np.random.seed(100)
grating_period = 4
N = 4
matrix = 2*np.random.random_sample((N,N)) - 1
As = np.random.random_sample((N,N))
# As[0,0], As[0,1] = 1, 1
# As = np.ones((N,N))
Phis = np.random.choice([0,np.pi], (N,N))
lpw = 100
r = 80
slm_shape = np.asarray(matrix.shape)*lpw
fig, axes = plt.subplots(4,3, figsize=(30,25))

expected_As, expected_phis = np.zeros(slm_shape), np.zeros(slm_shape)
psis = np.zeros(slm_shape)
X, Y = np.mgrid[0:slm_shape[0], 0:slm_shape[1]]
for i in range(N):
    for j in range(N):
        block = np.zeros((lpw,lpw))
        XX, YY = X[i*lpw: (i+1)*lpw, j*lpw:(j+1)*lpw], Y[i*lpw: (i+1)*lpw, j*lpw:(j+1)*lpw]
        # XX, YY = np.mgrid[0:lpw, 0:lpw]
        M = 1 - inverse_sinc(As[i,j])
        F = Phis[i,j] + np.pi*(1-M)
        u0 = 2*np.pi / grating_period
        psi = M*np.mod(F+u0*(XX+YY), 2*np.pi)
        psis[i*lpw: (i+1)*lpw, j*lpw:(j+1)*lpw] = psi
        expected_As[i*lpw: (i+1)*lpw, j*lpw:(j+1)*lpw] = As[i,j]
        expected_phis[i*lpw: (i+1)*lpw, j*lpw:(j+1)*lpw] = Phis[i,j]

imshow(As, axes[0,0])
axes[0,0].set_title(r"$A(x,y)$")
imshow(Phis, axes[0,1])
axes[0,1].set_title(r"$\phi(x,y)$")
imshow(psis, axes[0,2])
axes[0,2].set_title(r"$\psi(x,y)$")


ffta = fftshift(1/(psis.shape[0])*fft2(ifftshift(
    0*np.exp(1j*psis) + 1*psis
    )))
iris = ovmm.generate_iris(r=r, order = [1,1], grating_period=grating_period, shape=slm_shape)
# iris += ovmm.generate_iris(r=r, order = [-1,-1], grating_period=4, shape=slm_shape)
imshow(np.log1p(np.abs(ffta)), axes[1,1])
axes[1,1].contour(iris, levels=[0.5], colors='red', linewidths=1)
# axes[1,1].set_title(r"Fourier Transform with an Iris")

fft_mid = ffta*iris
imshow(np.log1p(np.abs(fft_mid)), axes[1,2])
fftb= fftshift(1/(psis.shape[0])*fft2(ifftshift(fft_mid)))
fftb = np.flip(fftb)
d_psis = downsample(psis, lpw)
d_fftb = downsample(fftb, lpw)


def downsample_complex(matrix, lpw):
    target_shape = np.asarray(matrix.shape) // lpw
    mat = np.zeros(target_shape, dtype=complex)
    for ii in range(target_shape[0]):
        for jj in range(target_shape[1]):
            block = matrix[ii*lpw:(ii+1)*lpw, jj*lpw:(jj+1)*lpw]
            mat[ii,jj] = np.mean(block)         # complex mean preserves phase
    return mat

d_fftb_complex = downsample_complex(fftb, lpw)

# compare block phases to Phis
recovered_block_phases = np.mod(np.angle(d_fftb_complex),(np.pi+1))
expected_block_phases = np.mod( expected_phis[::lpw, ::lpw], 2*np.pi )  # sample one pixel per block

phase_err = np.mod(recovered_block_phases - expected_block_phases + np.pi, 2*np.pi) - np.pi   # signed diff in [-pi,pi]
print("max phase error (rad):", np.max(np.abs(phase_err)))

imshow(recovered_block_phases, axes[3,0])
imshow(expected_block_phases, axes[3,1])
axes[3,2].scatter(range(recovered_block_phases.size), recovered_block_phases.flatten())
axes[3,2].grid("on")

imshow(np.abs(fftb), axes[2,0])
imshow(np.abs(d_fftb), axes[2,1])
imshow(np.mod(np.angle(fftb), 2*np.pi), axes[2,2])

x_vals = np.linspace(0, 1.0, 5000)  # avoid singularity at 0
sinc_vals = np.sinc(x_vals)
axes[1,0].plot(x_vals, sinc_vals)
plt.savefig("/ysinha/projects/Reports/OVMM/figs/phase_visualization.png")
print(np.linalg.norm(np.abs(d_fftb)-As))




