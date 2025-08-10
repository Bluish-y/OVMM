import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def blaze_first_order_amplitude(phase_depth):
    """
    Amplitude of 1st diffraction order for a blaze grating of given phase depth.
    Uses sinc formula where numpy.sinc(x) = sin(pi*x)/(pi*x).
    phase_depth in radians.
    """
    x = (phase_depth / 2) - np.pi
    return np.abs(np.sinc(x / np.pi))

# Precompute inverse mapping amplitude -> phase_depth
phase_depths = np.linspace(0, 2*np.pi, 1000)
amps = blaze_first_order_amplitude(phase_depths)
amp_to_depth_interp = interp1d(amps, phase_depths, bounds_error=False, fill_value=(phase_depths[0], phase_depths[-1]))

def generate_slm_pattern_blaze(As, phis, lpw, period):
    nrows, ncols = As.shape
    slm_rows = nrows * lpw
    slm_cols = ncols * lpw
    slm_pattern = np.zeros((slm_rows, slm_cols))

    u = np.arange(lpw)
    v = np.arange(lpw)
    U, V = np.meshgrid(u, v, indexing='xy')

    for i in range(nrows):
        for j in range(ncols):
            A_target = np.clip(As[i, j], 0, np.max(amps))  # Clamp amplitude
            phi_target = phis[i, j]

            # Get blaze depth for target amplitude
            depth = amp_to_depth_interp(A_target)

            # Create 2D blaze grating along diagonal (U+V)
            phase_ramp = (U + V) * (depth / period)  # Linear ramp over lpw pixels
            local_phase = (phase_ramp + phi_target) % (2*np.pi)

            r_start = i * lpw
            c_start = j * lpw
            slm_pattern[r_start:r_start+lpw, c_start:c_start+lpw] = local_phase

    return slm_pattern

def apply_iris_filter(ft, radius=1, period=4):
    """
    Apply circular iris in Fourier domain.
    ft: 2D Fourier transform magnitude or complex array
    radius_frac: radius as fraction of half the smaller dimension
    """
    ny, nx = ft.shape
    cy, cx = ny // 2 + 0.5*ny//period, nx // 2 + 0.5*nx//period
    y, x = np.ogrid[:ny, :nx]
    mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
    return ft * mask, mask

def main():
    # Parameters
    lpw = 60  # pixels per SLM pixel block
    period = 4
    nrows, ncols = 3,3

    # Example amplitude and phase matrices (random for demo)
    np.random.seed(42)
    As = np.clip(np.random.rand(nrows, ncols), 0, 1)
    phis = np.random.choice([0,np.pi],(nrows, ncols))

    # Generate SLM pattern with blazed gratings
    slm_pattern = generate_slm_pattern_blaze(As, phis, lpw, period)

    # Calculate Fourier transform (centered)
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j*slm_pattern))))

    # Apply iris filter on first order
    ft_filtered, iris_mask = apply_iris_filter(ft, radius=20, period=period)

    # Inverse FT to get output field
    output_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ft_filtered)))

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Amplitude matrix As
    im0 = axs[0,0].imshow(As, cmap='viridis', vmin=0, vmax=1)
    axs[0,0].set_title('Target Amplitude $A_s$')
    fig.colorbar(im0, ax=axs[0,0])

    # Phase matrix phis
    im1 = axs[0,1].imshow(phis, cmap='hsv', vmin=0, vmax=2*np.pi)
    axs[0,1].set_title('Target Phase $\phi_s$')
    fig.colorbar(im1, ax=axs[0,1], ticks=[0, np.pi, 2*np.pi], format=lambda x, _: f"{x/np.pi:.1f}π")

    # SLM phase pattern
    im2 = axs[0,2].imshow(slm_pattern, cmap='hsv', vmin=0, vmax=2*np.pi)
    axs[0,2].set_title('SLM Phase Pattern')
    fig.colorbar(im2, ax=axs[0,2], ticks=[0, np.pi, 2*np.pi], format=lambda x, _: f"{x/np.pi:.1f}π")

    # Fourier magnitude with iris overlay
    axs[1,0].imshow(np.log1p(np.abs(ft)), cmap='inferno')
    axs[1,0].contour(iris_mask, colors='cyan', linewidths=1)
    axs[1,0].set_title('Fourier Magnitude with Iris')

    # Output amplitude after iris
    im4 = axs[1,1].imshow(np.abs(output_field), cmap='viridis')
    axs[1,1].set_title('Output Amplitude')
    fig.colorbar(im4, ax=axs[1,1])

    # Output phase after iris
    im5 = axs[1,2].imshow(np.angle(output_field), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axs[1,2].set_title('Output Phase')
    fig.colorbar(im5, ax=axs[1,2], ticks=[-np.pi, 0, np.pi], format=lambda x, _: f"{x/np.pi:.1f}π")

    # for ax in axs.flat:
    #     ax.axis('off')

    plt.tight_layout()
    plt.savefig("/data.nst/ysinha/projects/Reports/OVMM/test.png")

if __name__ == "__main__":
    main()
