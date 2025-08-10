import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as colors
from numpy.fft import ifftshift, fftshift
from scipy.optimize import curve_fit

import utils

# Constants
wavelength = 400e-9        # wavelength in meters (He-Ne laser)
k = 2 * np.pi / wavelength


def draw_letter(letter='B', size=(50, 50), font_size=40):
    image = Image.new('L', size, color=0)  # Black background
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Get bounding box of the letter
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    draw.text(position, letter, fill=255, font=font)

    # Convert to binary matrix
    matrix = np.array(image)
    binary_matrix = (matrix > 50).astype(int)
    return binary_matrix

def custom_imshow(ax, matrix, title, dx=1, scale=None):
    if scale is None:
        im = ax.imshow(matrix, aspect="auto", origin="upper",
                    extent=[-dx/2, (matrix.shape[0]+.5)*dx, matrix.shape[1]*dx+.5*dx, 0.5*dx])
    elif scale =="log":
        im = ax.imshow(matrix, aspect="auto", origin="upper",
                    extent=[-dx/2, (matrix.shape[0]+.5)*dx, matrix.shape[1]*dx+.5*dx, 0.5*dx], norm = colors.LogNorm())
    ax.set_title(title)
    plt.colorbar(im)

def generate_iris(r, order, grating_period=1, shape=(512, 512)):
    cy, cx = np.array(order)*shape[0]/grating_period
    Y, X = np.ogrid[:shape[0], :shape[1]]
    return fftshift((X - cx)**2 + (Y - cy)**2 <= r**2)

def block_sum_downsample(mat, block_size=(60, 60), over_phase = False):
    M, N = mat.shape
    m, n = block_size
    assert M % m == 0 and N % n == 0, "Matrix shape must be divisible by block size"
    matrix = np.zeros((M//m, N//n))
    for i in range(M//m):
        for j in range(N//n):
            if over_phase:
                matrix[i,j] = np.mean(np.angle(mat[i*m:i*m+m, j*n:j*n+n]))
            else:
                matrix[i,j] = np.mean(np.abs(mat[i*m:i*m+m, j*n:j*n+n]))
    return matrix

def repeat_vector(vector, nrows, E0):
    mat = vector.reshape(1,-1).repeat(nrows, axis=0)
    mat[-1,:] = 1
    return E0*mat

# Model function for fitting
def cosine_model(delta, A, B, phi):
    return A + B * np.cos((4 * np.pi / wavelength) * delta + phi)


def extract_phase_from_cosine(all_outputs, deltas):
    num_steps, H, W = all_outputs.shape
    phase_map = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            I_vals = all_outputs[:, i, j]
            guess = [np.mean(I_vals), (np.max(I_vals) - np.min(I_vals)) / 2, 0]
            try:
                popt, _ = curve_fit(cosine_model, deltas, I_vals, p0=guess)
                _, _, phi = popt
                phase_map[i, j] = phi
            except RuntimeError:
                phase_map[i, j] = np.nan

    return phase_map

if __name__ == "__main__":
    # Input vector and weight matrix
    # vector = [0.2, 0.5, -0.4, 0.1]

    lens1a,lens1b = utils.Lens(1), utils.Lens(1.0)
    lens2a, lens2b = utils.Lens(1), utils.Lens(1)
    lens3a, lens3b, lens3c = utils.CylindricalLens(1, axis=1), utils.CylindricalLens(1, axis=0), utils.CylindricalLens(1, axis=1)
    grating_period = 4
    matrix_shape, slm_shape, dmd_shape = (6,6),[360,360],[360,360]
    lpw = 60
    iris = generate_iris(r=80, order = [1,1], grating_period=grating_period, shape=slm_shape)
    mirror = utils.Mirror()
    beam_splitter = utils.BeamSplitter(R=0.5)
    piezo_mirror = utils.PiezoMirror(wavelength = wavelength)
    interferometer = utils.Interferometer()
    camera = utils.Camera()
    E0 = 1

    ####### DMD
    np.random.seed(42) 
    vector = np.random.random_sample((matrix_shape[1],1))
    dmd = utils.DMD(vector, n_rows=matrix_shape[0], logical_pixel_width=lpw, dmd_dims=dmd_shape)
    dmd_output = dmd.propagate(E0)

    ####### 4f imaging
    four_f1 = utils.FourFSystem(lens1a, lens1b, input_field = dmd_output, dx_in = 1.0, iris=None)
    bs_inp, _, dx_out = four_f1.propagate(lambda0=wavelength)
    dx_out_dmd = np.copy(dx_out)

    bs_inp = mirror.forward(bs_inp)

    ######## BS
    slm_inp, Eref = beam_splitter.forward(bs_inp)

    ####### SLM
    matrix = draw_letter('B', matrix_shape)
    matrix[0,0] = -0.5
    slm = utils.SLM(matrix, logical_pixel_width = lpw, slm_dims = slm_shape)
    slm_output = slm.propagate(slm_inp)

    
    # Store results for all displacements
    all_outputs = []

    deltas = np.linspace(0, piezo_mirror.max_displacement, 1)
    expected_hadamard = np.abs(matrix*repeat_vector(vector, matrix_shape[0], E0)).astype(float)
    expected_output = np.zeros_like(expected_hadamard)
    col_sum = expected_hadamard.sum(axis=1)
    expected_output[:, 3] = col_sum

    for delta in deltas:
        Eref = piezo_mirror.forward(Eref, delta)
        Eout = interferometer.combine(slm_output, Eref)

        # Spatial Filter
        four_f2 = utils.FourFSystem(lens2a, lens2b, input_field=Eout, dx_in=dx_out, iris=iris)
        fourier_inp, diffraction, dx_out = four_f2.propagate(lambda0=wavelength)
        dx_out_interferometer = np.copy(dx_out)

        # Fix spatial flips
        fourier_inp = mirror.forward(fourier_inp)

        ### 
        d_finp = block_sum_downsample(fourier_inp, (lpw,lpw))
        error = np.linalg.norm(expected_hadamard-d_finp)
        print("Error for delta:", delta, "is:", error)

        # Fourier Transform
        # test = np.asarray([[1,1,1], [1,1,1], [1,1,0]])
        camera_inp, dx_out = lens3a.forward(fourier_inp, dx = dx_out)
        dx_out_camera = np.copy(dx_out)
        # four_f3 = utils.FourFSystem(lens3b, lens3c, input_field=imaging_inp, dx_in=dx_out)
        # camera_inp, _, dx_out = four_f3.propagate(lambda0=wavelength)
        # dx_out_camera = np.copy(dx_out)

        # Camera
        output = camera.record(camera_inp)

        output = ifftshift(output)[:,0]

        # # Downsample digitally
        # d_output = block_sum_downsample(output[:,slm_shape[0]//2].reshape(-1,1), block_size=(lpw, 1))


        # all_outputs.append(d_output)
    
    ##### Extract phase
    # all_outputs = np.array(all_outputs)
    # phis = extract_phase_from_cosine(all_outputs, deltas)

    print("Using r:", 82)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12,12))
    custom_imshow(axes[0,0],matrix, title="Matrix", dx=1)
    custom_imshow(axes[0,1],dmd_output, title="DMD output", dx = dx_out_dmd, scale = None)
    custom_imshow(axes[0,2],expected_hadamard, title="Expected Hadamard", dx = 1)
    axes[0,1].vlines(ymin=0.5, ymax=360, x=np.asarray([1, 2, 3, 4, 5, 6])*lpw, color="red")

    custom_imshow(axes[1,0],slm.psis, title="SLM Phase Output", dx=dx_out_dmd)
    custom_imshow(axes[1,1],np.angle((fourier_inp)), title="(Last) Interferometer Output", dx = dx_out_interferometer)
    custom_imshow(axes[1,2], np.abs(d_finp).astype(float), title="(Last) Downsampled Interferometer Output", dx = dx_out_interferometer)

    custom_imshow(axes[2,0],np.abs(np.fft.fft(fourier_inp, axis=1)), title="(Last) Fourier Lens output")
    # custom_imshow(axes[2,1],output, title = "(Last) camera output")
    # custom_imshow(axes[2,2],output[:,slm_shape[0]//2].reshape(-1,1), title="Digital slice of zeroth component")

    # for i in range(5):
    #     axes[3,0].plot(deltas, all_outputs[:,i, 0])
    # custom_imshow(axes[3,1], phis.reshape(-1,1), title="Digital slice of phi map")
    # custom_imshow(axes[2,1], expected_output, title="Expected camera output")
    # custom_imshow(axes[3,0], phis, title="Final Phase Map")
    # custom_imshow(axes[3,1], d_output, title  = "Downsampled camera output")
    # print(vector)


    # # Create meshgrid for diagonal coordinates
    # x = np.linspace(0, 60 - 1, 60)
    # y = np.linspace(0, 60 - 1, 60)
    # X, Y = np.meshgrid(x, y)
    # # print(X.shape, Y.shape)

    # # Compute correction factors
    # # M = 1 - 1/np.pi*self._inverse_sinc(A)  # amplitude correction
    # M=1
    # phi = 0 # phase for real values
    # F = phi + (1 - M) * np.pi  # phase correction

    # # Compute diagonal blazed grating
    # Phi = np.mod(F + (2*np.pi/4)*(X + Y), 2*np.pi)
    # Phi[np.isclose(Phi, 2*np.pi, atol=1e-12)] = 0.0
    # psi = M * Phi
    # print(0 if np.isclose(np.mod(2*np.pi/4*44, 2*np.pi), 2*np.pi, atol=1e-12) else np.mod(2*np.pi/4*44, 2*np.pi))

    # custom_imshow(axes[2,1], psi, title="single grating block")
    plt.savefig("/data.nst/ysinha/projects/Reports/OVMM/slm_test.png")
    print(np.angle((fourier_inp)))
