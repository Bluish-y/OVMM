import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fft2, ifftshift
from scipy.ndimage import zoom

def snap_to_zero_np(arr, tol=1e-15):
    arr = arr.copy()
    arr.real[np.abs(arr.real) < tol] = 0
    arr.imag[np.abs(arr.imag) < tol] = 0
    return arr

class DMD:
    def __init__(self, vector, n_rows, logical_pixel_width, dmd_dims = [1024,768]):
        self.dmd_dims = np.asarray(dmd_dims)
        self.setup_time = 1.25e-5 # in seconds
        self.pixel_pitch = 1.368e-5  # in meters
        self.mirrors = np.zeros(self.dmd_dims) # I am ignoring that the POM (pond of micromirrors) on the border are electrically limited for simplicity
        self.nrows = n_rows
        self.ncols = vector.size
        self.block_width = logical_pixel_width
        self.pattern = self._generate_pattern(vector)
        # self.array_size = [1.0506e-2, 1.4008e-2] # in meters
        self.array_size = self.used_dmd_dims*self.pixel_pitch  # in meters
    
    def _generate_pattern(self, vector):
        """Generate a pattern based on the input vector and matrix dimensions."""
        nrows, ncols = self.nrows, self.ncols
        pattern = np.zeros((nrows, ncols, self.block_width, self.block_width))
        for i, value in enumerate(vector):
            num_pixels = int(np.round(value * self.block_width))
            block = np.zeros((self.block_width, self.block_width))
            block[:,:num_pixels] = 1
            pattern[:, i, :, :] = block  # broadcast to all rows
        self.used_dmd_dims = np.asarray([nrows*self.block_width, ncols*self.block_width])
        return pattern

    def propagate(self, E0):
        """The number of pixels turned on in each block is determined by the vector elements."""
        mirrors = np.zeros(self.used_dmd_dims)
        for i in range(self.nrows):
            for j in range(self.ncols):
                row_start = i*self.block_width
                row_end = row_start + self.block_width
                col_start = j*self.block_width
                col_end = col_start + self.block_width
                if i==self.nrows-1:
                    mirrors[row_start:row_end,col_start:col_end] = E0
                else:
                    mirrors[row_start:row_end,col_start:col_end] = E0*self.pattern[i,j]
        return mirrors


class Lens:
    def __init__(self, focal_length=1.0):
        self.f = focal_length
    
    def transform(self, field):
        N = field.shape[0]
        return fftshift(1/N*fft2(ifftshift(field)))

    def forward(self, field, dx, lambda0=1.0):
        """
        field: input complex field (NxN)
        dx: sampling interval in input plane
        Returns: (output_field, dx_out)
        """
        N = field.shape[0]
        # FFT simulates propagation through lens + free space to Fourier plane
        field_ft = self.transform(field)
        
        # Scale output sampling interval
        df = 1 / (N * dx)
        self.dx_out = lambda0 * self.f * df  # physical coordinate spacing in output plane

        return field_ft, self.dx_out

class CylindricalLens(Lens):
    def __init__(self, focal_length=1.0, axis=1):
        super().__init__(focal_length)
        self.axis = axis

    def transform(self, input_field):
        return fftshift(1/np.sqrt(input_field.shape[0])*fft(ifftshift(input_field, axes=self.axis), axis=self.axis), axes=self.axis)

class FourFSystem:
    def __init__(self, lens1, lens2, input_field, iris=None, dx_in=1.0):
        self.lens1 = lens1
        self.lens2 = lens2
        self.N = input_field.shape[0]
        assert self.N == input_field.shape[1], "Insert a square shaped input!"

        self.dx_in = dx_in
        self.input_field = input_field
        self.iris = iris  # iris effect (masking)

    def propagate(self, lambda0=1.0):
        # First lens: FFT + update dx
        field1, dx1 = self.lens1.forward(self.input_field, self.dx_in, lambda0)
        # Apply iris effect if present
        if self.iris is not None:
            field2 = field1*self.iris
        else:
            field2 = field1
        # Second lens: another FFT + update dx
        field3, dx2 = self.lens2.forward(field2, dx1, lambda0)
        field3 = snap_to_zero_np(field3, tol=1e-15)

        # Final output field and physical sampling
        return field3, field1, dx2

    def get_grid(self, dx):
        x = np.linspace(-self.N//2, self.N//2 - 1, self.N) * dx
        return np.meshgrid(x, x)

class SLM:
    def __init__(self, matrix, logical_pixel_width, pp_in = 1.5e-5, slm_dims=[512, 512], grating_period = 4):
        self.slm_dims = np.asarray(slm_dims)
        self.input_pp = pp_in  # pixel pitch of the imaged vector in meters
        self.pixel_pitch = 1.5e-5 # in meters
        self.lpw = logical_pixel_width

        self.block_width = slm_dims[0] // matrix.shape[0]
        self.u0 = 2*np.pi/grating_period  # spatial frequency for diagonal blaze
        self.grating_period = grating_period
        self.psis = self._generate_modulations(matrix)
    
    def propagate(self, input_field):
        """
        Propagate the input field through the SLM, applying the pixel modulations.
        """
        # Resample input field to match SLM pixel pitch
        assert input_field.shape == self.psis.shape, f"Shape mismatch: input.shape ={input_field.shape} and Psis.shape={self.psis.shape}"
        # Multiply with pixel modulations
        self.output = input_field * np.exp(1j*self.psis)
        
        return self.output

    def _generate_modulations(self, matrix):
        psis = np.zeros(self.slm_dims)
        phis = np.zeros_like(psis)
        num_rows, num_cols = matrix.shape

        for i in range(0,num_rows):
            for j in range(0,num_cols):
                A = matrix[i, j]  # amplitude value in [0, 1]

                # Define block location
                row_start = i * self.block_width
                row_end = row_start + self.block_width
                col_start = j * self.block_width
                col_end = col_start + self.block_width

                if row_end > self.slm_dims[0] or col_end > self.slm_dims[1]:
                    continue  # stay within bounds

                # Create meshgrid for diagonal coordinates
                x = np.linspace(0, self.block_width - 1, self.block_width)
                y = np.linspace(0, self.block_width - 1, self.block_width)
                X, Y = np.meshgrid(x, y)
                

                # Compute correction factors
                M = 1 - self._inverse_sinc(np.abs(A))  # amplitude correction
                # M=1
                phi = 0 if A>=0 else np.pi # phase for real values
                F = phi + 0*np.pi*(1-M)  # phase correction

                # Compute diagonal blazed grating
                Phi = np.mod(F + self.u0*(X + Y), 2*np.pi)
                Phi[np.isclose(Phi, 2*np.pi, atol=1e-12)] = 0.0
                psi = M * Phi
                
                # Assign to SLM
                psis[row_start:row_end, col_start:col_end] = 1*psi
                phis[row_start:row_end, col_start:col_end] = phi
        self.phis = phis
        return psis

    def _inverse_sinc(self, target):
        """Approximate the inverse sinc function over [0, 1] using interpolation."""
        x_vals = np.linspace(0, 1.0, 1000)  # avoid singularity at 0
        sinc_vals = np.sinc(x_vals)
        return np.interp(target, sinc_vals[::-1], x_vals[::-1])
    

class Mirror:
    def __init__(self):
        pass
    def forward(self,field):
        return np.flip(field)

class BeamSplitter:
    def __init__(self, R=0.5, ):
        if not (0 <= R <= 1):
            raise ValueError("Reflectivity must be between 0 and 1.")
        self.R = R
        self.T = 1 - R

        self.t = np.sqrt(self.T)
        self.r = 1j * np.sqrt(self.R) #Assuming symmetric phase coefficients in a 50:50 lossless beam-splitter

    def forward(self, Ein1, Ein2=0):
        Eout1 = self.t * Ein1 + self.r * Ein2
        Eout2 = self.r * Ein1 + self.t * Ein2
        return Eout1, Eout2

class PiezoMirror:
    def __init__(self, wavelength):
        self.lambda0 = wavelength
        self.max_displacement = 2e-5 # in meters

    def forward(self, field, displacement):
        phi = np.pi + (4 * np.pi / self.lambda0) * displacement
        return field * np.exp(1j*phi)

class Interferometer:
    def __init__(self):
        pass

    def combine(self, E1, E2):
        return E1+E2

class Camera:
    def __init__(self):
        pass

    def record(self, field):
        return np.abs(field)**2

class OVMMSystem:
    def __init__(self, vector, matrix):
        self.dmd = DMD(vector)
        self.slm = SLM(matrix)
        self.lens = CylindricalLens()
        self.interferometer = Interferometer()
        self.camera = Camera()

    def run(self, use_interferometry=True, visualize=True):
        input_pattern = self.dmd.modulate()
        modulated_field = self.slm.modulate(input_pattern)

        # Apply Fourier transform (cylindrical lens effect)
        output_field = np.array([self.lens.transform(row) for row in modulated_field])

        if use_interferometry:
            detected = np.array([self.interferometer.apply(row) for row in output_field])
        else:
            detected = np.array([self.camera.record(row) for row in output_field])

        if visualize:
            plt.figure(figsize=(10, 4))
            plt.imshow(detected, cmap='inferno', aspect='auto')
            plt.title("Camera Output (Intensity of Each Row After Fourier Transform)")
            plt.xlabel("Fourier Axis (pixels)")
            plt.ylabel("Row (Dot Product Output Index)")
            plt.colorbar()
            plt.show()

        return detected

def zoom_to_shape(arr, target_shape, order=3):
    input_shape = arr.shape
    zoom_factors = [t / s for t, s in zip(target_shape, input_shape)]
    return zoom(arr, zoom_factors, order=order)