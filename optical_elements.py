# Import necessary libraries
from globals import *
import functions_gpu
import torch
import copy

# Define the base class for all optical elements
class OpticalElement:
    # The propagate method should be implemented by each subclass
    def propagate(self, field_in):
        raise NotImplementedError  # Raise an error if this method is called



# Define the Lens class, which is a type of OpticalElement
class Lens(OpticalElement):
    # The constructor takes the focal length, lens radius, and lens center as arguments
    # The default lens center is at (0, 0)
    def __init__(self, focal_length, lens_radius, name, lens_center=(0, 0)):
        self.focal_length = focal_length  # Store the focal length
        self.lens_radius = lens_radius  # Store the lens radius
        self.lens_center = lens_center  # Store the lens center
        self.name = name


    # The propagate method applies the lens phase to the field
    def propagate(self, field_in, reverse=False):
        # Calculate the wave number
        k0 = 2 * torch.pi / field_in.wavelength
        # Create a meshgrid of the field's coordinates
        xx, yy = torch.meshgrid(field_in.y_coordinates, field_in.x_coordinates)

        # Create a lens shape based on the lens radius and center
        lens_shape = torch.zeros(len(field_in.y_coordinates), len(field_in.x_coordinates), dtype=torch.int).to(device)
        lens_shape[((xx - self.lens_center[0]) ** 2 + (yy - self.lens_center[1]) ** 2) < self.lens_radius ** 2] = 1

        # Calculate the lens phase
        lens_phase = torch.exp(-1j * k0 / (2 * self.focal_length) * ((xx - self.lens_center[0]) ** 2 + (yy - self.lens_center[1]) ** 2))
        # Set the phase to 0 outside the lens
        lens_phase[((xx - self.lens_center[0]) ** 2 + (yy - self.lens_center[1]) ** 2) > self.lens_radius ** 2] = 0

        # Apply the lens phase to the field
        field_out = copy.deepcopy(field_in)
        field_out.field = field_in.field * lens_shape * lens_phase

        # Return the propagated field
        return field_out


# Define the Iris class, which is a type of OpticalElement
class Iris(OpticalElement):
    # The constructor takes the radius and center position of the iris as arguments
    # The default iris center is at (0, 0)
    def __init__(self, radius, name, iris_center=(0, 0)):
        self.radius = radius  # Store the radius
        self.iris_center = iris_center  # Store the center position
        self.name = name


    # The propagate method applies the iris aperture to the field
    def propagate(self, field_in, reverse=False):
        # Create a meshgrid of the field's coordinates
        xx, yy = torch.meshgrid(field_in.y_coordinates, field_in.x_coordinates)

        # Create a circular aperture
        circ_iris = xx * 0
        circ_iris[((xx - self.iris_center[0]) ** 2 + (yy - self.iris_center[1]) ** 2) < self.radius ** 2] = 1

        # Apply the aperture to the field
        field_out = copy.deepcopy(field_in)
        field_out.field = field_in.field * circ_iris

        # Return the propagated field
        return field_out


# Define the FreeSpace class, which is a type of OpticalElement
# Define the FreeSpace class, which is a type of OpticalElement
class FreeSpace(OpticalElement):
    # The constructor takes the length of the free space as an argument
    def __init__(self, length, name):
        self.length = length  # Store the length of the free space
        self.name = name


    # The propagate method propagates the field through the free space
    def propagate(self, field_in, reverse=False):
        # Calculate the wave number
        k0 = 2 * torch.pi / field_in.wavelength

        # Calculate the distance to propagate
        z_in = field_in.z
        z_out = z_in - self.length if reverse else z_in + self.length

        # Get the dimensions of the field
        dim_y, dim_x = torch.tensor(field_in.field.shape).to(device)
        # Get the step size
        step = field_in.step

        # Calculate the frequency components of the field
        fx = torch.fft.fftshift(torch.fft.fftfreq(dim_x, step)).to(device)
        fy = torch.fft.fftshift(torch.fft.fftfreq(dim_y, step)).to(device)

        # Calculate the Fourier transform of the field
        angular_0 = torch.fft.fftshift(torch.fft.fft2(field_in.field)).to(device)

        # Create a meshgrid of the frequency components
        fxfx, fyfy = torch.meshgrid(fy, fx)
        # Calculate the transfer function for free space propagation
        transfer_function = torch.exp(k0 * 1j * self.length * torch.sqrt(1 - (field_in.wavelength * fxfx) ** 2 - (field_in.wavelength * fyfy) ** 2))
        # Set the transfer function to 0 outside the unit circle
        transfer_function[(fxfx ** 2 + fyfy ** 2) > 1 / (field_in.wavelength ** 2)] = 0

        # Apply the transfer function to the Fourier transform of the field
        angular_z = angular_0 * transfer_function

        # Create a copy of the field at the new position
        field_out = copy.deepcopy(field_in)
        field_out.z = z_out
        field_out.field = 0 * field_out.field

        # Calculate the inverse Fourier transform of the propagated field
        temp_field = torch.fft.ifft2(torch.fft.ifftshift(angular_z))

        # Update the field with the propagated field
        field_out.field = temp_field

        # Return the propagated field
        return field_out




# Define the OpticalSystem class, which contains a list of OpticalElements
class OpticalSystem:
    def __init__(self, elements):
        """
        Initialize an optical system with a list of optical elements.

        Parameters
        ----------
        elements : list
            A list of optical elements that make up the system.
        """
        self.elements = elements

    def propagate(self, field_in, imaging_method=None, plot_fields=False, reverse=False):
        """
        Propagate a field through the optical system.

        Parameters
        ----------
        field_in : Field
            The initial field to propagate through the system.
        imaging_method : str, optional
            The imaging method being used. This is used to name the final field.
        plot_fields : bool, optional
            Whether to plot the initial and final fields. Default is False.
        reverse : bool, optional
            Whether to propagate the field in reverse order through the elements. Default is False.

        Returns
        -------
        dict
            A dictionary where the keys are the names of the fields at each step of the propagation,
            and the values are the corresponding Field objects.
        """
        # Start with the initial field
        field_states = {}
        # field_states = {field_in.name: field_in}

        # Plot the initial field, if requested
        if plot_fields:
            title = 'field in - ' + field_in.name
            functions_gpu.plot_field(field_in, title=title)

        # Reverse the order of the elements if reverse is True
        elements = self.elements[::-1] if reverse else self.elements

        for i, element in enumerate(elements):
            # Propagate the field through the current element
            field_out = element.propagate(field_in, reverse=reverse)

            # Generate the name for the output field
            try:
                var_no_of_field_in = int(field_in.name[1:3])
            except ValueError:
                raise ValueError(f"Field name '{field_in.name}' doesn't follow the expected format 'vXX...'")
            var_no_of_field_out = var_no_of_field_in + 1

            # If this is the last element in the system and an imaging method is provided,
            # name the output field accordingly
            if i == len(self.elements) - 1 and imaging_method is not None:
                if reverse:
                    if var_no_of_field_out > 9:
                        name = 'v' + str(var_no_of_field_out) + '_field_at_obj_plane'
                    else:
                        name = 'v0' + str(var_no_of_field_out) + '_field_at_obj_plane'
                else:
                    if var_no_of_field_out > 9:
                        name = 'v' + str(var_no_of_field_out) + '_' + imaging_method
                    else:
                        name = 'v0' + str(var_no_of_field_out) + '_' + imaging_method
            else:
                if var_no_of_field_out > 9:
                    name = 'v' + str(var_no_of_field_out) + '_field_after_' + element.name
                else:
                    name = 'v0' + str(var_no_of_field_out) + '_field_after_' + element.name

            # Set the name of the output field
            field_out.name = name

            # Save the field state after the current element
            field_states[name] = field_out

            # The output field becomes the input field for the next element
            field_in = field_out

        # Plot the final field, if requested
        if plot_fields:
            functions_gpu.plot_field(field_out)

        return field_states
