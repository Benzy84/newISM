import numpy as np
import propagators as prop
import functions_gpu
import copy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Field:
    def __init__(self, field):
        self.field = field.to(device)
        self.x_coordinates = torch.tensor([])
        self.y_coordinates = torch.tensor([])
        self.z = torch.tensor([])
        self.mesh = torch.tensor([])
        self.extent = torch.tensor([])
        self.length_x = torch.tensor([])
        self.length_y = torch.tensor([])
        self.padding_size = torch.tensor([])
        self.step = torch.tensor([])



class System(object):
    def __init__(self):
        pass


def test(field, _00_system, padding_size, _04_padded_filtered_field):

    fields = []
    dim_y, dim_x = field.field.shape
    num_x_points = 3
    num_of_x_intervals = num_x_points + 1
    num_of_pix_in_x_interval = (dim_x - num_x_points) / num_of_x_intervals
    num_of_pix_in_x_interval = np.ceil(num_of_pix_in_x_interval)
    dim_x = num_of_x_intervals * num_of_pix_in_x_interval + num_x_points
    dim_x = int(dim_x)

    num_y_points = 3
    num_of_y_intervals = num_y_points + 1
    num_of_pix_in_y_interval = (dim_y - num_y_points) / num_of_y_intervals
    num_of_pix_in_y_interval = np.ceil(num_of_pix_in_y_interval)
    dim_y = num_of_y_intervals * num_of_pix_in_y_interval + num_y_points
    dim_y = int(dim_y)

    del num_of_x_intervals, num_of_y_intervals, dim_x, dim_y

    column = 0
    idx = 0
    for x_point in np.arange(num_x_points):
        column += num_of_pix_in_x_interval
        row = 0
        for y_point in np.arange(num_y_points):
            fields.append(copy.deepcopy(field))
            fields[idx].field = np.zeros(field.field.shape)
            row += num_of_pix_in_y_interval
            # print([int(row), int(column)])
            fields[idx].field[int(row), int(column)] = 1
            fields[idx].field = gaussian_filter(fields[idx].field, sigma=np.sqrt(1))
            fields[idx] = functions_gpu.pad(fields[idx], padding_size)
            row += 1
            idx += 1
        column += 1




    """
    
    to_show = fields[idx]
    fig0, ax0 = plt.subplots()
    im0 = ax0.imshow(to_show.field, extent=to_show.extent)
    ax0.set_title('initial field')
    plt.show(block=False)
    del fig0, ax0, im0
    
    """


    for test_num in np.arange(9):

        fig, axs = plt.subplots(2, 2)
        field_in = copy.deepcopy(fields[test_num])
        z_out = field_in.z + _00_system.u
        field_before_lens = prop.distance_z(field_in, z_out, _00_system.wave_length, plot=0)

        ax00 = axs[0, 0]
        ax00.set_title('field_in')
        im00 = ax00.imshow(np.abs(field_in.field), extent=field_in.extent)

        ax11 = axs[0, 1]
        ax11.set_title('field_before_lens')
        im11 = ax11.imshow(np.abs(field_before_lens.field), extent=field_before_lens.extent)

        # Lens 1
        field_in = copy.deepcopy(field_before_lens)
        field_after_lens = prop.thin_lens(field_in, _00_system.wave_length, _00_system.lens_radius, _00_system.lens_center_pos, _00_system.f, plot=0)

        ax11 = axs[1, 0]
        ax11.set_title('field_after_lens')
        im11 = ax11.imshow(np.abs(field_after_lens.field), extent=field_after_lens.extent)

        # propagation v
        field_in = copy.deepcopy(field_after_lens)
        z_out = field_in.z + _00_system.v
        field_at_image = prop.distance_z(field_in, z_out, _00_system.wave_length)
        # field_at_image = field_at_image * padded_field

        ax20 = axs[1, 1]
        ax20.set_title('field_at_image')
        im20 = ax20.imshow(np.abs(field_at_image.field), extent=field_at_image.extent)

    plt.show(block=False)