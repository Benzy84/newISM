from globals import *


def distance_z(field_in, z_out, wave_length, plot=0):
    """

    :param field_in:
    :param coordinates:
    :param z_in:
    :param z_out:
    :param wave_length:
    :param plot:
    :return:
    """
    """
    Field propagation from z_in to z_out

    Parameters
    ----------
    field_in : torch 2d matrix
        field at z_in
    coordinates : torch 2Xn array
        real space coordinates of the field matrix
    extent : list
        The extent of
    z_in : str
        The file location of the spreadsheet
    z_out : str
        The file location of the spreadsheet
    wave_length : str
        The file location of the spreadsheet
    plot : int (1 or 0)
        tels the function to plot or not figures

    Returns
    -------
    field_z : str
        a list of strings used that are the header columns
    """

    k0 = 2 * torch.pi / wave_length
    z = abs(z_out - field_in.z)

    """
    fig, ax = plt.subplots()
    ax.set_title('field in')
    im1 = ax.imshow(torch.abs(field_in.field))
    plt.show(block=False)
    """

    dim_y, dim_x = torch.tensor(field_in.field.shape).to(device)
    step = field_in.step

    fx = torch.fft.fftshift(torch.fft.fftfreq(dim_x, step)).to(device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(dim_y, step)).to(device)

    start_fx = torch.min(fx)
    end_fx = torch.max(fx)
    start_fy = torch.min(fy)
    end_fy = torch.max(fy)

    angular_0 = torch.fft.fftshift(torch.fft.fft2(field_in.field)).to(device)

    """
    fig, ax = plt.subplots()
    ax.set_title('angular at')
    im1 = ax.imshow(torch.abs(angular_0), extent=[start_fx, end_fx, start_fy, end_fy], origin='lower')
    plt.show(block=False)
    """

    fxfx, fyfy = torch.meshgrid(fy, fx)
    transfer_function = torch.exp(k0 * 1j * z * torch.sqrt(1 - (wave_length * fxfx) ** 2 - (wave_length * fyfy) ** 2))
    # transfer_function = transfer_function1 * transfer_function1
    transfer_function[(fxfx ** 2 + fyfy ** 2) > 1 / (wave_length ** 2)] = 0

    """
    fig, ax = plt.subplots()
    ax.set_title('transfer_function')
    im1 = ax.imshow(torch.abs(transfer_function), extent=[start_fx, end_fx, start_fy, end_fy], origin='lower')
    plt.show(block=False)
    """

    angular_z = angular_0 * transfer_function
    """
    fig, ax = plt.subplots()
    ax.set_title('angular at z')
    im1 = ax.imshow(torch.abs(angular_z) - torch.abs(angular_0), extent=[start_fx, end_fx, start_fy, end_fy], origin='lower')
    plt.show(block=False)
    """

    field_z = copy.deepcopy(field_in)
    field_in.field.device
    field_z.z = z_out
    field_z.field = 0 * field_z.field
    field_z.field.device

    temp_field = torch.fft.ifft2(torch.fft.ifftshift(angular_z))
    # temp_field = torch.real_if_close(temp_field)
    # temp_field = torch.round(temp_field, 12)
    # temp_field[temp_field == 0] = 0

    # start_row = field_z.padding_size
    # end_row = dim_y - field_z.padding_size
    # start_col = field_z.padding_size
    # end_col = dim_x - field_z.padding_size
    # field_z.field[start_row:end_row, start_col:end_col] = temp_field[start_row:end_row, start_col:end_col]
    field_z.field = temp_field

    """
    fig, ax = plt.subplots()
    ax.set_title('padded field at z = %.2f' % z)
    im1 = ax.imshow(torch.abs(temp_field))
    plt.show(block=False)
    """

    if plot:
        min1 = torch.min(torch.abs(field_in.field))
        min2 = torch.min(torch.abs(field_z.field))
        max1 = torch.max(torch.abs(field_in.field))
        max2 = torch.max(torch.abs(field_z.field))

        minmin = torch.min(min1, min2)
        maxmax = torch.max(max1, max2)

        fig, axs = plt.subplots(1, 2)

        ax = axs[0]
        ax.set_title('field at z = %.2f' % field_in.z)
        im1 = ax.imshow(np.abs(field_in.field.to('cpu').numpy()), extent=field_in.extent.to('cpu').numpy())
        ax = axs[1]
        ax.set_title('field at z = %.2f' % z_out)
        im2 = ax.imshow(np.abs(field_z.field.to('cpu').numpy()), extent=field_z.extent.to('cpu').numpy())

        """
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        """

        plt.show(block=False)

        """
        fig, ax = plt.subplots()
        ax.set_title('angular at z')
        im1 = ax.imshow(torch.abs(field_in) - torch.abs(field_z), extent=[start_fx, end_fx, start_fy, end_fy], origin='lower')
        plt.colorbar(im1)
          plt.show(block=False)
        """

    return field_z


def iris(field_in, padded_coordinates, padding_size, radius, iris_center_pos, plot=0):
    """


    :param field_in:
    :param padded_coordinates:
    :param padding_size:
    :param radius:
    :param iris_center_pos:
    :param plot:
    :return:
    """
    # This function gets only a padded field, and his coordinates, creates circular aperture
    # and returns the field after the aperture.
    padded_x_coordinates = padded_coordinates[0]
    padded_y_coordinates = padded_coordinates[1]
    start_x = torch.min(padded_x_coordinates)
    end_x = torch.max(padded_x_coordinates)
    start_y = torch.min(padded_y_coordinates)
    end_y = torch.max(padded_y_coordinates)
    padded_extent = [start_x, end_x, start_y, end_y]

    step_x = torch.diff(padded_x_coordinates)[0]
    start_x = torch.min(padded_x_coordinates) + padding_size * step_x
    end_x = torch.max(padded_x_coordinates) - padding_size * step_x
    step_y = torch.diff(padded_y_coordinates)[0]
    start_y = torch.max(padded_y_coordinates) + padding_size * step_y
    end_y = torch.min(padded_y_coordinates) - padding_size * step_y

    dim_x = len(padded_x_coordinates) - 2 * padding_size
    dim_y = len(padded_y_coordinates) - 2 * padding_size

    x_coordinates = torch.linspace(start_x, end_x, dim_x)
    y_coordinates = torch.linspace(start_y, end_y, dim_y)

    xx, yy = torch.meshgrid(x_coordinates, y_coordinates)

    circ_iris = xx * 0

    circ_iris[((xx - iris_center_pos[0]) ** 2 + (yy - iris_center_pos[1]) ** 2) < radius ** 2] = 1

    padded_circ = torch.pad(circ_iris, padding_size)

    field_out = field_in * padded_circ

    if plot:
        min1 = torch.min(torch.abs(circ_iris))
        min2 = torch.min(torch.abs(field_out))
        max1 = torch.max(torch.abs(circ_iris))
        max2 = torch.max(torch.abs(field_out))

        minmin = torch.min([min1, min2])
        maxmax = torch.max([max1, max2])

        fig, axs = plt.subplots(1, 3)

        ax = axs[0]
        ax.set_title('field before iris')
        im1 = ax.imshow(np.abs(field_in), extent=padded_extent)

        ax = axs[1]
        ax.set_title('circular aperture')
        im1 = ax.imshow(padded_circ, extent=padded_extent)

        ax = axs[2]
        ax.set_title('field after iris')
        im2 = ax.imshow(np.abs(field_out), extent=padded_extent)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        plt.show(block=False)

    return field_out


def thin_lens(field_in, wave_length, lens_radius, lens_center_pos, focal_length, plot=0):
    """

    :param field_in:
    :param coordinates:
    :param wave_length:
    :param lens_radius:
    :param lens_center_pos:
    :param focal_length:
    :param plot:
    :return:
    """

    extent = field_in.extent
    x_coordinates = field_in.x_coordinates
    y_coordinates = field_in.y_coordinates
    xx, yy = torch.meshgrid(y_coordinates, x_coordinates)

    k0 = 2 * torch.pi / wave_length

    lens_shape = torch.zeros(len(y_coordinates), len(x_coordinates), dtype=torch.int).to(device)
    lens_shape[((xx - lens_center_pos[0]) ** 2 + (yy - lens_center_pos[1]) ** 2) < lens_radius ** 2] = 1

    lens_phase = torch.exp(-1j * k0 / (2 * focal_length) * ((xx - lens_center_pos[0]) ** 2 + (yy - lens_center_pos[1]) ** 2))
    lens_phase[((xx - lens_center_pos[0]) ** 2 + (yy - lens_center_pos[1]) ** 2) > lens_radius ** 2] = 0

    """
    tf = torch.fft.fftshift(torch.fft.fft2(torch.abs(lens_phase)))
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(torch.abs(tf))
    ax1.set_title('tf')
    del fig1, ax1, im1
    """

    field_before_lens = copy.deepcopy(field_in)
    field_before_lens.field = []
    field_after_lens = copy.deepcopy(field_in)
    field_after_lens.field = []

    field_before_lens.field = field_in.field * lens_shape
    field_after_lens.field = field_before_lens.field * lens_phase


    if plot:
        min1 = torch.min(torch.abs(lens_shape)).double()
        min2 = torch.min(torch.angle(lens_phase)).double()
        min3 = torch.min(torch.abs(field_before_lens.field)).double()
        min4 = torch.min(torch.abs(field_after_lens.field)).double()
        max1 = torch.max(torch.abs(lens_shape)).double()
        max2 = torch.max(torch.angle(lens_phase)).double()
        max3 = torch.max(torch.abs(field_before_lens.field)).double()
        max4 = torch.max(torch.abs(field_after_lens.field)).double()

        minmin = torch.min(torch.tensor([min1, min2, min3, min4]))
        maxmax = torch.max(torch.tensor([max1, max2, max3, max4]))

        fig, axs = plt.subplots(2, 2)

        ax = axs[0, 0]
        ax.set_title('lens shape')
        im1 = ax.imshow(np.abs(lens_shape.to('cpu').numpy()), vmin=minmin, vmax=maxmax, extent=extent.to('cpu').numpy())

        ax = axs[0, 1]
        ax.set_title('lens_phase')
        im2 = ax.imshow(np.angle(lens_phase.to('cpu').numpy()), vmin=minmin, vmax=maxmax, extent=extent.to('cpu').numpy())

        ax = axs[1, 0]
        ax.set_title('field before lens')
        im3 = ax.imshow(np.abs(field_before_lens.field.to('cpu').numpy()), extent=extent.to('cpu').numpy())

        ax = axs[1, 1]
        ax.set_title('field after lens')
        im4 = ax.imshow(np.abs(field_after_lens.field.to('cpu').numpy()), extent=extent.to('cpu').numpy())

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im2, cax=cbar_ax)

        plt.show(block=False)

    return field_after_lens


def gauss(w_0, wave_length, z):
    E_after = [w_0 + wave_length + z]
    return E_after
