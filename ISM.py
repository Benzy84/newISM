import copy

from optical_elements import *
import functions_gpu
import gc
import propagators_gpu as prop
matplotlib.use('TkAgg')

###############################
###############################
### setting up field matrix ###
###############################
###############################


# Method 1: Create a field matrix with points
field_dim_x=100
field_dim_y=200
points_value=0
bg_value=1
num_x_points=2
num_y_points=3
field1 = functions_gpu.create_field(1, field_dim_x=field_dim_x, field_dim_y=field_dim_y, points_value=points_value, bg_value=bg_value, num_x_points=num_x_points, num_y_points=num_y_points)
del field_dim_x, field_dim_y, points_value, bg_value, num_x_points, num_y_points
#
# # Method 2: Create a field from an image
# field2 = functions_gpu.create_field(2)
#
# # Method 3: Cut a part of the picture
# field3 = functions_gpu.create_field(3)


field = field1; del field1

# Save teh field as png image
# functions_gpu.save_field_as_image(field)

start = time.time()

#########################################
### real space coordinates definition ###
###        all units in [meter]       ###
#########################################
# Prepare parameters
# Define the parameters
length_x = 0.2  # The real length in the x dimension
wavelength = 632.8e-9  # The wavelength of the field
z = 0  # The z position of the field

# Create the initial field
v01_initial_field = Field(field, name='01-Initial Field')
del field


# Set the attributes of the field
v01_initial_field.z = torch.tensor(z).to(device)  # Set the z position of the field
v01_initial_field.length_x = torch.tensor(length_x).to(device)  # Set the real length in the x dimension
v01_initial_field.wavelength = torch.tensor(wavelength).to(device)  # Set the wavelength of the field
v01_initial_field.step = v01_initial_field.length_x / (v01_initial_field.field.shape[1] - 1)  # Calculate and set the step size
# Calculate the real length in the y dimension
v01_initial_field.length_y = v01_initial_field.step * (v01_initial_field.field.shape[0] - 1)

# Calculate the coordinates such that 0 is in the middle
start_x = -0.5 * v01_initial_field.length_x
end_x = 0.5 * v01_initial_field.length_x
start_y = 0.5 * v01_initial_field.length_y
end_y = -0.5 * v01_initial_field.length_y

# Set the coordinates of the field
v01_initial_field.x_coordinates = torch.linspace(start_x, end_x, v01_initial_field.field.shape[1]).to(device)
v01_initial_field.y_coordinates = torch.linspace(start_y, end_y, v01_initial_field.field.shape[0]).to(device)

# Create a meshgrid of the coordinates
v01_initial_field.mesh = torch.meshgrid(v01_initial_field.y_coordinates, v01_initial_field.x_coordinates)

# Set the extent of the field
v01_initial_field.extent = torch.tensor([start_x, end_x, end_y, start_y]).to(device)

# Set the padding size of the field
v01_initial_field.padding_size = torch.tensor(0).to(device)

# Delete unnecessary variables
del length_x, wavelength, z, start_x, end_x, start_y, end_y


##################
## show field ##
##################
functions_gpu.plot_field(v01_initial_field)

####################################
### padding the field with zeros ###
####################################
# Define the size of the padding to be added to the field.
small_padding_size = 50
small_padding_size = torch.tensor(small_padding_size).to(device)

# Pad the initial field with zeros on all sides.
# This is done to prepare the field for further processing steps.
v02_small_padded_field = functions_gpu.pad(v01_initial_field, small_padding_size)
v02_small_padded_field.name = 'v02-small padded field'

# Create a copy of the padded field.
# We will apply a Gaussian filter to this copy in the next step.
v03_filtered_small_padded_field = copy.deepcopy(v02_small_padded_field)
v03_filtered_small_padded_field.name = 'v03-filtered small padded field'

# Convert the field data to a numpy array and apply a Gaussian filter.
# The Gaussian filter smooths the field, reducing high-frequency noise.
v03_field = v03_filtered_small_padded_field.field.to('cpu').numpy()
v03_field = gaussian_filter(v03_field, sigma=np.sqrt(0))

# Convert the filtered field back to a tensor and store it in the field object.
v03_filtered_small_padded_field.field = torch.from_numpy(v03_field).to(device)

# show padded field
functions_gpu.plot_field(v03_filtered_small_padded_field)
del v03_field, small_padding_size


# Define the size of the additional padding to be added to the field.
# We choose a larger padding size for this step.
padding_size = 250
padding_size = torch.tensor(padding_size).to(device)

# Pad the filtered field with additional zeros on all sides.
# This is done to prepare the field for further processing steps.
v04_padded_object = functions_gpu.pad(v03_filtered_small_padded_field, padding_size)

v04_padded_object.name = 'v04_padded_object'

# show padded field
functions_gpu.plot_field(v04_padded_object)
del padding_size


##############################
##############################
#####     simulation     #####
##############################
##############################


# Create some optical elements
focal_length=30e-2
lens_radius=5

# Define the first FreeSpace object
free_space1 = FreeSpace(length=2*focal_length, name='free_space1')

# Define the Lens object
lens = Lens(focal_length, lens_radius, name='lens')

# Define the first FreeSpace object
free_space2 = FreeSpace(length=2*focal_length, name='free_space2')

# Define the Iris object
iris = Iris(radius=0.015, name='iris')
del focal_length, lens_radius,

# Create an optical system with these elements
optical_system = OpticalSystem([free_space1, lens, free_space2])

# Propagate a field through the system
field_states_wide_field_imaging = optical_system.propagate(v04_padded_object, imaging_method='wide_field_imaging', plot_fields=True)

# Define a list of optical elements after which you want to plot the field.
elements_to_plot_after = [lens, iris]

# Get the names of all the fields that have been propagated through the optical system.
all_field_names = list(field_states_wide_field_imaging.keys())

# Iterate over the field states. The 'enumerate' function provides a counter 'i' along with the field name and field object.
for i, (current_field_name, current_field) in enumerate(field_states_wide_field_imaging.items()):
    # Check if the current field name contains the name of any of the elements in 'elements_to_plot_after'.
    # This means we're at a point in the propagation where we've just passed through one of these elements.
    # Also check if this is the last field in the list, which we want to plot regardless of which element it's after.
    if any(element.name in current_field_name for element in elements_to_plot_after):
        # If either of the above conditions is true, plot the current field.
        functions_gpu.plot_field(current_field)
del i, current_field_name, current_field, all_field_names, elements_to_plot_after
del free_space1, free_space2, lens, iris

#
#
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# v20_object_fft = torch.fft.fftshift(torch.fft.fft2(abs(v04_padded_object.field)))
# v21_regular_imaging_fft = torch.fft.fftshift(torch.fft.fft2(abs(v07_image_wide_field_imaging.field)))
#
# fig00, axs00 = plt.subplots(2, 2)
# ax = axs00[0, 0]
# ax.set_title('object fft')
# im2 = ax.imshow(np.abs(v20_object_fft.to('cpu').numpy()))
# fig00.colorbar(im2)
# del ax, im2
#
# ax = axs00[0, 1]
# ax.set_title('regular_imaging_fft')
# im2 = ax.imshow(np.abs(v21_regular_imaging_fft.to('cpu').numpy()))
# fig00.colorbar(im2)
# del ax, im2
#
# plt.show(block=False)
#
#
#
#
#
#
#
# ############################
# ###   confocal imaging   ###
# ############################
# v08_temp_field = copy.deepcopy(v04_padded_object)
# v08_temp_field.field = 0 * v08_temp_field.field
# v08_temp_field.z = v00_system.u + v00_system.v
#
# v16_final_confocal_field = copy.deepcopy(v04_padded_object)
# v16_final_confocal_field.field = 0 * v16_final_confocal_field.field * 1j
#
# points_around_for_ism = 100
# ism_dim = (1 * points_around_for_ism + 1, 1 * points_around_for_ism + 1)
#
# v17_ism_pic_by_resize1 = copy.deepcopy(v04_padded_object)
# v17_ism_pic_by_resize2 = copy.deepcopy(v04_padded_object)
# v17_ism_pic_by_resize1.field = 0 * v17_ism_pic_by_resize1.field * 1j
# v17_ism_pic_by_resize2.field = 0 * v17_ism_pic_by_resize2.field * 1j
# v18_ism_pic_by_down_sampling = copy.deepcopy(v04_padded_object)
# v18_ism_pic_by_down_sampling.field = 0 * v18_ism_pic_by_down_sampling.field * 1j
# v19_wide_field_by_sum = copy.deepcopy(v07_image_wide_field_imaging)
# v19_wide_field_by_sum.field = 0 * v19_wide_field_by_sum.field
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# Get the last field from field_states_wide_field_imaging
last_field_name = list(field_states_wide_field_imaging.keys())[-1]
field_by_wide_field_imaging = field_states_wide_field_imaging[last_field_name]

confocal_field_in = field_by_wide_field_imaging
del last_field_name, field_by_wide_field_imaging

# row = column = 0
counter = 0
points_around_for_ism = 100
for row in tqdm(np.arange(v02_small_padded_field.field.shape[0])):
    for column in (np.arange(v02_small_padded_field.field.shape[1])):
        counter = counter + 1
        padding_size_x = confocal_field_in.padding_size[0]
        padding_size_y = confocal_field_in.padding_size[1]
        [x, y] = [column + padding_size_x, row + padding_size_y]
        confocal_field_in.field = 0 * confocal_field_in.field
        confocal_field_in.field[y, x] = 1

        field_states_reversed_confocal_imaging = optical_system.propagate(confocal_field_in, reverse=True)
        last_field_name = list(field_states_reversed_confocal_imaging.keys())[-1]
        last_field_reversed_confocal_imaging = field_states_reversed_confocal_imaging[last_field_name]
        field_states_confocal_imaging = optical_system.propagate(last_field_reversed_confocal_imaging,
                                                                 imaging_method='confocal_imaging')
        if counter==1:
            last_field_name = list(field_states_confocal_imaging.keys())[-1]
            final_confocal_field = copy.deepcopy(field_states_confocal_imaging[last_field_name])
            final_confocal_field.field = 0*final_confocal_field.field
            final_confocal_field.name = 'final_confocal_field'
            ism_pic_by_down_sampling = copy.deepcopy(final_confocal_field)
            ism_pic_by_down_sampling.name = 'ism_pic_by_down_sampling'
            ism_pic_by_Tal = copy.deepcopy(final_confocal_field)
            ism_pic_by_Tal.name = 'ism_pic_by_Tal'
            original_dimensions = ism_pic_by_Tal.field.shape
            padding_size = (original_dimensions[0]//2, original_dimensions[1]//2)
            padding_size = torch.tensor(padding_size).to(device)
            padded_field = functions_gpu.pad(ism_pic_by_Tal, padding_size)
            ism_pic_by_Tal.field = padded_field.field
            wide_field_by_sum = copy.deepcopy(final_confocal_field)
            wide_field_by_sum.name = 'wide_field_by_sum'
            del padded_field

        field_at_image_plane = field_states_confocal_imaging[last_field_name]
        final_confocal_field.field[y, x] = field_at_image_plane.field[y, x]

        #########################
        #######    ISM    #######
        #########################

        # by cropping area
        area_for_ism = field_at_image_plane.field[y - points_around_for_ism:y + 1 + points_around_for_ism, x - points_around_for_ism:x + 1 + points_around_for_ism]

        ##################
        ## resize image ##
        ##################
        size = area_for_ism.shape
        rows_idx_to_cut = torch.arange(0, size[0], 2).to(device)
        columns_idx_to_cut = torch.arange(0, size[1], 2).to(device)
        down_sampled_area = area_for_ism[rows_idx_to_cut, :][:, columns_idx_to_cut]

        start_row = int(y - 0.5 * points_around_for_ism)
        end_row = int(y + 1 + 0.5 * points_around_for_ism)
        start_col = int(x - 0.5 * points_around_for_ism)
        end_col = int(x + 1 + 0.5 * points_around_for_ism)
        ism_pic_by_down_sampling.field[start_row:end_row, start_col:end_col] += down_sampled_area
        start_row = int(y)
        end_row = start_row + field_at_image_plane.field.shape[0]
        start_col = int(x)
        end_col = start_col + field_at_image_plane.field.shape[1]
        ism_pic_by_Tal.field[start_row:end_row, start_col:end_col] += field_at_image_plane.field
        wide_field_by_sum.field += field_at_image_plane.field

        # ism_pic_by_Tal[]

        # del start_row, end_row, start_col, end_col, down_sampled_area, resized_area1, resized_area2, v15_field_at_image, a, x, y
# del counter, column, row, points_around_for_ism, ism_dim

end = time.time()
time_el = (end - start) / 60
print(time_el)

v22_confocal_fft = torch.fft.fftshift(torch.fft.fft2(abs(v16_final_confocal_field.field)))
v23_ism_fft1 = torch.fft.fftshift(torch.fft.fft2(abs(v17_ism_pic_by_resize1.field)))
v23_ism_fft2 = torch.fft.fftshift(torch.fft.fft2(abs(v17_ism_pic_by_resize2.field)))

ax = axs00[1, 0]
ax.set_title('confocal fft')
im2 = ax.imshow(np.abs(v22_confocal_fft.to('cpu').numpy()))
fig00.colorbar(im2)
del ax, im2

ax = axs00[1, 1]
ax.set_title('v29_ism_fft2')
im2 = ax.imshow(np.abs(v23_ism_fft2.to('cpu').numpy()))
fig00.colorbar(im2)
del ax, im2
plt.show(block=False)
# # field_z = padded_field_z[padded_size: dim_x - padded_size, padded_size:dim_x - padded_size]
# fig, ax = plt.subplots()
# ax.set_title('field' )
# im1 = ax.imshow(circ_iris, extent=padded_extent)
# plt.show(block=False)

del fig00, axs00, start, end, time_el, device

bk = {}
bk.update({'v00_system': v00_system})
bk.update({'v01_initial_field': v01_initial_field})
bk.update({'v02_small_padded_field': v02_small_padded_field})
bk.update({'v03_filtered_small_padded_field': v03_filtered_small_padded_field})
bk.update({'v04_padded_object': v04_padded_object})
bk.update({'v05_field_before_lens': v05_field_before_lens})
bk.update({'v06_field_after_lens': v06_field_after_lens})
bk.update({'v07_image_wide_field_imaging': v07_image_wide_field_imaging})
bk.update({'v16_final_confocal_field': v16_final_confocal_field})
bk.update({'v17_ism_pic_by_resize1': v17_ism_pic_by_resize1})
bk.update({'v17_ism_pic_by_resize2': v17_ism_pic_by_resize2})
bk.update({'v18_ism_pic_by_down_sampling': v18_ism_pic_by_down_sampling})
bk.update({'v19_wide_field_by_sum': v19_wide_field_by_sum})

# save session
with open('./ism example single lens r 55 cm.pkl', 'wb') as f:
    # Move tensors to CPU device before saving
    bk_cpu = {}
    for name, creature in bk.items():
        members = [attr for attr in dir(creature) if
                   not callable(getattr(creature, attr)) and not attr.startswith("__")]
        for member in members:
            attr_value = getattr(creature, member)

            if isinstance(attr_value, torch.Tensor):
                setattr(creature, member, attr_value.to(torch.device('cpu')))

            if member == 'mesh':
                for i in range(len(attr_value)):
                    if isinstance(attr_value[i], torch.Tensor):
                        setattr(creature, member, attr_value[i].to(torch.device('cpu')))

        bk_cpu[name] = creature
    pickle.dump(bk_cpu, f)

plt.close('all')