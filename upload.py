from globals import *

matplotlib.use('TkAgg')

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

with open(file_path, 'rb') as f:
    bk_restore = pickle.load(f)

del f, file_path, root,

v00_system = bk_restore['v00_system']
v01_initial_field = bk_restore['v01_initial_field']
v02_small_padded_field = bk_restore['v02_small_padded_field']
v03_filtered_small_padded_field = bk_restore['v03_filtered_small_padded_field']
v04_padded_object = bk_restore['v04_padded_object']
v05_field_before_first_lens = bk_restore['v05_field_before_first_lens']
v06_field_after_first_lens = bk_restore['v06_field_after_first_lens']
v07_field_before_second_lens = bk_restore['v07_field_before_second_lens']
v08_field_after_second_lens = bk_restore['v08_field_after_second_lens']
v09_image_wide_field_imaging = bk_restore['v09_image_wide_field_imaging']
v22_final_confocal_field = bk_restore['v22_final_confocal_field']
v23_ism_pic_by_resize1 = bk_restore['v23_ism_pic_by_resize1']
v23_ism_pic_by_resize2 = bk_restore['v23_ism_pic_by_resize2']
v24_ism_pic_by_down_sampling = bk_restore['v24_ism_pic_by_down_sampling']
v25_wide_field_by_sum = bk_restore['v25_wide_field_by_sum']

del bk_restore

# show padded field
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(v04_padded_object.field.to('cpu').numpy(), extent=v04_padded_object.extent.to('cpu').numpy())
ax1.set_title('padded object')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(v09_image_wide_field_imaging.field.to('cpu').numpy()),
                 extent=v09_image_wide_field_imaging.extent.to('cpu').numpy())
ax1.set_title('field_at_image_wide_field')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(v22_final_confocal_field.field.to('cpu').numpy()),
                 extent=v22_final_confocal_field.extent.to('cpu').numpy())
ax1.set_title('final_confocal_field')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(v23_ism_pic_by_resize1.field.to('cpu').numpy()),
                 extent=v23_ism_pic_by_resize1.extent.to('cpu').numpy())
ax1.set_title('v23_ism_pic_by_resize1')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(v23_ism_pic_by_resize2.field.to('cpu').numpy()),
                 extent=v23_ism_pic_by_resize2.extent.to('cpu').numpy())
ax1.set_title('v23_ism_pic_by_resize2')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(v24_ism_pic_by_down_sampling.field.to('cpu').numpy()),
                 extent=v24_ism_pic_by_down_sampling.extent.to('cpu').numpy())
ax1.set_title('v24_ism_pic_by_down_sampling')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(v25_wide_field_by_sum.field.to('cpu').numpy()),
                 extent=v25_wide_field_by_sum.extent.to('cpu').numpy())
ax1.set_title('v25_wide_field_by_sum')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(v24_ism_pic_by_down_sampling.field.to('cpu').numpy()),
                 extent=v24_ism_pic_by_down_sampling.extent.to('cpu').numpy())
ax1.set_title('v24_ism_pic_by_down_sampling')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

plt.show(block=False)

#############################
#####     post proc     #####
#############################
img = v09_image_wide_field_imaging.field
img = img / torch.max(torch.max(np.abs(img)))
# Compute conv2 using manual padding and normalization
padding_vec = (0, img.shape[1] - 1, 0, img.shape[0] - 1)
padded_im = torch.nn.functional.pad(img, padding_vec).to(device)
padded_im_ft = torch.fft.fftshift(torch.fft.fft2(padded_im))
conv_ft = padded_im_ft ** 2
conv = torch.fft.ifft2(torch.fft.ifftshift(conv_ft))

# Check if the results are the same
# imag = v09_image_wide_field_imaging.field.to('cpu').numpy()
# imag = imag / np.max(np.max(np.abs(imag)))
# conv1 = signal.fftconvolve(imag, imag)
# print(np.allclose(conv1, conv.to('cpu').numpy()))

# down sampling wide field imaging
rows_idx_to_cut = torch.arange(0, conv.shape[0], 2).to(device)
columns_idx_to_cut = torch.arange(0, conv.shape[1], 2).to(device)

down_sampled_conv = copy.deepcopy(v09_image_wide_field_imaging)
down_sampled_conv.field = conv[rows_idx_to_cut, :][:, columns_idx_to_cut]

# padded_conv = np.pad(conv, [(conv.shape[0] // 2), (conv.shape[1] // 2)])
padding_vec2 = (img.shape[1], img.shape[1], img.shape[0], img.shape[0])
padded_conv = torch.nn.functional.pad(conv, padding_vec2).to(device)

conv_cs = conv[conv.shape[0] // 2 + 1, :]
conv_cs = conv_cs / torch.max(abs(conv_cs))
padded_conv_cs = padded_conv[padded_conv.shape[0] // 2 + 1, :]
padded_conv_cs = padded_conv_cs / torch.max(abs(padded_conv_cs))
down_sampled_conv_cs = down_sampled_conv.field[down_sampled_conv.field.shape[0] // 2 + 1, :]
down_sampled_conv_cs = down_sampled_conv_cs / torch.max(abs(down_sampled_conv_cs))
wide_field_cs = v09_image_wide_field_imaging.field[v09_image_wide_field_imaging.field.shape[0] // 2 + 1, :]
wide_field_cs = wide_field_cs / torch.max(abs(wide_field_cs))
ism_cs = v24_ism_pic_by_down_sampling.field[v24_ism_pic_by_down_sampling.field.shape[0] // 2 + 1, :]
ism_cs = ism_cs / torch.max(abs(ism_cs))
x_coor = v09_image_wide_field_imaging.x_coordinates.to('cpu').numpy()
x_coor1 = np.linspace(np.min(x_coor), np.max(x_coor), conv_cs.shape[0])
x_coor2 = np.linspace(np.min(x_coor), np.max(x_coor), padded_conv_cs.shape[0])
x_coor3 = np.linspace(np.min(x_coor), np.max(x_coor), down_sampled_conv_cs.shape[0])
x_coor4 = np.linspace(np.min(x_coor), np.max(x_coor), wide_field_cs.shape[0])
x_coor5 = np.linspace(np.min(x_coor), np.max(x_coor), ism_cs.shape[0])
fig, ax = plt.subplots()
line1, = ax.plot(x_coor1, abs(conv_cs.to('cpu').numpy()), label="conv_cs")
line2, = ax.plot(x_coor2, abs(padded_conv_cs.to('cpu').numpy()), label="padded_conv_cs")
line3, = ax.plot(x_coor3, abs(down_sampled_conv_cs.to('cpu').numpy()), label="down_sampled_conv_cs")
line4, = ax.plot(x_coor4, abs(wide_field_cs.to('cpu').numpy()), label="wide_field_cs")
line5, = ax.plot(x_coor4, abs(ism_cs.to('cpu').numpy()), label="ism_cs")
ax.legend()

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(conv.to('cpu').numpy()), extent=v09_image_wide_field_imaging.extent.to('cpu').numpy())
ax1.set_title('conv')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(padded_conv.to('cpu').numpy()), extent=v09_image_wide_field_imaging.extent.to('cpu').numpy())
ax1.set_title('padded_conv')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(down_sampled_conv.field.to('cpu').numpy()), extent=v09_image_wide_field_imaging.extent.to('cpu').numpy())
ax1.set_title('down_sampled_conv')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs(down_sampled_conv.field), extent=v09_image_wide_field_imaging.extent.to('cpu').numpy())
ax1.set_title('down_sampled_wide_field')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(np.abs((v09_image_wide_field_imaging.field ** 2).to('cpu').numpy()),
                 extent=v09_image_wide_field_imaging.extent.to('cpu').numpy())
ax1.set_title('squared field_at_image_wide_field')
plt.show(block=False)
fig1.colorbar(im1)
del fig1, ax1, im1

x_coor = v09_image_wide_field_imaging.x_coordinates.to('cpu').numpy()
row = v09_image_wide_field_imaging.field.shape[0] // 2 + 1
wide_field_cs = v09_image_wide_field_imaging.field[row, :]
wide_field_cs = wide_field_cs / torch.max(abs(wide_field_cs))
con_focal_cs = v22_final_confocal_field.field[row, :]
con_focal_cs = con_focal_cs / torch.max(abs(con_focal_cs))
ism_cs = v24_ism_pic_by_down_sampling.field[row, :]
ism_cs = ism_cs / torch.max(abs(ism_cs))
wide_field_squered_cs = v09_image_wide_field_imaging.field ** 2
wide_field_squered_cs = wide_field_squered_cs[row, :]
wide_field_squered_cs = wide_field_squered_cs / torch.max(abs(wide_field_squered_cs))
auto_correlated_wide_field_cs = down_sampled_conv.field[row, :]
auto_correlated_wide_field_cs = auto_correlated_wide_field_cs / np.max(auto_correlated_wide_field_cs)

fig, ax = plt.subplots()
line1, = ax.plot(x_coor, abs(wide_field_cs.to('cpu').numpy()), label="wide_field_cs")
line2, = ax.plot(x_coor, abs(con_focal_cs.to('cpu').numpy()), label="con_focal_cs")
ax.legend()

fig, ax = plt.subplots()
line1, = ax.plot(x_coor, abs(wide_field_cs.to('cpu').numpy()), label="wide_field_cs")
line2, = ax.plot(x_coor, abs(ism_cs.to('cpu').numpy()), label="ism_cs")
ax.legend()

fig, ax = plt.subplots()
line1, = ax.plot(x_coor, abs(ism_cs.to('cpu').numpy()), label="ism_cs")
line2, = ax.plot(x_coor, abs(con_focal_cs.to('cpu').numpy()), label="con_focal_cs")
ax.legend()

fig, ax = plt.subplots()
line1, = ax.plot(x_coor, abs(con_focal_cs.to('cpu').numpy()), label="con_focal_cs")
line2, = ax.plot(x_coor, abs(wide_field_squered_cs.to('cpu').numpy()), label="wide_field_squered_cs")
ax.legend()

fig, ax = plt.subplots()
line1, = ax.plot(x_coor, abs(ism_cs.to('cpu').numpy()), label="ism_cs")
line2, = ax.plot(x_coor, abs(auto_correlated_wide_field_cs), label="autocorelated_wide_field_cs")
ax.legend()

fig, ax = plt.subplots()
line1, = ax.plot(x_coor, abs(wide_field_cs.to('cpu').numpy()), label="wide_field_cs")
line2, = ax.plot(x_coor, abs(con_focal_cs), label="con_focal_cs")
line3, = ax.plot(x_coor, abs(ism_cs.to('cpu').numpy()), label="ism_cs")
line4, = ax.plot(x_coor, abs(wide_field_squered_cs.to('cpu').numpy()), label="wide_field_squered_cs")
line5, = ax.plot(x_coor, abs(auto_correlated_wide_field_cs), label="autocorelated_wide_field_cs")

ax.legend()
