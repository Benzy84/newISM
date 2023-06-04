from globals import *


# pygame.init()


def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width = pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)


def setup(path):

    px = pygame.image.load(path)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px


def mainLoop(screen, px):
    topleft = bottomright = prior = None
    n=0
    while n!=1:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    n=1
        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return ( topleft + bottomright )


def get_points_to_crop(file_path):

    screen, px = setup(file_path)
    left, upper, right, lower = mainLoop(screen, px)
    pygame.display.quit()

    # ensure output rect always has positive width, height
    if right < left:
        left, right = right, left
    if lower < upper:
        lower, upper = upper, lower



    return left, upper, right, lower


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def getimage():

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    im = imread(file_path)
    if len(im.shape) == 3:
        im = rgb2gray(im)
    im = im / np.max(np.max(im))
    im[im > 0.5] = 1
    im[im <= 0.5] = 0

    """
    fig, ax = plt.subplots()
    im1 = ax.imshow(im)
    plt.show(block=False)
    """

    return im, file_path


def pad(field_in, padding_size):
    padded_field = copy.deepcopy(field_in)
    pading_vec = (padding_size, padding_size, padding_size, padding_size)
    padded_field.field = torch.nn.functional.pad(field_in.field, pading_vec).to(device)
    step = field_in.step

    start_x = torch.min(field_in.x_coordinates) - padding_size * step
    end_x = torch.max(field_in.x_coordinates) + padding_size * step

    start_y = torch.max(field_in.y_coordinates) + padding_size * step
    end_y = torch.min(field_in.y_coordinates) - padding_size * step

    padded_dim_y, padded_dim_x = torch.tensor(padded_field.field.shape).to(device)

    padded_field.x_coordinates = torch.linspace(start_x, end_x, padded_dim_x).to(device)
    padded_field.y_coordinates = torch.linspace(start_y, end_y, padded_dim_y).to(device)
    padded_field.mesh = torch.meshgrid(padded_field.y_coordinates, padded_field.x_coordinates)

    padded_field.length_x = padded_field.x_coordinates.max() - padded_field.x_coordinates.min()
    padded_field.length_y = padded_field.y_coordinates.max() - padded_field.y_coordinates.min()

    padded_field.extent = torch.tensor([start_x, end_x, end_y, start_y]).to(device)

    padded_field.padding_size = padding_size

    return padded_field


def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True


def create_field(method_number, **kwargs):
    """
    Create a field matrix based on the specified option.

    Parameters:
    - method_number: The option to use for creating the field matrix. Options are:
        1: Create a field matrix
        2: Create a field matrix with points
        3: Create a field from an image
        4: Cut a part of the picture
    - **kwargs: Additional arguments depending on the method:
        - field_dim_x, field_dim_y: The dimensions of the field matrix for the 'matrix' and 'points' options. Default is 512 for both.
        - num_x_points, num_y_points: The number of points in the x and y dimensions for the 'points' option. Default is 1 for both.
        - image_path: The path to the image file for the 'image' and 'crop' options.
        - crop_region: A tuple specifying the region to crop from the image for the 'crop' option.

    Returns:
    - field: The created field matrix.
    """

    # Get the parameters from kwargs, or use default values
    field_dim_x = kwargs.get('field_dim_x', 512)
    field_dim_y = kwargs.get('field_dim_y', 512)
    num_x_points = kwargs.get('num_x_points', 1)
    num_y_points = kwargs.get('num_y_points', 1)
    image_path = kwargs.get('image_path', None)
    crop_region = kwargs.get('crop_region', None)

    if method_number == 1:
        # Option 1: Create a field matrix
        field = torch.ones((field_dim_y, field_dim_x)).to(device)
        field[field_dim_x // 2, field_dim_y // 2] = 0

    elif method_number == 2:
        # Option 2: Create a field matrix with points
        field = torch.zeros((field_dim_y, field_dim_x)).to(device)
        num_of_x_intervals = num_x_points + 1
        num_of_pix_in_x_interval = torch.tensor((field_dim_x - num_x_points) / num_of_x_intervals).to(device)
        num_of_pix_in_x_interval = torch.ceil(num_of_pix_in_x_interval)
        num_of_y_intervals = num_y_points + 1
        num_of_pix_in_y_interval = torch.tensor((field_dim_y - num_y_points) / num_of_y_intervals).to(device)
        num_of_pix_in_y_interval = torch.ceil(num_of_pix_in_y_interval)
        column = 0
        for x_point in range(num_x_points):
            column += num_of_pix_in_x_interval
            row = 0
            for y_point in range(num_y_points):
                row += num_of_pix_in_y_interval
                field[int(row), int(column)] = 1
                row += 1
            column += 1

    elif method_number == 3:
        #Option 3: Create a field from an image
        field, _ = getimage()
        field = torch.from_numpy(field)



    elif method_number == 4:
        # Option 4: Cut a part of the picture
        image, file_path = getimage()
        left, upper, right, lower = get_points_to_crop(file_path)
        field = image[upper:lower, left:right]
        field = torch.from_numpy(field)


    else:
        raise ValueError(f"Invalid option: {method_number}")

    return field

import tkinter as tk
from tkinter import filedialog
from PIL import Image

def save_field_as_image(field):
    # Convert the field to a numpy array
    field = field.cpu().numpy()

    # Normalize the field to the range [0, 255]
    field = ((field - field.min()) * (255 / (field.max() - field.min()))).astype(np.uint8)

    # Create an image from the array
    img = Image.fromarray(field)

    # Open a file dialog for saving the image
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".png")

    # Save the image
    img.save(file_path)


def plot_field(field):
    fig, ax = plt.subplots()
    im = ax.imshow(field.field.to('cpu').numpy(), extent=field.extent.to('cpu').numpy())
    title = field.name if field.name is not None else 'Default Title'
    ax.set_title(title)
    plt.show()
