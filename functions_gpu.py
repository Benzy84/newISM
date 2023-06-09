import torch

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
    """
    This function pads a given field with zeros.
    Parameters:
    - field_in: The input field to be padded.
    - padding_size: An integer or a tuple/list of size 2 representing the padding size for each dimension.
    Returns:
    - padded_field: The padded field.
    """
    # Create a copy of the input field
    padded_field = copy.deepcopy(field_in)

    # Check if padding_size is a single tensor
    if isinstance(padding_size, torch.Tensor) and torch.numel(padding_size) == 1:
        # If so, use the same padding size for both dimensions
        padding_size = torch.tensor([padding_size.item(), padding_size.item()])
    elif isinstance(padding_size, (tuple, list)):
        # If padding_size is a tuple or list, convert it to a tensor
        padding_size = torch.tensor(padding_size)

    # Define the padding vector
    padding_vec = (padding_size[1], padding_size[1], padding_size[0], padding_size[0])

    # Pad the field with zeros
    padded_field.field = torch.nn.functional.pad(field_in.field, padding_vec).to(device)

    # Calculate the step size
    step = field_in.step

    # Calculate the start and end coordinates in x and y dimensions
    start_x = torch.min(field_in.x_coordinates) - padding_size[0] * step
    end_x = torch.max(field_in.x_coordinates) + padding_size[0] * step
    start_y = torch.max(field_in.y_coordinates) + padding_size[1] * step
    end_y = torch.min(field_in.y_coordinates) - padding_size[1] * step

    # Calculate the dimensions of the padded field
    padded_dim_y, padded_dim_x = torch.tensor(padded_field.field.shape).to(device)

    # Set the x and y coordinates of the padded field
    padded_field.x_coordinates = torch.linspace(start_x, end_x, padded_dim_x).to(device)
    padded_field.y_coordinates = torch.linspace(start_y, end_y, padded_dim_y).to(device)

    # Create a meshgrid of the coordinates
    padded_field.mesh = torch.meshgrid(padded_field.y_coordinates, padded_field.x_coordinates)

    # Calculate the length in x and y dimensions
    padded_field.length_x = padded_field.x_coordinates.max() - padded_field.x_coordinates.min()
    padded_field.length_y = padded_field.y_coordinates.max() - padded_field.y_coordinates.min()

    # Set the extent of the field
    padded_field.extent = torch.tensor([start_x, end_x, end_y, start_y]).to(device)

    # Set the padding size of the field
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
    This function creates a field matrix based on the method number provided.

    Parameters:
    method_number (int): The method number to use for creating the field matrix.
    **kwargs: Additional parameters required for each method.

    Returns:
    field (torch.Tensor): The created field matrix.
    """
    # Method 1: Create a field matrix with points
    if method_number == 1:
        # Get the dimensions of the field from the kwargs, defaulting to 512 if not provided
        field_dim_x = kwargs.get('field_dim_x', 512)
        field_dim_y = kwargs.get('field_dim_y', 512)

        # Get the value for the points and the background from the kwargs, defaulting to 1 and 0 respectively if not provided
        points_value = kwargs.get('points_value', 1)
        bg_value = kwargs.get('bg_value', 0)

        # Get the number of points in x and y directions from the kwargs, defaulting to 1 if not provided
        num_x_points = kwargs.get('num_x_points', 1)
        num_y_points = kwargs.get('num_y_points', 1)

        # Create a field filled with the background value
        field = torch.full((field_dim_y, field_dim_x), bg_value, device=device)

        # Calculate the number of pixels in x and y intervals

        num_of_pix_in_x_interval = torch.tensor(field_dim_x / (num_x_points + 1))
        num_of_pix_in_x_interval = torch.ceil(num_of_pix_in_x_interval)
        num_of_pix_in_y_interval = torch.tensor(field_dim_y / (num_y_points + 1))
        num_of_pix_in_y_interval = torch.ceil(num_of_pix_in_y_interval)

        # Fill the field with points
        for x_point in range(num_x_points):
            for y_point in range(num_y_points):
                field[int((y_point + 1) * num_of_pix_in_y_interval), int((x_point + 1) * num_of_pix_in_x_interval)] = points_value

    # Method 2: Create a field from an image
    elif method_number == 2:
        field, _ = getimage()
        field = torch.from_numpy(field)

    # Method 3: Cut a part of the picture
    elif method_number == 3:
        # Option 3: Cut a part of the picture
        image, file_path = getimage()
        left, upper, right, lower = get_points_to_crop(file_path)
        field = image[upper:lower, left:right]
        field = torch.from_numpy(field)

    else:
       raise ValueError("Invalid method number. Please choose between 1, 2, and 3.")

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


def plot_field(field, title=None):
    """
    This function plots a given field.

    Parameters:
    - field: The field to be plotted.
    - title (optional): The title of the plot. If not provided, the name of the field will be used.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(field.field.to('cpu').numpy()), extent=field.extent.to('cpu').numpy())
    # If a title is provided, use it. Otherwise, use the name of the field.
    title = title if title is not None else field.name
    ax.set_title(title)
    plt.show(block=False)
