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

