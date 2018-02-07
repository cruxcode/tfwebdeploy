from skimage import io as skio
from skimage import transform as sktr
def _read_img(filename):
    #Args:
    #   filename: full address of file
    #Returns:
    #   numpy.ndarray
    return skio.imread(filename)

def preprocess(filename, new_size):
    #Args:
    #   filename: full address of file
    #Returns:
    #   processed numpy.ndarray
    img = _read_img(filename) 
    img = sktr.resize(img, tuple(new_size))
    return img
