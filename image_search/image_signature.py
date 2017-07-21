from skimage.io import imread, imsave
from skimage.color import gray2rgb
from scipy.misc import imresize
from PIL import Image
try:
    from cairosvg import svg2png
except ImportError:
    pass
from io import BytesIO
import numpy as np
import xml.etree
from network import predict_model, load_deep_model
from get_label import read_labels
from imread import file_or_url_context


class CorruptImageError(RuntimeError):
    pass


class ImageSignature(object):
    """Image signature generator.

    """

    def __init__(self, network='vgg_16', weight_path='../model/vgg16_weights.h5'):
        """Initialize the signature generator.

        The default parameters based on deep model.

        Args:
            network (Optional): currently only support 'vgg_16'
            weight_path (Optional): model path

        """

        self.network = network
        self.weight_path = weight_path
        self.model = load_deep_model(weight_path)
        self.labels_id, self.labels_name = read_labels()

    def generate_signature(self, path_or_image, bytestream=False):
        """Generates an image signature.

        Args:
            path_or_image (string or numpy.ndarray): image path, or image array
            thumbnail_path(Optional): thumbnail save path
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image?
                (default False)

        Returns:
            The image signature: A rank 1 numpy array of length n x n x 8
                (or n x n x 4 if diagonal_neighbors == False)

        Examples:
            >>> from image_match.goldberg import ImageSignature
            >>> gis = ImageSignature()
            >>> gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
        """
        # Step 1:    Load image as array of grey-levels
        im_array = self.preprocess_image(path_or_image, bytestream=bytestream)

        # Step 2:    Extract image feature
        try:
            out, labels = predict_model(self.model, im_array, self.labels_id, self.labels_name)
        except IOError:
            raise TypeError('Cannot predict image successfully.')
        return out, labels

    @staticmethod
    def preprocess_image(image_or_path, bytestream=False):
        if bytestream:
            try:
                img = Image.open(BytesIO(image_or_path))
            except IOError:
                # could be an svg, attempt to convert
                try:
                    img = Image.open(BytesIO(svg2png(image_or_path)))
                except (NameError, xml.etree.ElementTree.ParseError):
                    raise CorruptImageError()
        elif type(image_or_path) is str:
            try:
                # img = imread(image_or_path)
                with file_or_url_context(image_or_path) as img_name:
                    img = imread(img_name)
            except IOError:
                raise TypeError('Cannot read image successfully.')
            if len(img.shape) == 2:
                img = gray2rgb(img)
            elif img.shape[-1] == 4:
                img = img[:,:,:3]
            img_size = (224, 224, 3)
            img = imresize(img, img_size)

            img = img.astype('float32')
            # We normalize the colors (in RGB space) with the empirical means on the training set
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            img = img.transpose((2, 0, 1))
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        elif type(image_or_path) is bytes:
            try:
                img = Image.open(image_or_path)
                img = np.array(img.convert('RGB'))
            except IOError:
                # try again due to PIL weirdness
                img = imread(image_or_path)
        elif type(image_or_path) is np.ndarray:
            img = image_or_path
        else:
            raise TypeError('Path or image required.')

        return img

    @staticmethod
    def normalized_distance(_a, _b):
        """Compute normalized distance between two points.

        Computes 1 - a * b / ( ||a|| * ||b||)

        Args:
            _a (numpy.ndarray): array of size m
            _b (numpy.ndarray): array of size m

        Returns:
            normalized distance between signatures (float)

        Examples:
            >>> a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> gis.normalized_distance(a, b)
            0.0332806110382

        """

        return (1.0 - np.dot(_a, _b) / (np.linalg.norm(_a) * np.linalg.norm(_b)))
