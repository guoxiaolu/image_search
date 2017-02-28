from image_search.image_signature import ImageSignature
from operator import itemgetter
import numpy as np


class SignatureDatabaseBase(object):
    """Base class for storing and searching image signatures in a database

    Note:
        You must implement the methods search_single_record and insert_single_record
        in a derived class

    """

    def search_single_record(self, rec):
        """Search for a matching image record.

        Must be implemented by derived class.

        Args:
            rec (dict): an image record. Will be in the format returned by
                make_record

                For example, rec could have the form:

                {'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
                 'signature': [0.123456, 0.234567, ... ]
                 'metadata': {...},
                 }

        Returns:
            a formatted list of dicts representing matches.

            For example, if three matches are found:

            [
             {'dist': 0.069116439263706961,
              'id': u'AVM37oZq0osmmAxpPvx7',
              'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg'},
             {'dist': 0.22484320805049718,
              'id': u'AVM37nMg0osmmAxpPvx6',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg'},
             {'dist': 0.42529792112113302,
              'id': u'AVM37p530osmmAxpPvx9',
              'metadata': {...},
              'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg'}
            ]

            You can return any fields you like, but must include at least dist and id. Duplicate entries are ok,
            and they do not need to be sorted

        """
        raise NotImplementedError

    def insert_single_record(self, rec):
        """Insert an image record.

        Must be implemented by derived class.

        Args:
            rec (dict): an image record. Will be in the format returned by
                make_record

                For example, rec could have the form:

                {'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
                 'signature': [0.123456, 0.234567, ... ]
                 'metadata': {...}
                 }

                 The number of simple words corresponds to the attribute N

        """
        raise NotImplementedError

    def __init__(self, distance_cutoff=0.45,
                 *signature_args, **signature_kwargs):
        """Set up storage scheme for images

        Args:
            distance_cutoff (Optional [float]): maximum image signature distance to
                be considered a match (default 0.45)
            *signature_args: Variable length argument list to pass to ImageSignature
            **signature_kwargs: Arbitrary keyword arguments to pass to ImageSignature

        """

        # Check float input
        if type(distance_cutoff) is not float:
            raise TypeError('distance_cutoff should be a float')
        if distance_cutoff < 0.:
            raise ValueError('distance_cutoff should be > 0 (got %r)' % distance_cutoff)

        self.distance_cutoff = distance_cutoff


        self.gis = ImageSignature(*signature_args, **signature_kwargs)

    def add_image(self, path, img=None, bytestream=False, metadata=None, refresh_after=False):
        """Add a single image to the database

        Args:
            path (string): path or identifier for image. If img=None, then path is assumed to be
                a URL or filesystem path
            img (Optional[string]): usually raw image data. In this case, path will still be stored, but
                a signature will be generated from data in img. If bytestream is False, but img is
                not None, then img is assumed to be the URL or filesystem path. Thus, you can store
                image records with a different 'path' than the actual image location (default None)
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image? If img is None but, this
                argument will be ignored.  If img is not None, and bytestream is False, then the behavior
                is as described in the explanation for the img argument
                (default False)
            metadata (Optional): any other information you want to include, can be nested (default None)

        """
        rec = make_record(path, self.gis, img=img, bytestream=bytestream, metadata=metadata)
        self.insert_single_record(rec, refresh_after=refresh_after)

    def search_image(self, path, bytestream=False):
        """Search for matches

        Args:
            path (string): path or image data. If bytestream=False, then path is assumed to be
                a URL or filesystem path. Otherwise, it's assumed to be raw image data
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image?
                (default False)

        Returns:
            a formatted list of dicts representing unique matches, sorted by dist

            For example, if three matches are found:

            [
             {'dist': 0.069116439263706961,
              'id': u'AVM37oZq0osmmAxpPvx7',
              'path': u'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg'},
             {'dist': 0.0148712559918,
              'id': u'AVM37nMg0osmmAxpPvx6',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg'},
             {'dist': 0.0307221687987,
              'id': u'AVM37p530osmmAxpPvx9',
              'path': u'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg'}
            ]

        """
        img = self.gis.preprocess_image(path, bytestream)

        # generate the signature
        record = make_record(img, self.gis)

        result = self.search_single_record(record)

        ids = set()
        unique = []

        for item in result:
            if item['id'] not in ids:
                unique.append(item)
                ids.add(item['id'])

        r = sorted(unique, key=itemgetter('dist'))
        return r


def make_record(path, gis, img=None, bytestream=False, metadata=None):
    """Makes a record suitable for database insertion.

    Note:
        This non-class version of make_record is provided for
        CPU pooling. Functions passed to worker processes must
        be picklable.

    Args:
        path (string): path or image data. If bytestream=False, then path is assumed to be
            a URL or filesystem path. Otherwise, it's assumed to be raw image data
        gis (ImageSignature): an instance of ImageSignature for generating the
            signature
        img (Optional[string]): usually raw image data. In this case, path will still be stored, but
            a signature will be generated from data in img. If bytestream is False, but img is
            not None, then img is assumed to be the URL or filesystem path. Thus, you can store
            image records with a different 'path' than the actual image location (default None)
        bytestream (Optional[boolean]): will the image be passed as raw bytes?
            That is, is the 'path_or_image' argument an in-memory image? If img is None but, this
            argument will be ignored.  If img is not None, and bytestream is False, then the behavior
            is as described in the explanation for the img argument
            (default False)
        metadata (Optional): any other information you want to include, can be nested (default None)

    Returns:
        An image record.

        For example:

        {'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
         'signature': [0.123456, 0.234567, ... ]
         'metadata': {...}
         }

    """
    record = dict()
    record['path'] = path
    if img is not None:
        signature = gis.generate_signature(img, bytestream=bytestream)
    else:
        signature = gis.generate_signature(path)

    record['signature'] = signature.tolist()

    if metadata:
        record['metadata'] = metadata

    return record

def normalized_distance(_target_array, _vec):
    """Compute normalized distance to many points.

    Computes 1 - a * b / ( ||a|| * ||b||) for every a in target_array

    Args:
        _target_array (numpy.ndarray): N x m array
        _vec (numpy.ndarray): array of size m
    Returns:
        the normalized distance (float)
    """

    topvec = np.dot(_target_array, _vec.reshape(_vec.size, 1))
    norm_a = np.linalg.norm(_target_array, axis=1)
    norm_a = norm_a.reshape(norm_a.size,1)
    norm_b = np.linalg.norm(_vec)
    finvec = 1.0 - topvec / (norm_a * norm_b)

    return finvec
