from .signature_database_base import SignatureDatabaseBase
from .signature_database_base import normalized_distance
import numpy as np
import os


class SignatureES(SignatureDatabaseBase):
    """Elasticsearch driver for image-match

    """

    def __init__(self, es, index='images', doc_type='image', timeout='10s', size=100,
                 *args, **kwargs):
        """Extra setup for Elasticsearch

        Args:
            es (elasticsearch): an instance of the elasticsearch python driver
            index (Optional[string]): a name for the Elasticsearch index (default 'images')
            doc_type (Optional[string]): a name for the document time (default 'image')
            timeout (Optional[int]): how long to wait on an Elasticsearch query, in seconds (default 10)
            size (Optional[int]): maximum number of Elasticsearch results (default 100)
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        Examples:
            >>> from elasticsearch import Elasticsearch
            >>> from image_match.elasticsearch_driver import SignatureES
            >>> es = Elasticsearch()
            >>> ses = SignatureES(es)
            >>> ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            [
             {'dist': 0.0,
              'id': u'AVM37nMg0osmmAxpPvx6',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
              'score': 0.28797293}
            ]

        """
        self.es = es
        self.index = index
        self.doc_type = doc_type
        self.timeout = timeout
        self.size = size

        super(SignatureES, self).__init__(*args, **kwargs)

    def search_single_record(self, rec):
        path = rec.pop('path')
        signature = rec.pop('signature')
        thumbnail_path = rec.pop('thumbnail')
        if 'metadata' in rec:
            rec.pop('metadata')

        res = self.es.search(index=self.index,
                              doc_type=self.doc_type,
                              size=self.size,
                              timeout=self.timeout)['hits']['hits']

        sigs = np.array([x['_source']['signature'] for x in res])

        if sigs.size == 0:
            return []

        dists = normalized_distance(sigs, np.array(signature))

        formatted_res = [{'id': x['_id'],
                          # 'score': x['_score'],
                          'msg_id': x['_source'].get('msg_id'),
                          'pic_id': x['_source'].get('pic_id'),
                          'thumbnail': x['_source'].get('thumbnail'),
                          'path': x['_source'].get('url', x['_source'].get('path'))}
                         for x in res]

        for i, row in enumerate(formatted_res):
            row['dist'] = dists[i]
        formatted_res = filter(lambda y: y['dist'] < self.distance_cutoff, formatted_res)

        return formatted_res

    def insert_single_record(self, rec, refresh_after=False):
        self.es.index(index=self.index, doc_type=self.doc_type, body=rec, refresh=refresh_after)

    def delete_duplicates(self, path):
        """Delete all but one entries in elasticsearch whose `path` value is equivalent to that of path.
        Args:
            path (string): path value to compare to those in the elastic search
        """
        result = self.es.search(body={'query':
                                 {'match':
                                      {'path': path}
                                  }
                             },
                       index=self.index)['hits']['hits']

        matching_paths = []
        matching_thumbnail = []
        for item in result:
            if item['_source']['path'] == path:
                matching_paths.append(item['_id'])
                matching_thumbnail.append(item['_source']['thumbnail'])

        if len(matching_paths) > 0:
            for i, id_tag in enumerate(matching_paths[1:]):
                self.es.delete(index=self.index, doc_type=self.doc_type, id=id_tag)
                if os.path.isfile(matching_thumbnail[i]):
                    os.remove(matching_thumbnail[i])
