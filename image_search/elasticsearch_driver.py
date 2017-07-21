from .signature_database_base import SignatureDatabaseBase
import os
from get_label import LABEL_OTHER
from datetime import datetime, timedelta

class SignatureES(SignatureDatabaseBase):
    """Elasticsearch driver for image-match

    """

    def __init__(self, es, index='images', doc_type='image', timeout='10s', size=20,
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

    def search_single_record(self, rec, term_after):
        signature = rec.pop('signature')
        top_1 = rec.pop('top_1')
        # top_2 = rec.pop('top_2')
        # top_3 = rec.pop('top_3')

        if 'metadata' in rec:
            rec.pop('metadata')
        # a common query DSL
        body={"terminate_after":term_after,
        "query": {
        "function_score": {
             "query" : {
                "bool" : {
                  "must" : []
               }
           },
            "script_score": {
                "script": {
                    "inline": "payload_vector_score",
                    "lang": "native",
                    "params": {
                        "field": "signature",
                        "vector": signature,
                        "cosine": True
                    }
                }
            },
            "min_score" : self.distance_cutoff,
            "boost_mode":"replace"
        }
        }
        }
        # label is useful or 'other(10000)'
        # if 'other', just search 'other', then return directly
        # else, first search top_1(search) or top_2(search) or top_3(search) == top_1 or top_2 or top_3
        # then search 'other' if search result number is not enough

        if top_1 == LABEL_OTHER:
            now = datetime.now()
            delta = timedelta(days=30)
            body["query"]["function_score"]["query"]["bool"]["must"] = [{"multi_match": {
                            "query": top_1,
                            "fields": ["top_1"]
                        }},
                        {"range": {
                            "timestamp": {
                                "gt": now-delta,
                                "lt": now
                            }
                        }}]
            es_res = self.es.search(index=self.index,
                                 doc_type=self.doc_type,
                                 size=term_after*5,
                                 body=body,
                                 _source_exclude=['signature', 'timestamp', 'top_1', 'top_2', 'top_3'],
                                 timeout=self.timeout)['hits']
            res = es_res['hits']
        else:
            # body["query"]["function_score"]["query"]["bool"]["should"] = [{"multi_match": {
            #                 "query": top_1,
            #                 "fields": ["top_1", "top_2", "top_3"]
            #             }},
            #             {"multi_match": {
            #                 "query": top_2,
            #                 "fields": ["top_1", "top_2", "top_3"]
            #             }},
            #             {"multi_match": {
            #                 "query": top_3,
            #                 "fields": ["top_1", "top_2", "top_3"]
            #             }}]
            now = datetime.now()
            delta = timedelta(days=30)
            body["query"]["function_score"]["query"]["bool"]["must"] = [{"multi_match": {
                "query": top_1,
                "fields": ["top_1", "top_2", "top_3"]
                }},
                {"range": {
                    "timestamp": {
                        "gt": now-delta,
                        "lt": now
                    }
                }}]
            es_res = self.es.search(index=self.index,
                                 doc_type=self.doc_type,
                                 size=term_after*5,
                                 body=body,
                                 _source_exclude=['signature', 'timestamp', 'top_1', 'top_2', 'top_3'],
                                 timeout=self.timeout)
            res = es_res['hits']['hits']
            # total = es_res['total']
            # if total < term_after:
            #     body["query"]["function_score"]["query"]["bool"]["should"] = [{"multi_match": {
            #         "query": LABEL_OTHER,
            #         "fields": ["top_1"]
            #     }}]
            #     body["terminate_after"] = term_after - total
            #     es_res = self.es.search(index=self.index,
            #                             doc_type=self.doc_type,
            #                             size=term_after - total,
            #                             body=body,
            #                             _source_exclude=['signature', 'timestamp', 'thumbnail', 'top_1', 'top_2', 'top_3'],
            #                             timeout=self.timeout)['hits']['hits']
            #     res += es_res
        return res

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
