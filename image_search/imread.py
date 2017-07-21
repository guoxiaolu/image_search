import urllib2
import re
import os
import tempfile
import six
from contextlib import contextmanager

HEADER = {
	'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
	'Accept-Encoding':'gzip, deflate, sdch',
	'Accept-Language':'zh-CN,zh;q=0.8,en;q=0.6',
	'Cache-Control':'max-age=0',
	'Connection':'keep-alive',
	'Upgrade-Insecure-Requests':'1',
	'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')
def is_url(filename):
    """Return True if string is an http or ftp path."""
    return (isinstance(filename, six.string_types) and
            URL_REGEX.match(filename) is not None)

@contextmanager
def file_or_url_context(resource_name):
    """Yield name of file from the given resource (i.e. file or url)."""
    if is_url(resource_name):
        _, ext = os.path.splitext(resource_name)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                proto, rest = urllib2.splittype(resource_name)
                HOST, rest = urllib2.splithost(rest)
                HEADER['Host'] = HOST
                req = urllib2.Request(resource_name, headers=HEADER)
                u = urllib2.urlopen(req, timeout=4)
                f.write(u.read())
            # f must be closed before yielding
            yield f.name
        finally:
            os.remove(f.name)
    else:
        yield resource_name

# from skimage.io import imread
# fname = "https://imgsa.baidu.com/forum/w%3D580/sign=e960450646086e066aa83f4332097b5a/36844b59252dd42a79cdd89c093b5bb5c8eab874.jpg"
# with file_or_url_context(fname) as f:
#     img = imread(f)
#     print img.shape