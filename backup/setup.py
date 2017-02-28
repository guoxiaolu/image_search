# import io
# import os
# import re
#
# from setuptools import setup, find_packages
#
#
# def read(*names, **kwargs):
#     with io.open(
#         os.path.join(os.path.dirname(__file__), *names),
#         encoding=kwargs.get('encoding', 'utf8')
#     ) as fp:
#         return fp.read()
#
#
# def find_version(*file_paths):
#     version_file = read(*file_paths)
#     version_match = re.search(
#         r'^__version__ = [\'"]([^\'"]*)[\'"]', version_file, re.M)
#     if version_match:
#         return version_match.group(1)
#     raise RuntimeError('Unable to find version string.')
#
# setup(
#     name='image_search',
#     version=find_version('image_search', '__init__.py'),
#     description='image_search is a simple package for finding approximate '\
#                 'image matches from a corpus.',
#     author='Lavector',
#     zip_safe=True,
#
#     classifiers=[
#         'Development Status :: 4 - Beta',
#         'Intended Audience :: Developers',
#         'Topic :: Database',
#         'Topic :: Database :: Database Engines/Servers',
#         'Topic :: Software Development',
#         'Natural Language :: English',
#         'Programming Language :: Python :: 2.7.6',
#         'Operating System :: MacOS :: MacOS X',
#         'Operating System :: POSIX :: Linux',
#         'Topic :: Multimedia :: Graphics',
#     ],
#
#     packages=find_packages(),
#
#     install_requires=[
#         'scikit-image>=0.12,<0.13',
#         'elasticsearch>=2.3,<2.4',
#     ],
#     extras_require={
#         'extra': ['cairosvg>1,<2'],
#     },
# )
#
