import os
import sys
from os.path import join

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

lib_path = join(this_dir, 'S3FD/layers')
add_path(lib_path)
lib_path = join(this_dir, 'S3FD/functions')
add_path(lib_path)
lib_path = join(this_dir, 'S3FD/data')
add_path(lib_path)
lib_path = join(this_dir, 'S3FD/utils')
add_path(lib_path)