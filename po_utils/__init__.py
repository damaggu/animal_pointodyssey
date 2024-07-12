import os

current_dir = os.path.dirname(os.path.realpath(__file__))
__path__.append(os.path.join(
    os.path.abspath(os.path.join(current_dir, '..')),
    'subPackageB'
))
