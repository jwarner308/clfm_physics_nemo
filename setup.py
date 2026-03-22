from setuptools import setup, find_packages

setup(
    name = 'random-field-clfm',
    version = '1.0',
    packages = find_packages(include=['clfm', 'clfm.*', 'clfm_pn', 'clfm_pn.*']),
)
