from distutils.core import setup
from setuptools import setup, Extension, find_namespace_packages

setup(
    name='CoupledRadioInterferometer',
    version='0.1.0',
    author='A. T. Josaitis',
    author_email='atj28@cam.ac.uk',
    packages=find_namespace_packages(),
    url='http://pypi.python.org/pypi/CoupledRadioInterferometer/',
    license='LICENSE.txt',
    description='For calculating and analyzing array element coupling effects in radio interferometers',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy>=1.15, <1.21',
        'matplotlib>=2.2'
        'pyuvdata',
        'astropy>=2.0',
        'healpy',
        'ruamel.yaml',
        'healvis @ git+git://github.com/rasg-affiliates/healvis@setup_all_bls_and_peak_norm_beam#egg=healvis',
        'uvtools @ git+git://github.com/HERA-Team/uvtools@plot_range_control#egg=uvtools',
        'hera_cal @ git+git://github.com/HERA-Team/hera_cal#egg=hera_cal',
        'hera_pspec @ git+git://github.com/alphatangojuliett/hera_pspec@atj-patch-1_uvtools_branch#egg=hera_pspec',
    ],
    include_package_data = True,
)
