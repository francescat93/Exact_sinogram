from setuptools import setup

setup(name='exact_sinogram',
      version='0.1.2',
      description='A Python package for the computation of the exact sinogram of different cathegories of phantoms for CT reconstruciton',
      url='http://github.com/francescat93/Exact_sinogram',
      author='Monica Dessole, Marta Gatto, Davide Poggiali, Francesca Tedeschi',
      author_email='davide.poggiali@unipd.it',
      license='MIT',
      packages=['exact_sinogram'],
      install_requires=[
          'matplotlib', 'numpy>=1.13', 'scipy',
          'scikit-image'
      ],
      zip_safe=False)
