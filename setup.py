from proseco import __version__

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='proseco',
      author='Tom Aldcroft',
      description='Probabilistic star evaluation and catalog optimization',
      author_email='taldcroft@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      packages=['proseco', 'proseco.tests'],
      package_data={'proseco': ['*index_template.html']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
