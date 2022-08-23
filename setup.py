from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(
    name="proseco",
    author="Tom Aldcroft",
    description="Probabilistic star evaluation and catalog optimization",
    author_email="taldcroft@cfa.harvard.edu",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    zip_safe=False,
    packages=["proseco", "proseco.tests"],
    package_data={"proseco": ["index_template*.html", "maxmags.npz"]},
    tests_require=["pytest"],
    cmdclass=cmdclass,
)
