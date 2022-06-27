from setuptools import setup


setup(
    name="pyBinMixt",
    version="1.0.0",  # version number is set in pyMixtComp/_version.py
    author="Tua mamma",
    description="Fa cose.",
    keywords=["clustering", "mixture model", "heterogeneous", "missing data"],
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    zip_safe=False,
    install_requires=["numpy", "scikit-learn"],
)