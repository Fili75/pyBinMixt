from setuptools import setup


setup(
    name="pyBinMixt",
    version="1.0.0",  # version number is set in pyMixtComp/_version.py
    author="Filippo Antonazzo",
    description="Frugal bin-marginal Gaussian model-based clustering",
    keywords=["clustering", "mixture model", "heterogeneous", "missing data"],
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    zip_safe=False,
    package_data={"binmixtC": ["binmixtC.so"]},
    install_requires=["numpy", "scikit-learn"],
)
