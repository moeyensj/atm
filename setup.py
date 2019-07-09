from setuptools import setup

setup(
   name="atm",
   version="0.9.0",
   license="BSD 3-Clause License",
   author="Joachim Moeyens",
   author_email="moeyensj@uw.edu",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "astropy",
        "pymc3",
        "theano",
        "corner",
        "astroML",
        "pytest",
        "pytest-cov"
   ],
   url="https://github.com/moeyensj/atm",
   packages=["atm"],
   package_dir={"atm": "atm"},
   package_data={"atm": ["models/tables/*.npz"]},
   setup_requires=["pytest-runner"],
   tests_require=["pytest"],
)