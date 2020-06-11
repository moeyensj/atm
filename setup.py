from setuptools import setup

setup(
   name="atm",
   license="BSD 3-Clause License",
   author="Joachim Moeyens",
   author_email="moeyensj@uw.edu",
   long_description=open("README.md").read(),
   long_description_content_type="text/markdown",
   url="https://github.com/moeyensj/atm",
   packages=["atm"],
   package_dir={"atm": "atm"},
   package_data={"atm": ["models/tables/*.npz"]},
   use_scm_version=True,
   setup_requires=["setuptools_scm","pytest-runner"],
   tests_require=["pytest"],
)
