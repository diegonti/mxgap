import os
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mxgap",                              # Name of your package
    version="0.1",                                  # Version of your package
    description="ML models to predict terminated MXene bandgaps.",
    long_description=long_description,              # Long description from README
    long_description_content_type="text/markdown",
    author="Diego Ontiveros",
    author_email="diego.ontiveros@ub.edu",
    url="https://github.com/diegonti/mxgap",   # Project URL (e.g., GitHub)
    license="MIT",
    license_file = "LICENSE",

    packages=["mxgap"],
    package_dir={"mxgap":"mxgap"},
    include_package_data=True,                      # This will include files from MANIFEST.in
    install_requires=[                              # List dependencies your package requires
        'scikit-learn==1.4.1.post1',
        'numpy',
        'ase',
        'pymatgen',
        'pytest',
    ],
    python_requires='>=3.9',           # Python version requirement
    entry_points = {"console_scripts": ["mxgap = mxgap.cli:cli"]},
    classifiers=[                      # Metadata about your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)