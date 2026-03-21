from setuptools import setup, find_packages

# When building inside catkin, catkin_pkg is available.
# When installing outside catkin (e.g. for tests with MockTransport),
# fall back to plain setuptools.
try:
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(
        packages=find_packages("src"),
        package_dir={"": "src"},
    )
    setup(**d)
except ImportError:
    setup(
        name="milliampere_env",
        version="0.5.0",
        packages=find_packages("src"),
        package_dir={"": "src"},
        install_requires=[
            "gymnasium",
            "numpy",
            "pydantic>=2.0",
            "pyyaml",
            "milliampere-dp",
        ],
    )
