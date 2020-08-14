from setuptools import setup

setup(
    name="active_learning_cfd",
    version="1.0",
    description="Active learning based regression for CFD cases",
    url="",
    author="Gabriel Goncalves",
    author_email="g.goncalves18@imperial.ac.uk",
    license="MIT",
    packages=["active_learning_cfd"],
    install_requires=["numpy", "matplotlib", "modAL", "PyFoam"],
    zip_safe=False,
)
