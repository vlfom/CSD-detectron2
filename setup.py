from setuptools import find_packages, setup

setup(
    name="csd",
    version="0.1",
    description="An unofficial implementation of Consistency-Based Semi-Supervised Learning for Object Detection method in Detectron2",
    url="https://github.com/vlfom/CSD-detectron2/",
    author="Vladimir Fomenko",
    author_email="vlfomenk@gmail.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
)
