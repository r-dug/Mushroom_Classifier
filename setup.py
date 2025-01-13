from setuptools import find_packages
from setuptools import setup
with open("requirements.txt") as requirements:
    REQUIRED_PACKAGES =[line.strip('\n') for line in requirements.readlines()]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application.'
)