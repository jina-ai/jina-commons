from setuptools import find_packages
import setuptools

setuptools.setup(
    name="jina-commons",
    version="0.0.1",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Shared functions for Jina Executors",
    url="https://github.com/jina-ai/jina-commons",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(include='jina_commons.*'),
    python_requires=">=3.7",
    zip_safe=False,
    install_requires=[open('requirements.txt').readlines()]
)
