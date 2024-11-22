from setuptools import setup, find_packages

setup(
    name="deepgaze_pytorch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "scipy",
        "tqdm",
        "tensorboard"
    ]
) 
