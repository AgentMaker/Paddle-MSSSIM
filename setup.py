import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paddle_msssim",
    version="0.0.1",
    author="jm12138",
    description="Fast and differentiable MS-SSIM and SSIM for paddle.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AgentMaker/Paddle-MSSSIM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)