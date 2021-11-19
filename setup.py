import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="en-ems",
    version="0.2.0",
    url="https://gitlab.com/adlzanchetta_phd-codes/en-ems",
    author="Andre D. L. Zanchetta",
    author_email="adlzanchetta@gmail.com",
    description="A package for selecting ensemble members using entropy theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["pyitlib==0.2.2", "scikit-learn==0.23"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    include_package_data=True,
    package_data={"": ["data/*.csv", "example_data/*.pickle"]},
)
