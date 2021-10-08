import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ebemse',  
     version='0.1',
     author="Andre D. L. Zanchetta",
     author_email="adlzanchetta@gmail.com",
     description="A package for selecting ensemble members using entropy theory",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://gitlab.com/adlzanchetta_phd-codes/ebemse",
     packages=setuptools.find_packages(),
     package_dir = {'': 'src'},
     py_modules = ['ebemse'],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )