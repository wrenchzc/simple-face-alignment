import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple face alignment",
    version="0.0.1",
    author="Zhang, Chi",
    author_email="wrench@outlook.com",
    description="a simple implmentation for face align",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wrenchzc/simple-face-alignment",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    install_requires=[
        "opencv-python>=4.1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    keywords="face alignment",
)
