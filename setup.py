from setuptools import setup, find_packages
import importlib


def get_requirements():
    with open('requirements.txt', 'r') as f:
        ret = [line.strip() for line in f.readlines()]
        print("requirements:", ret)
    return ret


setup(
    name='deepthulac',
    # packages = ['deepthulac'], # this must be the same as the name above
    version='0.0.0',
    description='A High-Performance Lexical Analyzer for Chinese',
    author='THUNLP',
    url='https://github.com/thunlp/DeepTHULAC',
    author_email='chengzl22@mails.tsinghua.edu.cn',
    download_url='https://github.com/thunlp/DeepTHULAC/archive/master.zip',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    keywords=['segmentation', 'Chinese', 'lexical'],
    python_requires=">=3.6.0",
    install_requires=get_requirements(),
    packages=find_packages(),
    package_data={'': ['*.yaml']}
)


required_list = []
for package in required_list:
    try:
        m = importlib.import_module(package)
    except ModuleNotFoundError:
        print("\n"+"="*30+"  WARNING  "+"="*30)
        print(f"{package} is not found on your environment, please install it manually.")
        print("We do not install it for you because the environment sometimes needs special care.")

try:
    import torch
except Exception:
    hint = "\nPlease install torch manually before installing deepthulac.\n"\
        + "See https://pytorch.org/get-started/locally/\n"
    raise Exception(hint)
