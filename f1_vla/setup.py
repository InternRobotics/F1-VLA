from setuptools import setup, find_packages

setup(
    name='f1_vla',
    version='0.1.0',
    author='aopolin-lv',
    author_email='aopolin.ii@gmail.com',
    description='F1: A Vision Language Action Model Bridging Understanding and Generation to Actions',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aopolin-lv/F1-VLA',
    packages=find_packages(),
    install_requires=[
        'lerobot',
        'numpy',
        'omegaconf',
        'transformers==4.49.0',
        'numpy',
        'accelerate',
        'safetensors==0.5.3',
        'tensorboard',
        'typo',
        'opencv-python-headless',
        'pytest',
        'natsort',
        'setuptools==58.2.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)