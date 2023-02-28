from setuptools import setup
  
setup(
    name='toolbox_02450',
    version='0.1',
    description='Package for 02450 Introduction to machine learning and data mining',
    author='John Doe',
    author_email='jdoe@example.com',
    packages=['statistics', 'similarity', 'categoric2numeric', 'bin_classifier_ensemble'],
    install_requires=[
        'numpy',
        'scipy',
    ],
)
