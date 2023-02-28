import setuptools
  
setuptools.setup(
    name='toolbox02450',
    version='0.1',
    description='Package for 02450 Introduction to machine learning and data mining',
    author='John Doe',
    author_email='jdoe@example.com',
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="./src"),
    #['statistics', 'similarity', 'categoric2numeric', 'bin_classifier_ensemble'],
    install_requires=[
        'numpy',
        'scipy',
    ],
    python_requires='>=3.8'
)
