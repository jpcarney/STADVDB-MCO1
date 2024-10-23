from setuptools import setup

setup(
    name='STADVDB-MCO1',
    version='1.0',
    packages=['MCO1'],
    url='https://github.com/jpcarney/STADVDB-MCO1',
    license='',
    author='John Paul C. Carney',
    author_email='jpcarney@dlsu.edu.ph',
    description='OLAP Web Application for STADVDB MCO1 Output',
    install_requires = ['numpy', 'pandas', 'dash', 'kagglehub', 'pymysql'],
)
