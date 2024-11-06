from setuptools import setup
import os

# Read the version but avoid importing the __init__.py since
# we might not have all dependencies installed
with open(os.path.join(os.path.dirname(__file__), "sskit", "version.py")) as fp:
    exec(fp.read())

if __name__ == '__main__':
    setup(
        name='sskit',
        description='Spiideo Scenes development Kit',
        long_description='''
            Development Kit for the Spiideo SoccerNet SynLoc task and Challange
        ''',
        version=__version__,
        packages=['sskit'],
        zip_safe=False,
        url='https://github.com/Spiideo/sskit',
        author='Hakan Ardo',
        author_email='hakan.ardo@spiideo.com',
        license='MIT',
    )
