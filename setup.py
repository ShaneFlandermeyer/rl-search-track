from setuptools import setup, find_packages

setup(
    name='rl-search-track',
    version='0.0.0',    
    description='Relational RL for Autonomous Search-and-Track',
    url='https://github.com/ShaneFlandermeyer/rl-search-track',
    author='Shane Flandermeyer',
    author_email='shaneflandermeyer@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'motpy @ git+https://github.com/ShaneFlandermeyer/MOTpy.git@develop',
      'tdmpc2-jax @ git+https://github.com/ShaneFlandermeyer/tdmpc2-jax.git@develop',
      'igraph',
    ],

)
