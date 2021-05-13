from setuptools import setup

setup(name='gComm',
      version='1.0',
      description='environment for grounded language acquisition',
      keywords='environment, generalization, agent, rl, gym, embodied, communication',
      install_requires=[
          'gym>=0.9.6',
          'numpy>=1.18.1',
          'pyqt5>=5.10.1',
          "torch>=0.4.1",
          'matplotlib',
          'pronounceable',
          'six',
          'regex',
          'pandas',
          'gym_minigrid @ https://github.com/maximecb/gym-minigrid/archive/master.zip'
      ],
      )
