from setuptools import setup, find_packages

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='road_accident_predictor',
      version="0.1",
      description="MVP Road Accident Severity Predictor API",
      packages=find_packages(),
      install_requires=requirements
)
