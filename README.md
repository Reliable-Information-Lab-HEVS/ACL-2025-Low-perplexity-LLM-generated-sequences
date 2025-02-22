# HAIDI-Graphs

Trying to find harmful content in the training data of LLMs by looking at their outputs. 

## Cluster setup

On izar (EPFL scitas), with the following 

```
module load gcc python openmpi py-torch
virtualenv --system-site-packages .venv
source .venv/bin/activate
pip install --no-cache-dir transformers
```
More info in [the documentation](https://scitas-doc.epfl.ch/user-guide/software/python/python-venv/)

## Overleaf report link
The report can be seen [here](https://www.overleaf.com/read/mdhmztdpjvrd#749e7e) (the repo might not be constantly updated...)

## Credits

Semester project at EPFL of Arthur Wuhrmann, under the supervision of Antoine Bosselut, Anastasiia Kucherenko and Andrei Kucharavy.
