# HAIDI-Graphs

Trying to find harmful content in the training data of LLMs by looking at their outputs. 

The code allows to run inference from models using the module `transformers`, and compute the perplexity of the output tokens. It can also generate visualization of the perplexity per token.



## Files

```
├── outputs/              # Contains output txt files from the LLMs (prompts variations, LLM outputs, ...)
│
├── report/               # Documentation and project reports (
│
├── run_scripts/          # Scripts to run the project on the cluster (.sh shell or slurm scripts)
│
├── src/                  # Source code files (.py)
```

## Cluster setup

On izar (EPFL scitas), with the following 

```
module load gcc python openmpi py-torch
virtualenv --system-site-packages .venv
source .venv/bin/activate
pip install --no-cache-dir requirements.txt 
```
More info in [the documentation](https://scitas-doc.epfl.ch/user-guide/software/python/python-venv/)

## Overleaf report link
The report can be seen [here](https://www.overleaf.com/read/mdhmztdpjvrd#749e7e) (the repo might not be constantly updated...)

## Credits

Semester project at EPFL of Arthur Wuhrmann, under the supervision of Antoine Bosselut, Anastasiia Kucherenko and Andrei Kucharavy.
