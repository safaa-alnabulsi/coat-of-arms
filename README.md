# coat-of-arms

Work in progress 

![coa](images/coa-example.png)

## Setup Local Environment

1- Get the repo

    git clone git@github.com:safaa-alnabulsi/coat-of-arms.git
    cd coat-of-arms

2- Create virtual enviroment

    conda create --name thesis-py38 python=3.8
    conda activate thesis-py38
    conda install --file requirements.txt
    pip install automata-lib
    jupyter notebook

3- to run tests

    pytest

4- to see it visually (needs a dataset in a folder named `data/cropped_coas/out` ):
    
    streamlit run view_crops.py

## The Automata

The visual representation of the implemented automata in [LabelCheckerAutomata](src/label_checker_automata.py)
![alt automata](images/automata2.jpg)

The previous simple automata: 
![alt automata](images/simple-automata2.jpg)

## References:
- automata-lib: https://pypi.org/project/automata-lib/#class-dfafa
