# Mesh-Informed Neural Operator : A Transformer Generative Approach


### [MINO Paper](https://www.arxiv.org/abs/2506.16656) 
by Yaozhong Shi, Zachary E. Ross, Domniki Asimaki, Kamyar Azizzadenesheli

## Model architecture
![image](fig/model.PNG)

## Inference and zero-shot generation
![image](fig/inference.PNG)

## Setup
First download the processed dataset from [https://huggingface.co/datasets/Yaozhong/MINO](https://huggingface.co/datasets/Yaozhong/MINO), unzip it and place all files in the ``dataset`` folder 

To set up the environment, create a conda environment

```
# clone project
git clone https://github.com/yzshi5/MINO.git
cd MINO

# create conda environment
conda env create -f environment.yml

# Activate the `mino` environment
conda activate mino
```




Install the `ipykernel` to run the code in a jupyter notebook
```
conda install -c anaconda ipykernel

pip install ipykernel

python -m ipykernel install --user --name=mino
```
