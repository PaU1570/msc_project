# Guide: How to run ipkiss python scripts
## Step 1: Create a conda environment
We need a python environment running python version 2.7. To create one (with the name ```ipkiss_env```), run:
```
conda create --name ipkiss_env python=2.7
```

Then activate the environment:
```
conda activate ipkiss_env
```
You will see the terminal prompt starts with ```(ipkiss_env)```; this means the environment is active. We can check the python version with ```python --version``` (should print ```2.7.x```), as well as the location of the python executable that we are using with ```which python``` (should print a path in the conda environment we just created, probably ```/path/to/your/home/.conda/envs/ipkiss_env/bin/python```).

## Step 2: Install ipkiss
We will install ipkiss from the following github repository: https://github.com/jtambasco/ipkiss.
First clone the repo to your machine (can be anywhere, like ```/scratch```):
```
git clone https://github.com/jtambasco/ipkiss.git
```
Move to the directory where the repository was downloaded, then install with pip:
```
cd ipkiss
pip install .
```
You should see a message saying "Successfully installed IPKISS-2.4-ce".

## Step 3: Install other required packages
Use pip to install the rest of the required packages:
```
pip install numpy shapely descartes
```

## Step 4: Run
Now we should be able to successfuly run scripts to generate gds files.
```
python your_script.py
```
__Note:__ Every new session, make sure to activate the environment with ```conda activate ipkiss_env```. To go back to your normal python environment, deactivate with ```conda deactivate```.