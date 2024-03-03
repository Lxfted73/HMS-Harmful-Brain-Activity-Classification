# Project Overview

**HMS-Harmful-Brain-Activity-Classification**: Classify EEGs into one of 6 classifications.

## Methodology

1. After feature engineering is complete, complete model development of several different models, complete model ensemble, complete backtesting and various other model implementations, all while including data visualization and documentation along the way.

## Group Workflow

### 1. New Modules 
- If you download a new module like tensorflow, update the the requirements.txt like this:
	numpy
	pandas
	matplotlib
	tensorflow
- Run pip3 install -r requirements.txt
	- This ensure that you have all of the required modules
 	- Run this prior to working

### 2. Commits

- In each commit, please provide a 1 sentence summary of what you did, for example:
    - Fixed bugs in `Time-Domain-Features.py`.
    - Added `CNN_RNN_Models.py`. Still needs data visualization and hyperparameterization.
    - Modified `Frequency-Domain-Features.py` to include.
    - Added documentation to `Time-Domain-Features.py`.

### 2. Documentation

- Use ChatGPT for line by line comments.
- Use ChatGPT for entire .py summary to place at the top of the python file.
- Use ChatGPT for function summaries.

#### Examples:

- (Copy-Paste Code) - ChatGPT please write me line by line comments for this code.
- (Copy-Paste Code) - ChatGPT please write an entire summary for my code at the top of the python file.
- (Copy-Paste Code) - ChatGPT please include summaries for each of my functions including a: 
      description of what is happening, the arguments, and the outputs.

Modify each of these as you see fit.

### 4. Folder Structure

- Feature-Engineering
  - Time-Domain-Features.py
  - Frequency-Domain-Features.py
  - Time-Frequency-Domain-Features
- Models (Sub folders can be made based on the type of implementations)
  - NN_Model
    - CNN-RNN-Model
      	- CNN_RNN_Model.py
      	- Model_Hyperparameterization.py
      	- Example.py
  - Support-Vector-Machines
      	- Example
      	- Example
  - Gradient-Boosting-Machines
    	- Example
  		- Example
  - Random-Forest-Decision-Tree
    	- Example
    	- Example
- Model Ensemble
  - Example

