# Enhancing-Idiomatic-Representation-in-Multiple-Languages
My working implementation of the ACL paper 'Enhancing Idiomatic Representation in Multiple Languages via an Adaptive Contrastive Triplet Loss'

## Running the code

Requirements
- GPU with 24gb memory

Create a python virtual environment
```
python -m venv .venv
```
Activate it (bash or csh)
```
# For bash
source .venv/bin/activate
# For cshell (installed on UCL machines - not tested)
source .venv/bin/activate.csh
```
Install dependencies
```
pip install -r requirements.txt
```
Run the training
```
./run.sh
```
