#!/bin/bash

# pip install -r requirements.txt;
cat requirements.txt | xargs -n 1 pip install

# compile custom operators
cd pointnet2
python setup.py install --user;
cd ..;

# echo $pwd;