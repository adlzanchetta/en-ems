#!/bin/bash

cd ..
python setup.py bdist_wheel
cd test
pip install ../dist/ebemse-0.1-py3-none-any.whl --force-reinstall

python test.py

echo "ok"
