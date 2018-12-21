#!/bin/bash

cd ./docs
sphinx-apidoc -f source/ ../adversarials
make html
cd ./build/html
git add .
git commit -m "rebuilt docs"
git push origin gh-pages