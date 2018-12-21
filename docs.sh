#!/bin/bash

cd ./docs
sphinx-apidoc -f -o source/ ../adversarials
make html
cd ./build/html
git add .
git commit -m "rebuilt docs"
git checkout -b gh-pages
git push origin gh-pages