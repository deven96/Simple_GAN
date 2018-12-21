#!/bin/bash

cd ./docs
make html
cd ./build/html
git add .
git commit -m "rebuilt docs"
git push origin gh-pages