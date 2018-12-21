#!/bin/bash

cd ./docs
sphinx-apidoc -f -o source/ ../adversarials
make html
cd ..
git config --global push.default simple
git config --global user.email "travis@travis-ci.com"
git config --global user.name "Travis CI"
git checkout -b gh-pages
#remove existing files
shopt -s extglob
rm -r ./!(docs/html)/
cp -R ./docs/html/* .
rm -r ./docs
git commit -am "rebuilt docs"
git push -q https://${GITHUB_TOKEN}@github.com/deven96/Simple_GAN.git gh-pages --force