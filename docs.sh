#!/bin/bash

git clone https://${GITHUB_TOKEN}@github.com/deven96/Simple_GAN.git
cd ./Simple_GAN/docs
sphinx-apidoc -f -o source/ ../adversarials
make html
cd ./build/html
git config --global push.default simple
git config --global user.email "travis@travis-ci.com"
git config --global user.name "Travis CI"
git add .
git commit -m "rebuilt docs"
git checkout gh-pages
git push -q https://${GITHUB_TOKEN}@github.com/deven96/Simple_GAN.git gh-pages --force