cd ./docs
make html
cd ../adversarial-docs/html
git add .
git commit -m "rebuilt docs"
git push origin gh-pages