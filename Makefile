.PHONY: docs git

docs: 
	mkdocs gh-deploy --force

git: docs
	git add .