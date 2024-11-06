
#!/bin/sh

set  -e

if [ ! -z "`git diff`" ]; then
    echo Please commit first
    exit
fi

git tag -a v$(python -c 'import setup; print(setup.__version__)')  -m release
git push
git push --tags

