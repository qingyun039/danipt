#!/bin/bash

set -e
errors=0

# Run unit tests
python daan_nipt/daan_nipt_test.py || {
    echo "'python python/daan_nipt/daan_nipt_test.py' failed"
    let errors+=1
}

# Check program style
pylint -E daan_nipt/*.py || {
    echo 'pylint -E daan_nipt/*.py failed'
    let errors+=1
}

[ "$errors" -gt 0 ] && {
    echo "There were $errors errors found"
    exit 1
}

echo "Ok : Python specific tests"
