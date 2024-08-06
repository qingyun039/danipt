#!/usr/bin/env python

from distutils.core import setup

LONG_DESCRIPTION = \
'''NIPT分析，包含建立基线，性别判断，NIPT，NIPTplus，胎儿浓度的子程序

'''


setup(
    name='danipt',
    version='0.1.0.0',
    author='qingyun039',
    author_email='DAAN_NIPT_EMAIL',
    packages=['danipt'],
    package_dir={'danipt': 'danipt'},
    entry_points={
        'console_scripts': ['danipt = danipt.danipt:main']
    },
    url='https://github.com/qingyun039/danipt',
    license='LICENSE',
    description=('A prototypical bioinformatics command line tool'),
    long_description=(LONG_DESCRIPTION),
    install_requires=["pandas"],
)
