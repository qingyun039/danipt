import os
import sys

HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(os.path.join(HOME, 'danipt'))
import danipt

import json
import pandas as pd

EXAMPLE = os.path.join(HOME, 'example/baseline')
with open(os.path.join(EXAMPLE, 'baseline.json')) as f:
    baseline = json.load(f)
bins_attr = pd.read_pickle(os.path.join(EXAMPLE, 'bins_attr.pkl'))

bins_filter = pd.Series(baseline['bins_filter'])
use_bins = bins_filter == 'PASS'

bins_values = pd.DataFrame({'chrom': bins_attr.loc[use_bins, 'chrom'], 'values': baseline['bins_baseline']['mean']})
danipt.plot_bins_values(bins_values, 'bins_means')