'''
Module      : Main
Description : The main entry point for the program.
Copyright   : (c) qingyun039, 18 Oct 2021
License     : MIT
Maintainer  : DANIPT_EMAIL
Portability : POSIX

The program reads one or more input FASTA files. For each file it computes a
variety of statistics, and then prints a summary of the statistics as output.
'''

from argparse import ArgumentParser
import sys
import logging
import pkg_resources

import os
import json
import multiprocessing
import subprocess
from collections import namedtuple
from io import BytesIO

import pybedtools
import pyBigWig
import pysam

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.extmath import safe_sparse_dot
from scipy.signal import argrelextrema

#------------------------------------------ 变量默认值或者全局变量 --------------------------------------------

EXIT_DATAFRAME_ERROR = 1
FILE_NO_EXISTS = 2
EXIT_VALUE_ERROR = 3
DEFAULT_VERBOSE = False
PROGRAM_NAME = "danipt"

DEFAULT_GENOME = 'hg19'
DEFAULT_THREADS = 8
DEFAULT_BINSIZE = 50000
SCALE_NUMBER = 1000000   # 1M 对结果没有影响
MIN_READ_COUNT = 1000000
N_DISP = 3
MIN_N_SAMPLES = 20
MIN_BINSIZE = 10000
P_VALUE_THRESHOLD = 0.01
MALES_YFRAC = 2000

SEQDIR = '/data/Sequencer/V1004100210068'
QUAL = 20
MAPQ = 10
ADAPTER = 'AAGTCGGAGGCCAAGCGGTCTTAGGAAGACAA'
REF = '/data/database/aliger-index/bwa/hg19.p13.plusMT.no_alt_analysis_set.fa'

AUTOSOME = [ 'chr'+str(i) for i in range(1,23) ]
CHRX = 'chrX'
CHRY = 'chrY'
CHROMS = AUTOSOME + [CHRX, CHRY]

try:
    PROGRAM_VERSION = pkg_resources.require(PROGRAM_NAME)[0].version
except pkg_resources.DistributionNotFound:
    PROGRAM_VERSION = "undefined_version"

#-------------------------------------- 错误处理和日志信息 ----------------------------------------------

def exit_with_error(message, exit_status):
    '''Print an error message to stderr, prefixed by the program name and 'ERROR'.
    Then exit program with supplied exit status.

    Arguments:
        message: an error message as a string.
        exit_status: a positive integer representing the exit status of the
            program.
    '''
    logging.error(message)
    print("{} ERROR: {}, exiting".format(PROGRAM_NAME, message), file=sys.stderr)
    sys.exit(exit_status)

def init_logging(log_filename):
    '''If the log_filename is defined, then
    initialise the logging facility, and write log statement
    indicating the program has started, and also write out the
    command line from sys.argv

    Arguments:
        log_filename: either None, if logging is not required, or the
            string name of the log file to write to
    Result:
        None
    '''
    if log_filename is not None:
        logging.basicConfig(filename=log_filename,
                            level=logging.DEBUG,
                            filemode='w',
                            format='%(asctime)s %(levelname)s - %(message)s',
                            datefmt="%Y-%m-%dT%H:%M:%S%z")
        logging.info('program started')
        logging.info('command line: %s', ' '.join(sys.argv))
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    class NoParsingFilter(logging.Filter):
        def filter(self, record):
            if record.getMessage().startswith('findfont'):
                return False
            else:
                return True

    logging.getLogger().addFilter(NoParsingFilter())

#-------------------------------------- 命令行参数 ----------------------------------------------------

def parse_args():
    '''Parse command line arguments.
    Returns Options object with command line argument values as attributes.
    Will exit the program on a command line error.
    '''
    description = 'NIPT or NIPT plus'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '-t', '--threads',
        default = DEFAULT_THREADS,
        type=int,
        help='并行进程数量 (default {})'.format(DEFAULT_THREADS))
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s ' + PROGRAM_VERSION
    )
    parser.add_argument(
        '--log',
        metavar='LOG_FILE',
        type=str,
        help='record program progress in LOG_FILE'
    )

    subparsers = parser.add_subparsers(
        title = 'subcommands',
        required = True,
        metavar = ''
    )

    parser_preprocess = subparsers.add_parser('preprocess', help = '数据预处理')
    parser_preprocess.set_defaults(subcommand = preprocess)
    parser_preprocess.add_argument(
        '-q', '--qual',
        help = "碱基质量值过滤",
        default = 20
    )
    parser_preprocess.add_argument(
        '-Q', '--mapq',
        help = '比对质量值过滤',
        default = '10'
    )
    parser_preprocess.add_argument(
        '-a', '--adapter',
        help = '测序接头序列',
        default= 'AAGTCGGAGGCCAAGCGGTCTTAGGAAGACAA'
    )
    parser_preprocess.add_argument(
        '--rename',
        help = '输出使用样本名代替文库名',
        action = 'store_true'
    )
    parser_preprocess.add_argument(
        '--seqdir',
        help = '测序仪下机数据地址',
        default = SEQDIR
    )
    parser_preprocess.add_argument(
        '-r', '--ref',
        help = '参考基因组bwa索引',
        default = '/data/database/aliger-index/bwa/hg19.p13.plusMT.no_alt_analysis_set.fa'
    )
    parser_preprocess.add_argument(
        'samplesheet_or_fqfiles',
        nargs = '+',
        help = '样本表格',
    )

    parser_baseline = subparsers.add_parser('baseline', help = "建立基线")
    parser_baseline.set_defaults(subcommand = build_baseline)
    parser_baseline.add_argument(
        '-g', '--genome',
        help = "基因组版本，可选hg18, hg19, hg38 (default {})".format(DEFAULT_GENOME),
        default = 'hg19',
        choices = ['hg18', 'hg19', 'hg38']
    )
    parser_baseline.add_argument(
        '-r', '--ref',
        help = '基因组参考序列fasta文件',
        type = str,
        nargs = '?',
        default = '/data/database/ucsc-goldenPath/hg19/hg19.fa'
    )
    parser_baseline.add_argument(
        '-b', '--binsize',
        help = '切分的bin大小 (default {})'.format(DEFAULT_BINSIZE),
        type = int,
        default = DEFAULT_BINSIZE
    )
    parser_baseline.add_argument(
        '-o', '--outdir',
        help = "输出目录路径",
        default = '.'
    )
    parser_baseline.add_argument(
        '--align_bw',
        help = '可比对性BigWig文件',
        type = str,
        nargs = '?',
        default = '/data/database/ucsc-goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeCrgMapabilityAlign50mer.bigWig'
    )
    parser_baseline.add_argument(
        '--filter_sample',
        help = '过滤数据量异常的样本',
        action = 'store_true'
    )
    parser_baseline.add_argument(
        '--gender_cutoff',
        help = '高斯混合模型区分男女的阈值，不指定将自行计算',
        type = float,
    )
    parser_baseline.add_argument(
        '-m', '--males_yfrac',
        help = '男性血浆中Y染色体占比',
        type = float,
        default = MALES_YFRAC
    )
    parser_baseline.add_argument(
        '--male_samples',
        help = "男性样本",
        type = str
    )
    parser_baseline.add_argument(
        '--ttest_check',
        help = '使用t检验对过滤的bin进行检测',
        action = 'store_true',
        default = False
    )
    parser_baseline.add_argument(
        '--gc_correct',
        help = '进行GC校正',
        action = 'store_true',
        default = False
    )
    parser_baseline.add_argument(
        '--alignability_correct',
        help = '进行可比对性校正',
        action = 'store_true',
        default = False
    )
    parser_baseline.add_argument(
        '--pca_correct',
        help = '使用PCA方法进行偏差校正',
        action = 'store_true',
        default = False
    )
    parser_baseline.add_argument(
        'bamfiles',
        nargs = '+',
        metavar = 'BAM_FILE',
        type = str,
        help = 'Input BAM files'
    )

    parser_detect = subparsers.add_parser('detect', help = "NIPT检测")
    parser_detect.set_defaults(subcommand = nipt_detect)
    parser_detect.add_argument(
        '-p', '--plus',
        help = 'NIPTplus 检测，默认只是NIPT',
        default = False,
        action = 'store_true'
    )
    parser_detect.add_argument(
        '-s', '--sample_list',
        help = "样本信息表格",
        type = str,
        default = None
    )
    parser_detect.add_argument(
        'baseline',
        help = '基线文件，JSON格式',
        type = str
    )
    parser_detect.add_argument(
        'bamfile',
        help = "bam文件，接受标准输入，可指定多个",
        nargs = '*'
    )

    parser_viewer = subparsers.add_parser('viewer', help = "结果可视化")
    parser_viewer.set_defaults(subcommand = viewer)
    parser_viewer.add_argument(
        'baseline_dir',
        help = "基线构建目录",
    )
    parser_viewer.add_argument(
        'detect_dir',
        help = "检测结果输出目录"
    )
    parser_viewer.add_argument(
        'preprocess_dir',
        help = "数据预处理输出目录"
    )


    return parser.parse_args()

#-------------------------------------- BIN的切分及信息 -----------------------------------------------

def split_genome_bins(chroms, genome, binsize):
    '''将基因组切分为指定大小的BIN

    chroms: 包含的染色体
    genome：基因组版本
    binsize: BIN的大小

    return: BedTool对象
    '''
    genome_bins = (
        pybedtools.BedTool()     # 建立BedTool对象
            .window_maker(genome=genome, w=binsize, s=binsize) # 使用指定的窗口大小将基因组切分为BIN
            .filter(lambda feature: feature.chrom in chroms) # 过滤掉不需要的染色体和片段
            .saveas() # 保存为临时文件，方便传递
    )

    return genome_bins

def get_bins_attr(genome_bins, genome_fa, align_bw):
    '''给各个BIN添加序列相关的信息，GC含量、N含量和区域可比对性

    genome_bins: BedTool对象
    genome_fa: 基因组fasta文件路径
    align_bw: 可比对性BigWig文件路径

    return: DataFrame
    '''
    genome_bins = (
        genome_bins.nuc(genome_fa) # bedtools nuc获取BIN的碱基成分信息
            .each(extract_gc_Nc)  # 只提取出GC和NC含量信息
            .saveas() # 保存为临时文件
    )

    genome_bins = extract_alignability(genome_bins, align_bw)
    genome_bins.saveas()

    bins_attr = pd.read_table(genome_bins.fn, header=None)
    bins_columns = ['chrom', 'start', 'stop', 'gc', 'Nc', 'alignability']
    bins_attr.columns = bins_columns

    return bins_attr

def extract_gc_Nc(feature):
    '''根据bedtools nuc的返回提取其中的GC和NC

    feature: pybedtools 的Interval对象, 代表一个BIN

    return: 只保留了GC和NC信息的Interval对象
    '''
    gc = "{:.3f}".format(float(feature[4]))
    Nc = "{:.3f}".format(int(feature[9]) / int(feature[11]))
    new_feature = feature[0:3] + [gc, Nc]
    return pybedtools.create_interval_from_list(new_feature)

def extract_alignability(genome_bins, align_bw):
    '''根据可比对性BigWig文件，提取BIN的可比对性

    genome_bins: BedTool 对象
    align_bw: 包含可比对信息的BigWig文件

    return: BedTool对象
    '''
    bw = pyBigWig.open(align_bw)

    return genome_bins.each(
        lambda x: pybedtools.create_interval_from_list(
            [*x,
             "{:.3f}".format(
                 bw.stats(x.chrom, x.start, x.stop)[0] or 0  # 计算这个BIN的可比对性
             )]
        )
    ).saveas()

#-------------------------------------- BIN的read数及处理 ---------------------------------------------

def get_bins_counts(genome_bins, bamfiles, threads=8):
    '''所有样本在每个BIN中的read数

    genome_bins: 基因组划分的BIN，BedTool对象
    bamfiles: 样本比对BAM文件列表
    thread: 并行进程数

    return: DataFrame
    '''
    with multiprocessing.Pool(threads) as p:
        args = [ (genome_bins.fn, bamfile) for bamfile in bamfiles ]
        counts = p.starmap(extract_counts, args)

    bins_counts = pd.DataFrame(dict(counts))
    return bins_counts

def extract_counts(bedfile, bamfile):
    '''使用bedtools coverage获取各区间的read数

    bedfile: BED文件
    bamfile: BAM文件

    return：样本名（文件名）和read数组成的元组

    由于bedtools coverage无法进行read过滤，所以应该使用去除后的BAMFILE
    '''
    samplename = os.path.basename(bamfile).split('.')[0]

    genome_bins = pybedtools.BedTool(bedfile)
    bam = pybedtools.BedTool(bamfile)
    cov = genome_bins.coverage(bam, counts=True).saveas()

    counts = list(map(lambda x: int(x[-1]), cov))

    return (samplename, counts)

def bins_grouper(chroms, n = 2):
    '''结合pandas的groupby将两个bin合并成一个bin'''
    groups = []
    i = 0
    j = 0
    for chrom in chroms:
        if len(groups) != 0 and chrom != groups[-1].split('_')[0]:
            i = 0
            j = 0
        if i < n:
            groups.append(chrom+'_'+str(j))
            i += 1
        else:
            i = 1
            j += 1
            groups.append(chrom+'_'+str(j))
    return groups


#------------------------------------- BIN的统计信息 -------------------------------------------------

def bin_rc_summary(counts):
    '''描述性统计量

    bins_counts: 各read数的DataFrame

    return: 包含各统计量的DataFrame
    '''
    rc_stats = pd.DataFrame({
        'min': counts.min(axis=1),
        'q1': counts.quantile(.25, axis=1),
        'q3': counts.quantile(.75, axis=1),
        'max': counts.max(axis=1),
        'mean': counts.mean(axis=1),
        'std': counts.std(axis=1),
        'median': counts.median(axis=1),
        'mad': counts.mad(axis=1)
    })
    rc_stats['cv'] = rc_stats['std'] / rc_stats['mean']
    rc_stats['rcv'] = rc_stats['mad'] / rc_stats['median']
    rc_stats['iqr'] = rc_stats['q3'] - rc_stats['q1']

    return rc_stats

def bin_rc_stats(counts, calls):
    '''
    args = [['mean'], ['median'], ['quantile', {'q'=0.25}]]
    '''
    columns = [i[0] for i in calls]
    rc_stats = pd.DataFrame(0, index = counts.index, columns = columns)

    # 如果是lazy的执行方式可以节省时间，否则(@ _ @)
    counts.cv = lambda **x: counts.std(axis=1) / counts.mean(axis=1)
    counts.rcv = lambda **x: counts.mad(axis=1) / counts.median(axis=1)
    counts.iqr = lambda **x: counts.quantile(.75, axis=1) - counts.quantile(.25, axis=1)
    counts.range = lambda **x: counts.max(axis=1) - counts.min(axis=1)

    for call in calls:
        try:
            func = getattr(counts, call[0])
            if len(call) < 2:
                call.append({})
            if 'axis' not in call[1]:
                call[1]['axis'] = 1
            result = func(**call[1])
        except AttributeError:
            exit_with_error("方法没有定义")

        rc_stats[call[0]] = result

    return rc_stats

def outlier_mad_median(a, n, side = 'both'):
    median = a.median()
    mad = a.mad()

    if side == 'both':
        outlier = a > median + n * mad
        outlier = outlier | (a < median - n *mad)
    elif side == 'left':
        outlier = a < median - n * mad
    else:
        outlier = a > median + n * mad

    return outlier

def mean_std_zscore(ratios, is_shrink = False, include_zscore = False):
    '''
    计算BIN的均值和标准差，作为基线

    ratios：read比率数据框
    is_shtink: 是否根据zscore进行收敛
    include_zscore: 是否包含zscore的统计信息
    '''
    calls = [['mean'], ['std'], ['median'], ['mad']]
    region_stats = bin_rc_stats(ratios, calls)
    if is_shrink:
        sample_region_zscore = ratios.sub(region_stats['mean'], axis='index').div(region_stats['std'], axis='index')
        calls = [['median'], ['iqr']]
        zscore_stats = bin_rc_stats(sample_region_zscore, calls)
        tests = sample_region_zscore.sub(zscore_stats['median'], axis='index').abs().gt((3 * zscore_stats['iqr']), axis='index')

        for col in tests.columns[tests.any()]:
            sample_region_outiler = tests.loc[:, col]
            #ratios.loc[sample_region_outiler, col] = region_stats['mean'][sample_region_outiler].copy(deep=True)
            ratios.loc[sample_region_outiler, col] = region_stats.loc[sample_region_outiler, 'mean']

    calls = [['mean'], ['std'], ['median'], ['mad']]
    region_stats = bin_rc_stats(ratios, calls)

    rt = region_stats.to_dict(orient = 'list')

    if include_zscore:
        calls = [['min'], ['max'], ['mean'], ['std']]
        sample_region_zscore = ratios.sub(region_stats['mean'], axis='index').div(region_stats['std'], axis='index')
        zscore_stats = bin_rc_stats(sample_region_zscore, calls)
        rt.update(zscore_stats.to_dict(orient = 'list'))

    return rt

def extract_use_bins_ratios(bins_attr, counts, bins_filter, to_ratios=True, gcc=True, is_boys=None):
    '''根据过滤信息，过滤掉BIN。只有剩下的BIN将用于建立基线。

    bins_attr: BIN的基本信息数据框
    counts：最原始的read counts数据框
    bins_filter: 过滤BIN系列
    use_ratios: 是否转换成比率

    return: bins_attr, ratios
    '''
    use_bins = bins_filter == 'PASS'
    use_bins_attr = bins_attr.loc[use_bins, :]
    use_counts = counts.loc[use_bins, :]

    if gcc:
        if is_boys is None:
            is_boys = pd.Series(False, index=use_counts.columns)
        use_counts = use_counts.apply(bias_correct2, args=(use_bins_attr, is_boys))

    if to_ratios:
        use_ratios = use_counts / use_counts.loc[use_bins_attr['chrom'].isin(AUTOSOME), :].sum(axis=0) * SCALE_NUMBER
        #use_ratios = use_counts / use_counts.sum(axis=0) * SCALE_NUMBER
    else:
        use_ratios = use_counts

    return (use_bins_attr, use_ratios)

def calc_zscore(ratios, baseline, rubost=True):
    '''计算Z.score和R.Z.score
    '''
    if rubost:
        K = 0.6745
        zscore = ratios.sub(baseline['median'], axis='index').div(baseline['mad'], axis='index') * K
    else:
        zscore = ratios.sub(baseline['mean'], axis='index').div(baseline['std'], axis='index')
    return zscore


#------------------------------------- BIN过滤器 ------------------------------------------------------

def attr_filter(bins_attr, bins_filter = pd.Series(dtype=str), min_gc = 0.3, max_gc = 0.7, max_Nc = 0.05, alignability = 0.6):
    '''根据序列特征过滤BIN

    bins_attr: 包含序列特征的数据框
    min_gc: 最小gc含量
    max_gc: 最大gc含量
    max_Nc: 最大N含量
    alignability: 最小可比对性

    return: 包含过滤的系列
    '''
    items = ['gc', 'Nc', 'alignability']
    for item in items:
        if item not in bins_attr.columns:
            exit_with_error("bin_attr中没有{}列".format(item), 1)

    if bins_filter.empty:
        bins_filter = pd.Series('PASS', index=bins_attr.index)

    ranges = [(min_gc, max_gc), (0, max_Nc), (alignability, 1)]
    for i,r in zip(items, ranges):
        filter_out = (bins_filter == 'PASS') & ((bins_attr[i] < r[0]) | (bins_attr[i] > r[1]))
        bins_filter[filter_out] = i.title() + 'OutRange'

    return bins_filter

def rc_filter(bins_attr, bins_filter = pd.Series(dtype=str), n = 3.33, rubost=True):
    '''根据测序数据过滤BIN

    bin_attrs: 包含BIN统计量的数据框
    is_boys: 性别
    n: mead +/- n * std
    rubost: 使用中位数

    return: 包含过滤的系列
    '''
    if rubost:
        stats = [ ('median','both'), ('rcv', 'right')]
    else:
        stats = [('mean','both'), ('cv', 'right')]

    if bins_filter.empty:
        bins_filter = pd.Series('PASS', index=bins_attr.index)

    if len(bins_filter) != len(bins_attr):
        exit_with_error("bins_attr和bins_filter的长度不相等", EXIT_VALUE_ERROR)

    for chroms in [AUTOSOME, (CHRX,), (CHRY,)]:
        for ct, side in stats:
            idx = bins_attr.loc[:, 'chrom'].isin(chroms) & (bins_filter == 'PASS')
            # outlier_mad_median or outlier_iqr
            outliers = outlier_mad_median(bins_attr.loc[idx, ct], n, side)
            outliers_idx = outliers[outliers == True].index
            bins_filter[outliers_idx] = ct.title() + side.title() + 'Outlier'

    return bins_filter

#------------------------------------- 偏差校正 -------------------------------------------------------

def bias_correct(bins_attr, counts, bias='gc', factors = pd.Series(dtype=float), plot = False):
    '''偏差校正

    bins_attr: 包含信息的BIN数据框
    counts：read count数据框
    plot： 是否画出校正前后的变量与read数的分布图

    return：校正后的read数数据框和校正因子系列
    '''
    # 排除Y染色体
    idx = bins_attr['chrom'] != CHRY

    if factors.empty:
        rc = counts.loc[idx, :].median(axis=1)
        gc = bins_attr.loc[idx, bias]
        gc_rc = pd.DataFrame({'gc':gc, 'rc':rc})

        #os.environ['R_HOME'] = '/usr/lib/R'
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_gc_rc = ro.conversion.py2rpy(gc_rc)
        model = ro.r.loess('rc ~ gc', data = r_gc_rc)

        gc = gc_rc['gc']
        corr = ro.r.predict(model, ro.FloatVector(gc))

        median = gc_rc['rc'].median()
        factors = median - np.array(corr)
        factors = pd.Series(factors, index=counts.loc[idx, :].index)
        # 画图
        if plot:
            plt.figure(figsize=(16,6))
            plt.plot(gc_rc['gc'], gc_rc['rc'], 'r.', gc, gc_rc['rc'] + factors, 'g.')
            if isinstance(plot, bool):
                outfile = bias + '_correct'
            else:
                outfile = plot
            plt.savefig(outfile)

    # 校正
    #factors.index = counts.loc[idx, :].index
    correct_counts = counts.loc[idx, :].add(factors, axis = 0)

    # 加上Y染色体
    correct_counts = pd.concat([correct_counts, counts.loc[~idx, :]])

    return (correct_counts, factors)

def bias_correct2(ratio, bins_attr, is_boys, fVal=0.1, iVal=3):
    '''
    偏差校正, from Wisecondor

    ratio: 需要校正的序列
    gc: 偏差因素

    返回校正后的序列
    '''
    from statsmodels.nonparametric import smoothers_lowess as sm

    yidx = bins_attr['chrom'] == CHRY
    xidx = bins_attr['chrom'] == CHRX
    aidx = ~(xidx | yidx)
    lowessCurve = pd.Series(1, index=ratio.index)
    factors = pd.Series(0, index=ratio.index)
    for idx in (aidx, xidx):
        lowessTmp = sm.lowess(ratio[idx], bins_attr.loc[idx, 'gc'], frac=fVal, it=iVal, return_sorted = False)
        lowessCurve[idx] = lowessTmp

        # median correct
        factors[idx] = ratio[idx].median() - lowessTmp

    #corrected = ratio / lowessCurve
    corrected = ratio + factors
    return corrected

def pca_correct(ratios, factors=None, n=3):
    """使用PCA的方法进行偏差校正
    """
    #tData = StandardScaler().fit_transform(ratios.T)
    tData = ratios.T

    if factors is None:
        pca = PCA(n_components=n)
        pca.fit(tData)
        PCA(copy=True, whiten=False)
        transformed = pca.transform(tData)
        inversed = pca.inverse_transform(transformed)

        corrected = ratios / inversed.T
        factors = (pca.components_, pca.mean_)
    else:
        components = np.array(factors[0])
        mean = np.array(factors[1])

        pca = PCA(n_components=components.shape[0])
        pca.components_ = components
        pca.mean_ = mean

        transform = pca.transform(tData)

        #reconstructed = safe_sparse_dot(transform, pca.components_) + pca.mean_
        #reconstructed = reconstructed[0]
        #reconstructed = pd.Series(reconstructed, index=ratios.index)
        inversed = pca.inverse_transform(transform)

        corrected = ratios / inversed.T

        #corrected = ratios.div(reconstructed, axis=0)

    return corrected, factors

def pca_correct2(ratios, bins_attr, factors=None, is_boys=None, n=3):
    """使用PCA的方法进行偏差校正
       对性染色体进行区分
    """
    #tData = StandardScaler().fit_transform(ratios.T)
    tData = ratios.T
    yidx = bins_attr['chrom'] == CHRY
    xidx = bins_attr['chrom'] == CHRX
    aidx = ~(xidx | yidx)

    corrected = ratios.copy()
    s = pd.Series(True, index=tData.index)
    if factors is None:
        factors = []
        pca = PCA(n_components=n)
        pca.fit(tData.loc[s, aidx])
        PCA(copy=True, whiten=False)
        transformed = pca.transform(tData.loc[s, aidx])
        inversed = pca.inverse_transform(transformed)

        corrected.loc[aidx, s] = ratios.loc[aidx, s] / inversed.T
        factors.append((pca.components_, pca.mean_))
        for idx in (xidx, yidx):
            for s in (is_boys, ~is_boys):
                # 如果没有数据，跳过
                if s.sum() == 0:
                    continue
                pca = PCA(n_components=n)
                pca.fit(tData.loc[s, idx])
                PCA(copy=True, whiten=False)
                transformed = pca.transform(tData.loc[s, idx])
                inversed = pca.inverse_transform(transformed)

                corrected.loc[idx, s] = ratios.loc[idx, s] / inversed.T

                factors.append((pca.components_, pca.mean_))
    else:
        params = factors.pop(0)
        components = np.array(params[0])
        mean = np.array(params[1])

        pca = PCA(n_components=components.shape[0])
        pca.components_ = components
        pca.mean_ = mean

        transform = pca.transform(tData.loc[s, aidx])

        #reconstructed = safe_sparse_dot(transform, pca.components_) + pca.mean_
        #reconstructed = reconstructed[0]
        #reconstructed = pd.Series(reconstructed, index=ratios.index)
        inversed = pca.inverse_transform(transform)

        corrected.loc[aidx, s] = ratios.loc[aidx, s] / inversed.T

        #corrected = ratios.div(reconstructed, axis=0)
        for idx in (xidx, yidx):
            for s in (is_boys, ~is_boys):
                params = factors.pop(0)
                components = np.array(params[0])
                mean = np.array(params[1])

                # 如果没有数据， 跳过
                if s.sum() == 0:
                    continue

                pca = PCA(n_components=components.shape[0])
                pca.components_ = components
                pca.mean_ = mean

                transform = pca.transform(tData.loc[s, idx])

                inversed = pca.inverse_transform(transform)

                corrected.loc[idx, s] = ratios.loc[idx, s] / inversed.T


    return corrected, factors

#-------------------------------------- 胎儿性别 ------------------------------------------------------

def train_gender_model(yfrac, plot=True):
    '''使用高斯混合模型预测性别

    yfrac: Y ratio的array
    plot: 是否高斯混合模型图

    return: Y浓度的阈值
    如果女性样本太少，或者分布不匀，可能出错
    '''

    gmm = GaussianMixture(n_components=2, covariance_type='full', reg_covar=1e-99, max_iter=10000, tol=1e-99)
    gmm.fit(X=yfrac.reshape(-1, 1))
    gmm_x = np.linspace(0, np.max(yfrac)*1.1, 5000)
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

    # plot
    if plot:
        fig, ax = plt.subplots(figsize=(16,6))
        ax.hist(yfrac, bins=100, density=True)
        ax.plot(gmm_x, gmm_y, 'r-', label='Gaussian mixture fit')
        ax.set_xlim([0, gmm_x.max()])
        ax.legend(loc='best')
        plt.savefig('gender_gmm')

    sort_idd = np.argsort(gmm_x)
    sorted_gmm_y = gmm_y[sort_idd]

    local_min_i = argrelextrema(sorted_gmm_y, np.less)

    if local_min_i[0].size != 0:
        cut_off = gmm_x[local_min_i][0]
        return float(cut_off)
    else:
        exit_with_error("男/女性样本太少，无法训练出模型", 1)

def predict_gender(bins_attr, counts, gender_cutoff):
    '''预测胎儿性别
    '''
    ratios = counts / counts.sum() * SCALE_NUMBER
    yidx = bins_attr['chrom'] == CHRY
    yfrac = ratios.loc[yidx, :].sum()
    return(yfrac > gender_cutoff)

#-------------------------------------- 胎儿浓度 -----------------------------------------------------

def calc_samples_yfrac(ratios, is_samples, yidx=None, is_ratios=True):
    '''计算样本Y染色体占比的平均数

    ratios: BIN read ratio DataFrame
    yidx: chromsome Y index in use bin
    is_samples: is include this sample?

    return: int
    '''
    print(yidx)
    if is_ratios:
        chry_frac = ratios.loc[yidx, :].sum()
    else:
        chry_frac = ratios.loc[yidx, :].sum() / ratios.sum()
    samples_frac = chry_frac[is_samples].mean()
    return float(samples_frac)

def boys_ff_calc(yfracs, girls_yfrac, males_yfrac):
    '''计算男胎胎儿浓度
    '''
    #Based on: %chrY sample = meanY% males * FF + meanY% women with female fetusses * (1 - FF)
    #Taken from: Chiu et al, Non-invasive prenatal assessment of trisomy 21 by multiplexed maternal plasma DNA sequencing: large scale validity study, 2011
    return (yfracs - girls_yfrac) / (males_yfrac - girls_yfrac)
    #return(0.04042 * yfracs + 1.26161)

#-------------------------------------- 可视化输出 ---------------------------------------------------

def plot_bins_values(bins_values, output = 'bins_values'):
    '''画基线的点图

    bins_values: bin的区间和他的值
    '''
    chroms = bins_values['chrom'].unique()

    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(len(chroms), hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    i = 0
    for chrom in CHROMS:
        subset = (bins_values['chrom'] == chrom)
        data = bins_values['values'][subset]
        m = data.mean()
        s = data.std()
        data.index = range(len(data))
        axs[i].plot(bins_values.loc[subset, 'start'], bins_values.loc[subset, 'values'], '.g')
        axs[i].axhline(m + 2 * s, color = 'red')
        axs[i].axhline(m - 2 * s, color = 'red')
        axs[i].set(ylabel = chrom)
        i += 1

    fig.savefig(output)

def plot_baseline(baseline_dir):
    '''基线文件可视化'''
    baseline_file = os.path.join(baseline_dir, "baseline.json")
    if not os.path.exists(baseline_file):
        raise Exception("基线文件不存在")
    with open(baseline_file) as f:
        baseline = json.load(f)
    bins_filter = pd.Series(baseline['bins_filter'])
    f_pass = bins_filter == 'PASS'
    bins_baseline = extract_baseline(baseline, 'bins_baseline', bins_filter[f_pass].index)

    bins_attr_file = os.path.join(baseline_dir, 'bins_attr.pkl')
    if os.path.exists(bins_attr_file):
        bins_attr = pd.read_pickle(bins_attr_file)
        use_bins = bins_attr.loc[f_pass, :]
    else:
        # 重新生成bins_attr
        pass

    df = pd.concat([use_bins, bins_baseline], axis=1)
    # GC含量与中位数
    fig_gc = px.scatter(df, x='gc', y='median', color='chrom')

    # 中位数条形图
    chroms = df['chrom'].unique()
    hist_list = []
    for chrom in chroms:
        in_chrom = df['chrom'] == chrom
        hist_list.append(df.loc[in_chrom, 'median'])
    fig_median = ff.create_distplot(hist_list, chroms, show_hist=False)

    # bin在染色体的分布
    fig_bins = px.scatter(df, x='start', y='median', color='chrom', error_y='mad', hover_data=df.columns)

    # 染色体绘图
    chrom_len = (use_bins['stop'] - use_bins['start']).groupby(use_bins['chrom'], as_index=True).sum()
    chr_baseline = extract_baseline(baseline, 'chr_baseline', chroms)
    chr_baseline['chrom'] = chr_baseline.index
    chr_baseline['len'] = chrom_len
    fig_chroms = px.scatter(chr_baseline, x='len', y='median', error_y='mad',
                            color='chrom',
                            trendline='ols',
                            trendline_scope='overall',
                            trendline_color_override = 'black')


    return fig_gc, fig_median

#-------------------------------------- 格式化输出 ---------------------------------------------------

def extract_fastp_result(fastpjson):
    '''从fastp生成的json文件提取质控信息'''
    with open(fastpjson) as f:
        fastp = json.load(f)

    fqstats = {
        'total': fastp['summary']['before_filtering']['total_reads'],
        'q30': fastp['summary']['after_filtering']['q30_rate'],
        'adapter': fastp['adapter_cutting']['adapter_trimmed_reads'],
        'clean': fastp['summary']['after_filtering']['total_reads']
    }
    return fqstats

def extract_bamqc_result(bamqc):
    with open(bamqc) as f:
        lines = f.readlines()

    bamstats = dict(zip(lines[0].strip().split("\t"), lines[1].strip().split("\t")))
    return bamstats

def simple_output(is_boys, fracs, zscore, outfile='nipt_final_output.csv', segments=None, samplelist=None):
    '''结果输出到CSV表格
    '''
    sex = pd.Series('girl', index = is_boys.index)
    sex[is_boys] = 'boy'
    out = pd.DataFrame({'sex': sex})
    fracs.name = 'Fetal_Y'

    results = {}
    if segments is None:
        segments = pd.DataFrame(columns=['ID', 'chrom', 'loc.start', 'loc.end', 'num.mark', 'seg.mean'])
    for sample in zscore.columns:
        ff = fracs.get(sample, 0.15)
        result = []
        cnvflt = segments['ID'] == sample

        for chrom in ['chr13', 'chr18', 'chr21']:
            if zscore.loc[chrom, sample] > 3:
                cnvflt = (cnvflt & (segments['chrom'] != chrom))
                result.append(chrom.replace('chr', 'T'))

        # 性染色体异常结果判断
        if is_boys[sample] == True:
            if zscore.loc['chrX', sample] > 3 and zscore.loc['chrY', sample] > 3:
                cnvflt = (cnvflt & (segments['chrom'] != CHRX) & (segments['chrom'] != CHRY))
                result.append("XXYY")
            elif zscore.loc['chrX', sample] > 3:
                cnvflt = (cnvflt & (segments['chrom'] != CHRX))
                result.append("XXY")
            elif zscore.loc['chrY', sample] > 3:
                cnvflt = (cnvflt & (segments['chrom'] != CHRY))
                result.append("XYY")
        else:
            cnvflt = (cnvflt & (segments['chrom'] != CHRX))
            if zscore.loc['chrX', sample] > 3:
                result.append('XXX')
            elif zscore.loc['chrX', sample] < -3:
                result.append('XO')


        cnvflt = (cnvflt & (segments['chrom'] != 'chrX') & (segments['chrom'] != 'chrY'))
        for item in segments.loc[cnvflt, :].values:
            #print("{},{},{},{},{},{},{}".format(item[0], item[1], item[2], item[3], item[4], item[5], (2**(item[5]+1) - 2 * (1 - ff))/ ff))
            if item[4] < 40:
                continue
            if item[5] > 0.1 and item[6] > 1:
                result.append('Dup[{}:{}-{}]'.format(item[1], item[2], item[3]))
            if item[5] < -0.1 and item[6] < -1:
                result.append('Del[{}:{}-{}]'.format(item[1], item[2], item[3]))

        results[sample] = ';'.join(result)
    out = pd.concat([out, fracs.T, zscore.T.loc[:, ['chr13', 'chr18', 'chr21']], pd.Series(results, name="Result")], axis='columns')

    if samplelist:
        sample_info = pd.read_csv(samplelist, header=None, index_col=0)
        samples = sample_info[1] + '_' + sample_info[2] + '_' + sample_info[3].map(str)
        samples.name = 'seqinfo'
        samples = samples[samples.isin(out.index)]
        out = out.loc[samples, :]
        out.index = samples.index
        out.index.name = 'sample'
        out = pd.concat([samples, out], axis = 'columns')

    out.to_csv(outfile)

#-------------------------------------- 基线 --------------------------------------------------------

def build_baseline(args):
    '''建立NIPT检测的基线，主要步骤包括BIN的过滤及基线的中位数和绝对中位差

    args: 命令行参数

    return: None
    '''
    n_samples = len(args.bamfiles)

    if n_samples < MIN_N_SAMPLES:
        logging.warn("样本数目太少，可能得不到很好的基线模型")

    if not os.path.exists(args.ref):
        exit_with_error("参考基因组序列文件不存在", FILE_NO_EXISTS)

    if args.binsize < MIN_BINSIZE:
        exit_with_error("binsize设置得太小了", EXIT_VALUE_ERROR)

    # 得到bins_attr, counts, ratios
    logging.info("开始切分BIN...")
    genome_bins = split_genome_bins(CHROMS, args.genome, args.binsize)
    # DEBUG: 测试的时候节省时间
    bins_attr_file = os.path.join(args.outdir, 'bins_attr.pkl')
    if not os.path.exists(bins_attr_file):
        logging.info("开始获取BIN的GC，Nc，和可比对性...")
        bins_attr = get_bins_attr(genome_bins, args.ref, args.align_bw)
        bins_attr.to_pickle(bins_attr_file)
    else:
        logging.info("从保存的文件中获取BIN的GC，Nc，和可比对性...")
        bins_attr = pd.read_pickle(bins_attr_file)

    counts_file = os.path.join(args.outdir, 'counts.pkl')
    if not os.path.exists(counts_file):
        logging.info("开始获取用于建基线的每个样本的每个BIN的read数...")
        counts = get_bins_counts(genome_bins, args.bamfiles, args.threads)
        counts.to_pickle(counts_file)
    else:
        logging.info("从保存的文件中获取哟关于建基线的每个样本的每个BIN的read数...")
        counts = pd.read_pickle(counts_file)


    samplesinfo = "样本数量 total: {}".format(counts.columns.size)
    # 过滤数据量低或高的样本
    if args.filter_sample:
        filter_out = outlier_mad_median(counts.sum(), 3)
        counts = counts.loc[:, ~filter_out]
        samplesinfo += "\tfilter: {}".format(filter_out.sum())


    logging.info("counts -> ratios...")
    ratios = counts / counts.sum() * SCALE_NUMBER

    # 排除男性样本
    males_ratios = None
    males_counts = None
    if args.male_samples:
        male_samples = []
        with open(args.male_samples) as f:
            for i in f:
                male_samples.append(i.split('.')[0])

        is_males = ratios.columns.isin(male_samples)
        males_ratios = ratios.loc[:, is_males]
        males_counts = counts.loc[:, is_males]
        ratios = ratios.loc[:, ~is_males]
        counts = counts.loc[:, ~is_males]
        samplesinfo += '\tmales: {}'.format(len(male_samples))

    # 判断性别
    logging.info("判断性别和确定判断性别的阈值...")
    yidx = bins_attr['chrom'] == CHRY
    yfrac = ratios.loc[yidx, :].sum(axis=0)

    if not args.gender_cutoff:
        gender_cutoff = train_gender_model(yfrac.to_numpy())
    else:
        gender_cutoff = args.gender_cutoff

    is_boys = yfrac > gender_cutoff
    nboys = is_boys.sum()
    ngirls = (~is_boys).sum()
    samplesinfo += "\tboys: {}\tgirls: {}".format(nboys, ngirls)
    logging.info(samplesinfo)

    ####### BIN过滤步骤
    # 根据基因组序列信息的过滤
    logging.info("开始根据BIN属性过滤BIN...")
    bins_filter = attr_filter(bins_attr)

    # 根据read count过滤, 这里使用ratios
    logging.info("开始根据read counts过滤BIN...")
    rc_stats = bin_rc_summary(ratios)
    for chrom in [CHRX, CHRY]: # 性染色体独自处理
        idx = bins_attr['chrom'] == chrom
        sex_counts = ratios.loc[idx, is_boys] # 只使用男性样本
        sex_rc_stats = bin_rc_summary(sex_counts)
        rc_stats.loc[idx, :] = sex_rc_stats

    bins_attr_rc_stats = pd.concat([bins_attr, rc_stats], axis = 1)
    bins_filter = rc_filter( bins_attr_rc_stats, bins_filter )

    # t检验检查
    if nboys > 0 and ngirls > 0:
        logging.info("计算男女的差异显著性检测BIN...")
        if not (0.5 < nboys / ngirls < 2):
            logging.info("男女样本数相差过大")
        boy_ratios = ratios.loc[:, is_boys]
        girl_ratios = ratios.loc[:, ~is_boys]

        rvs = scipy.stats.ttest_ind(boy_ratios, girl_ratios, axis = 1)
        bins_attr_rc_stats['pvalue'] = rvs.pvalue

        bm = boy_ratios.mean(axis=1)
        gm = girl_ratios.mean(axis=1)
        bm[bm == 0] = np.nan
        gm[gm == 0] = np.nan

        bins_attr_rc_stats['fold'] = np.log2(bm / gm)
    if args.ttest_check:
        autosome_idx = bins_attr['chrom'].isin(AUTOSOME)
        bins_filter[ (bins_filter == 'PASS') & (autosome_idx) & (rvs.pvalue < P_VALUE_THRESHOLD) ] = "PValue"
        bins_filter[ (bins_filter == 'PASS') & (~autosome_idx) & (rvs.pvalue > P_VALUE_THRESHOLD) ] = "PValue"

    logging.info("BIN过滤完成, 剩余bin个数: " + str((bins_filter == 'PASS').sum()))

    bins_attr_rc_stats['filter'] = bins_filter

    ###### 这里使用过滤后的BIN来建立基线
    logging.info("过滤掉BIN并重新构建ratio数据框")
    pca_factors=None
    if args.gc_correct:
        logging.info("GC校正...")
        use_bins, use_ratios = extract_use_bins_ratios(bins_attr, counts, bins_filter, gcc=True)

        # 为了画图比较校正前后的分布
        tmp_bins, tmp_ratios = extract_use_bins_ratios(bins_attr, counts, bins_filter, gcc=False)
        outfile = os.path.join(args.outdir, 'gc_correct')
        plt.figure(figsize=(16,6))
        plt.plot(tmp_bins['gc'], tmp_ratios.iloc[:, 1], 'r.', use_bins['gc'], use_ratios.iloc[:, 1], 'g.')
        plt.savefig(outfile)
    else:
        use_bins, use_ratios = extract_use_bins_ratios(bins_attr, counts, bins_filter, gcc=False)
        if args.pca_correct:
            logging.info("使用PCA进行偏差校正...")
            #use_ratios, pca_factors = pca_correct(use_ratios)
            use_ratios, pca_factors = pca_correct2(use_ratios, bins_attr, is_boys=is_boys)

    yidx = use_bins['chrom'] == CHRY
    xidx = use_bins['chrom'] == CHRX
    aidx = ~(xidx | yidx)

    # 可比对性偏差校正
    if args.alignability_correct:  # 每个样本
        logging.info("可比对性校正...")
        use_ratios, alignability_factors = bias_correct(use_bins, use_ratios, 'alignability', plot = os.path.join(args.outdir, 'alignability_correct'))
        logging.debug("可比对性校正完成: \n" + str(alignability_factors))
    else:
        alignability_factors = pd.Series(0, index = use_bins.loc[~yidx, :].index)


    # 计算女胎孕妇Y染色体占比，用于估计胎儿浓度
    girls_yfrac = calc_samples_yfrac(counts, ~is_boys, bins_attr.chrom == CHRY, False)

    # 计算成年男性Y染色体占比，用于估计胎儿浓度 =======================================================================> 应跟胎儿样本的校正方式一致, 待完成
    males_yfrac = MALES_YFRAC
    if args.males_yfrac:
        males_yfrac = args.males_yfrac
    if args.male_samples:
        #_, males_ratios = extract_use_bins_ratios(bins_attr, males_counts, bins_filter, gcc=args.gc_correct)
        is_males = pd.Series(True, index=males_ratios.columns)
        males_yfrac = calc_samples_yfrac(males_counts, is_males, bins_attr.chrom == CHRY, False)
    logging.info("胎儿浓度参数: female_yfrac: {}\tmale_yfrac: {}".format(girls_yfrac, males_yfrac))

    # 基线
    # 男女性染色体分开基线
    logging.info("建立BIN基线...")
    use_bins_baseline = mean_std_zscore(use_ratios)
    boy_bins_baseline = mean_std_zscore(use_ratios.loc[:, is_boys])
    girl_bins_baseline = mean_std_zscore(use_ratios.loc[:, ~is_boys])

    logging.info("建立染色体基线...")
    chr_ratios = use_ratios.groupby(use_bins['chrom'], as_index=False).sum()
    chr_baseline = mean_std_zscore(chr_ratios)
    boy_chr_baseline = mean_std_zscore(chr_ratios.loc[:, is_boys])
    girl_chr_baseline = mean_std_zscore(chr_ratios.loc[:, ~is_boys])

    # 画下基线图
    plot_bins_values(pd.DataFrame({'chrom': use_bins['chrom'], 'start': use_bins['start'], 'values': use_bins_baseline['median']}),
            output = os.path.join(args.outdir, 'bins_value_plot'))

    # 保存表格
    tmp_df = pd.concat([bins_attr_rc_stats, pd.DataFrame(use_bins_baseline, index=use_bins.index)], axis=1)
    #tmp_df.loc[xidx, use_bins_baseline.columns] = girl_chr_baseline.loc[xidx,:]
    #tmp_df.loc[yidx, use_bins_baseline.columns] = boy_chr_baseline.loc[yidx,:]
    tmp_df.to_csv(os.path.join(args.outdir, 'baseline.csv'))

    # 保存基线为json格式
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='list')
            return super(NpEncoder, self).default(obj)

    baseline = {
        'n_samples': n_samples,
        'n_boys': nboys,
        'n_girls': ngirls,
        'genome': args.genome,
        'reference': args.ref,
        'binsize': args.binsize,
        'gcc': args.gc_correct,
        'pca_factors': pca_factors,
        'alignability_factors': alignability_factors,
        'gender_cutoff': gender_cutoff,
        'girls_yfrac': girls_yfrac,
        'males_yfrac': males_yfrac,
        'bins_attr': bins_attr_rc_stats,
        'bins_baseline': use_bins_baseline,
        'boy_bins_baseline': boy_bins_baseline,
        'girl_bins_baseline': girl_bins_baseline,
        'chr_baseline': chr_baseline,
        'boy_chr_baseline': boy_chr_baseline,
        'girl_chr_baseline': girl_chr_baseline
    }

    baseline_file = os.path.join(args.outdir, 'baseline.json')
    with open(baseline_file, 'w') as f:
        json.dump(baseline, f, cls=NpEncoder)

    logging.info("完成, 基线文件路径: {}".format(baseline_file))

def extract_baseline(baseline, name, use_bins, sex=None):
    '''基线从字典中提取到数据框
    '''
    baseline_df =  pd.DataFrame({
            'mean': baseline[name]['mean'],
            'std': baseline[name]['std'],
            'median': baseline[name]['median'],
            'mad': baseline[name]['mad']
            }, index = use_bins.index)

    if sex != 'boy' and sex != 'girl':
        return baseline_df

    name = sex + '_' + name
    sex_baseline_df = pd.DataFrame({
        'mean': baseline[name]['mean'],
        'std': baseline[name]['std'],
        'median': baseline[name]['median'],
        'mad': baseline[name]['mad']
        }, index = use_bins.index)

    yidx = use_bins['chrom'] == CHRY
    xidx = use_bins['chrom'] == CHRX
    baseline_df.loc[(yidx | xidx), :] = sex_baseline_df.loc[(yidx | xidx), :]

    return baseline_df

#-------------------------------------- 检测 ---------------------------------------------------------

def do_segmentation(bins_attr, log_ratios, weight, method='cbs', plot = True):
    '''segmentation, 使用CBS算法或其它算法，参考cnvlib的代码
    '''
    weight = weight.fillna(weight.max())
    weight = weight + 0.0001
    if method == 'cbs':
        if log_ratios.empty:
            return pd.DataFrame(columns=['ID', 'chrom', 'loc.start', 'loc.end', 'num.mark', 'seg.mean'])
        os.environ['R_HOME'] = '/usr/lib/R'  # 必须使用系统的R
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        ro.r('sink("/dev/null")')
        ro.r['options'](warn=-1)
        dnacopy = importr("DNAcopy")
        #rbase = importr("base")

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_bins_attr = ro.conversion.py2rpy(bins_attr)
            r_log_ratios = ro.conversion.py2rpy(log_ratios)
            r_weight = ro.conversion.py2rpy(weight)

        #chrom = rordered(r_bins_attr[0], levels=CHROMS)
        #maploc = r_bins_attr[1]
        cna = dnacopy.CNA(r_log_ratios, chrom = r_bins_attr[0], maploc = r_bins_attr[1], sampleid=log_ratios.columns.tolist(), data_type="logratio")
        cna = dnacopy.smooth_CNA(cna)
        fit = dnacopy.segment(cna, weight=r_weight, alpha = 0.01)

        with localconverter(ro.default_converter + pandas2ri.converter):
            segment_output = ro.conversion.rpy2py(fit[1])

    elif method == 'hmm':
        pass
    else:
        pass

    return segment_output

def nipt_detect(args):
    '''检测非整倍体'''
    if not os.path.exists(args.baseline):
        exit_with_error("基线文件不存在", FILE_NO_EXISTS)
    with open(args.baseline) as f:
        baseline = json.load(f)

    genome_bins = split_genome_bins(CHROMS, baseline['genome'], baseline['binsize'])
    if not args.bamfile: # stdin
        pass
    bamfiles = args.bamfile

    # 计算每个BIN的read count
    if not os.path.exists('counts.pkl'):
        logging.info("开始获取检测样本的每个BIN的read数...")
        counts = get_bins_counts(genome_bins, bamfiles, args.threads)
        counts.to_pickle('counts.pkl')
    else:
        logging.info("从保存的文件中获取检测样本的每个BIN的read数...")
        counts = pd.read_pickle('counts.pkl')

    bins_attr = pd.DataFrame(baseline['bins_attr'])
    bins_filter = bins_attr['filter']

    # 判断性别
    logging.info("判断胎儿性别...")
    is_boys = predict_gender(bins_attr, counts, baseline['gender_cutoff'])

    # 获取基线数据
    logging.info("偏差校正...")
    gcc = baseline['gcc']
    use_bins, use_ratios = extract_use_bins_ratios(bins_attr, counts, bins_filter, gcc = gcc)

    pca_factors = baseline['pca_factors']
    if pca_factors is not None:
        #use_ratios, _ = pca_correct(use_ratios, pca_factors)
        use_ratios, _ = pca_correct2(use_ratios, bins_attr, factors=pca_factors, is_boys=is_boys)

    yidx = use_bins['chrom'] == CHRY
    xidx = use_bins['chrom'] == CHRX
    aidx = ~(xidx | yidx)

    alignability_factors = pd.Series(baseline['alignability_factors'])
    alignability_factors.index = alignability_factors.index.astype('int64')
    use_ratios, _ = bias_correct(use_bins, use_ratios, 'alignability', alignability_factors)
    if args.plus: # NIPT plus
        logging.info("NIPT plus...")
        #bins_baseline = extract_baseline(baseline, 'bins_baseline', use_bins)
        boy_bins_baseline = extract_baseline(baseline, 'bins_baseline', use_bins, 'boy')
        girl_bins_baseline = extract_baseline(baseline, 'bins_baseline', use_bins, 'girl')

        #log_ratios = np.log2(use_ratios.loc[aidx, :].div(bins_baseline['mean'], axis='index'))
        boy_log_ratios = np.log2(use_ratios.loc[:, is_boys].div(boy_bins_baseline['mean'], axis='index'))
        girl_log_ratios = np.log2(use_ratios.loc[:, ~is_boys].div(girl_bins_baseline['mean'], axis='index'))
        log_ratios = pd.concat([boy_log_ratios, girl_log_ratios], axis=1)

        #cns = use_ratios.div(bins_baseline['mean'], axis='index') * 2
        boy_segments = do_segmentation(use_bins, boy_log_ratios, boy_bins_baseline['std'])
        girl_segments = do_segmentation(use_bins, girl_log_ratios, girl_bins_baseline['std'])
        segments = pd.concat([boy_segments, girl_segments])
        #segments.to_csv("segments.csv")

        #bins_zscore = calc_zscore(use_ratios.loc[aidx, :], bins_baseline.loc[aidx, :])
        boy_bins_zscore = calc_zscore(use_ratios.loc[:, is_boys], boy_bins_baseline)
        girl_bins_zscore = calc_zscore(use_ratios.loc[:, ~is_boys], girl_bins_baseline)
        bins_zscore = pd.concat([boy_bins_zscore, girl_bins_zscore], axis=1)
        bins_zscore.to_csv('bins_zscore.csv')

        region_zscores = []
        for item in segments.values:
            sample = item[0]
            chrom = item[1]
            start = item[2]
            end = item[3]

            s = ((use_bins['chrom'] == chrom) & (use_bins['start'] >= start) & (use_bins['start'] <= end))

            if is_boys[sample] == True:
                region_zscore = (use_ratios.loc[s, sample] -  boy_bins_baseline.loc[s, 'mean']) / boy_bins_baseline.loc[s, 'std']
            else:
                region_zscore = (use_ratios.loc[s, sample] -  girl_bins_baseline.loc[s, 'mean']) / girl_bins_baseline.loc[s, 'std']
            #print(region_zscore)
            region_zscores.append(region_zscore.mean())
        segments.loc[:, 'zscore'] = region_zscores
        segments.to_csv("segments.csv")


    logging.info("NIPT...")
    chr_ratios = use_ratios.groupby(use_bins['chrom'], as_index=True).sum()
    chr_bins = pd.DataFrame({'chrom': chr_ratios.index}, index=chr_ratios.index)
    boy_chr_baseline = extract_baseline(baseline, 'chr_baseline', chr_bins, 'boy')
    girl_chr_baseline = extract_baseline(baseline, 'chr_baseline', chr_bins, 'girl')

    #chr_zscore = calc_zscore(chr_ratios, chr_baseline)
    boy_chr_zscore = calc_zscore(chr_ratios.loc[:, is_boys], boy_chr_baseline)
    girl_chr_zscore = calc_zscore(chr_ratios.loc[:, ~is_boys], girl_chr_baseline)
    chr_zscore = pd.concat([boy_chr_zscore, girl_chr_zscore], axis=1)
    chr_zscore.to_csv('chroms_zscore.csv')

    # 计算胎儿浓度, 只对男胎
    logging.info("计算男胎胎儿浓度...")
    yfracs = counts.loc[bins_attr.chrom == CHRY, is_boys].sum() / counts.loc[:, is_boys].sum()
    #ffy = boys_ff_calc(chr_ratios.loc[CHRY, is_boys], baseline['girls_yfrac'], baseline['males_yfrac'])
    ffy = boys_ff_calc(yfracs, baseline['girls_yfrac'], baseline['males_yfrac'])

    if args.plus:
        simple_output(is_boys, ffy, chr_zscore, segments=segments, samplelist=args.sample_list)
    else:
        simple_output(is_boys, ffy, chr_zscore, samplelist=args.sample_list)

    logging.info("完成")

#-------------------------------------- 预处理 -------------------------------------------------------

Sinfo = namedtuple('Sinfo', 'name barcode pipeline slide lane type')
def read_samplesheet(samplesheet, pipeline='NIPT'):
    '''将下机数据样本表转成fastq列表'''
    if samplesheet[-4:] == 'xlsx':
        sample_infos = pd.read_excel(samplesheet)
    else:
        sample_infos = pd.read_csv(samplesheet)

    nipt = sample_infos['Pipeline'] == pipeline
    sample_infos = sample_infos.loc[nipt, :]

    sample_infos = sample_infos.T.to_dict('list')
    sample_infos = [ Sinfo(*sample[:6]) for sample in sample_infos.values() ]

    return sample_infos

def fastq2bam(sample):
    '''fastq to bam'''
    if isinstance(sample, Sinfo):
        samplename = '_'.join([sample.slide, sample.lane, str(sample.barcode)])
        fastqfile = os.path.join(SEQDIR, sample.slide, sample.lane, samplename+'.fq.gz')
    else: # fq
        samplename = os.path.basename(sample).split('.')[0]
        fastqfile = sample
    outfastq = samplename +  '.fq.gz'
    outjson = samplename+'.json'
    outhtml = samplename+'.html'
    if not os.path.exists(fastqfile):
        logging.warning("fastq no found: {}".format(fastqfile))
    cmdlist = ['fastp', '-i', fastqfile, '-o', outfastq, '-a', ADAPTER, '-q', str(QUAL), '--json', outjson, '--html', outhtml]

    if os.path.exists(outfastq) and os.path.exists(outjson):
        logging.warning("fastp: 使用之前的结果")
        fqstats = extract_fastp_result(outjson)
    else:
        rt = subprocess.run(cmdlist)
        if rt.returncode != 0:
            logging.error('executing error: {}'.format(' '.join(cmdlist)))
            return
        else:
            fqstats = extract_fastp_result(outjson)

    outbam = samplename + '.bam'
    outsmy = samplename + '.tsv'
    if os.path.exists(outbam) and os.path.exists(outsmy):
        logging.warning("bwa aln: 使用之前的结果")
        bamstats = extract_bamqc_result(outsmy)
    else:
        # https://www.titanwolf.org/Network/q/177c74a5-1733-4d8e-843b-217b57da96fe/y
        cmdline = "bwa aln -t 4 {} {} | bwa samse -n 10 -f /dev/stdout {} - {} | samtools view -@ 4 -bS | samtools sort -@ 4 | samtools markdup -@ 4 - /dev/stdout ".format(REF, outfastq, REF, outfastq)
        cmdline += "| python /data/home/chenshulin/project/NIPT/MGI_DATA/danipt/danipt/bam_filter.py - {} {}".format(outbam, samplename)
        rt = subprocess.run(cmdline, stdout=subprocess.PIPE, shell=True)
        if rt.returncode != 0:
            logging.error('executing error: {}'.format(cmdline))
            return
        else:
            bamstats = extract_bamqc_result(outsmy)

    # samtools index
    subprocess.run(['samtools', 'index', outbam])

    return (samplename, {**fqstats, **bamstats})

def is_fastq(path):
    if path[-3:] == '.fq':
        return True
    if path[-6:] == '.fastq':
        return True
    if path[-6:] == '.fq.gz':
        return True
    if path[-9:] == '.fastq.gz':
        return True
    return False

def preprocess(args):
    '''预处理: 从fastq到bam'''
    if len(args.samplesheet_or_fqfiles) == 1 and (not is_fastq(args.samplesheet_or_fqfiles[0])):
        if not os.path.exists(args.samplesheet_or_fqfiles[0]):
            raise ValueError("文件不存在: {}".format(args.samplesheet_or_fqfiles[0]))
        else:
            samples = read_samplesheet(args.samplesheet_or_fqfiles[0])
    else:
        samples = args.samplesheet_or_fqfiles


    if not os.path.exists(args.seqdir):
        raise ValueError("文件不存在: {}".format(seqdir))
    global QUAL, MAPQ, ADAPTER, REF, SEQDIR
    if args.qual != QUAL:
        QUAL = args.qual
    if args.mapq != MAPQ:
        MAPQ = args.mapq
    if args.adapter != ADAPTER:
        ADAPTER = args.adapter
    if args.ref != REF:
        REF = args.ref
    if args.seqdir != SEQDIR:
        SEQDIR = args.seqdir
    #for i in samples:
    #    print(i)
    with multiprocessing.Pool(args.threads) as p:
        result = p.map(fastq2bam, samples)
        result = dict(result)

    result = pd.DataFrame(result.values(), index=result.keys())
    result.to_csv("data_qc.csv")
    #print(result)

#------------------------------------- 报告 ----------------------------------------------------------

def viewer(args):
    '''可视化基线和检测结果'''
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, dcc, html, dash_table

    app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "backgroud-color": "#f8f9fa",
    }

    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "pading": "2rem 1rem",
    }

    sidebar = html.Div(
        [
            html.H2("DANIPT", className = "display-4"),
            html.Hr(),
            html.P(
                "达安NIPT检测系统"
            ),
            dbc.Nav(
                [
                    dbc.NavLink("概览", href="/", active="exact"),
                    dbc.NavLink("样本数据", href="/preprocess", active="exact"),
                    dbc.NavLink("基线信息", href="/baseline", active="exact"),
                    dbc.NavLink("检测结果", href="/detect", active="exact"),
                ],
                vertical = True,
                pills = True
            ),
        ],
        style = SIDEBAR_STYLE,
    )

    content = html.Div(id= "page-content", style = CONTENT_STYLE)

    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return html.P("This is the content of the home page!")
        elif pathname == "/preprocess":
            df = pd.read_csv("/data/ProjectRecord/NIPT/Analysis/Slide/Process/V350011009/1.preprocess/data_qc.csv")
            return dash_table.DataTable(
                id = "table",
                columns = [{"name":i, "id": i} for i in df.columns],
                data = df.to_dict('records'),
                page_size = 25,
                editable = True,
                filter_action = "native",
                sort_action = "native",
                sort_mode = "multi",

            )
        elif pathname == "/baseline":
            baseline_file = "/data/ProjectRecord/NIPT/Pipeline/Database/baseline.json"
            with open(baseline_file) as f:
                baseline = json.load(f)

            tbl_header1 = [
                html.Thead(html.Tr([html.Th("基线基本信息")]))
            ]
            baseinfo = [
                html.Tbody(
                    [
                        html.Tr([html.Td("样本总数"), html.Td(baseline['n_samples'])]),
                        html.Tr([html.Td("男胎样本"), html.Td(baseline['males'])]),
                        html.Tr([html.Td("性别阈值"), html.Td(baseline['gender_cutoff'])]),
                        html.Tr([html.Td("女胎Y浓度"), html.Td(baseline['girls_yfrac'])]),
                        html.Tr([html.Td("男性Y浓度"), html.Td(baseline['males_yfrac'])]),
                        html.Tr([html.Td("参考基因组"), html.Td(baseline['genome'])]),
                        html.Tr([html.Td("BIN大小"), html.Td(baseline['binsize'])]),
                    ]
                )
            ]
            tbl_header2 = [
                html.Thead(html.Tr([html.Th("BIN过滤信息")]))
            ]
            bins_filter = {}
            for i in baseline['bins_filter']:
                if i in bins_filter:
                    bins_filter[i] += 1
                else:
                    bins_filter[i] = 1

            filterinfo = [
                html.Tbody(
                    [
                        html.Tr([html.Td(i), html.Td(bins_filter[i])])  for i in bins_filter
                    ]
                )
            ]

            return dbc.Table(tbl_header1+baseinfo+tbl_header2+filterinfo, bordered=True)
        elif pathname == "/detect":
            return html.P("检测结果详细信息")

        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className = "text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )

    app.run_server(port=2814, host="10.10.9.16", debug=True)

#------------------------------------- 主函数及测试 --------------------------------------------------

def main():
    "Orchestrate the execution of the program"
    args = parse_args()
    init_logging(args.log)
    args.subcommand(args)

# If this script is run from the command line then call the main function.
if __name__ == '__main__':
    main()
