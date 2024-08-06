import sys
import pysam
import tempfile


def align_unique(bamin, bamout, mapq = 30):
    '''
    筛选唯一比对
    '''
    summary = {
        'mapped': 0,
        'unmap': 0,
        'duplicate': 0,
        'mapq': 0,
        'used': 0,
    }
    with pysam.AlignmentFile(bamin, 'rb') as samin, pysam.AlignmentFile(bamout, 'wb', template = samin) as samout:
        for read in samin:
            if read.is_unmapped:
                summary['unmap'] += 1
                continue
            summary['mapped'] += 1
            if read.is_duplicate:
                summary['duplicate'] += 1
                continue

            if read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < mapq:
                summary['mapq'] +=1
            #    if read.has_tag('XA'):
            #        inchrom = True
            #
            #        ref_chr = read.reference_name
            #
            #        alt_hits = read.get_tag('XA')
            #        alt_hits = alt_hits.split(';')
            #        chroms = [ hit.split(',')[0] for hit in alt_hits[:-1] ]
            #        for chrom in chroms:
            #            if chrom != ref_chr:
            #                inchrom = False
            #
            #        if inchrom:
            #            summary['clean'] += 1
            #            samout.write(read)
            else:
                summary['used'] += 1
                samout.write(read)

    return(summary)



def test():
    bamin = "/data/home/chenshulin/project/NIPT/MGI_DATA/2.align/S200019106_L01_90.bam"
    bamout = 'test.bam'

    align_unique(bamin, bamout)

if __name__ == '__main__':
    outfile = sys.stderr
    if len(sys.argv) == 2:
        bamin = sys.argv[1]
        bamout = '-'
    elif len(sys.argv) == 3:
        bamin, bamout = sys.argv[1:]
    elif len(sys.argv) == 4:
        bamin, bamout = sys.argv[1:3]
        outfile = open(sys.argv[3]+'.tsv', 'w')
    else:
        bamin, bamout = ('-', '-')

    summary = align_unique(bamin, bamout)
    fields = ['mapped', 'unmap', 'duplicate', 'mapq', 'used']
    outfile.write("\t".join(fields))
    outfile.write("\n")
    outfile.write("\t".join([ str(summary[i]) for i in fields ]))
    outfile.write("\n")
    outfile.close()
