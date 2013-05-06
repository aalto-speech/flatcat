import collections
import gzip
import sys
from optparse import OptionParser

def load_seg(filename):
    segs = collections.defaultdict(lambda: 'N/A')
    if filename[-3:] == '.gz':
        fobj = gzip.open(filename, 'r')
    else:
        fobj = open(filename, 'r')
    for line in fobj:
        if line[0] == '#':
            continue
        line = line.rstrip()
        w, a = line.split('\t')
        segs[w] = a
    fobj.close()
    return segs

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Usage: python remove_na.py larger smaller [-i] > filtered')

    larger = load_seg(sys.argv[1])
    smaller = load_seg(sys.argv[2])

    for word in larger:
        if word not in smaller:
            continue
        print('{}\t{}'.format(word, larger[word]))

    # with -i flag, also injects unsegmented analyses for words
    # missing from "larger" (which is a misnomer in this case)
    if len(sys.argv) >= 4 and sys.argv[3] == '-i':
        for word in smaller:
            if word in larger:
                continue
            print('{}\t{}'.format(word, word))

