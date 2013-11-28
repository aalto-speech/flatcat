import collections
import gzip
import sys
from optparse import OptionParser

def read_data(name):
    data = dict()
    with open('{}_overseg'.format(name), 'r') as f:
        data['over'] = set([line.rstrip() for line in f])
    with open('{}_underseg'.format(name), 'r') as f:
        data['under'] = set([line.rstrip() for line in f])
    with open('{}_not_underseg'.format(name), 'r') as f:
        data['not_under'] = set([line.rstrip() for line in f])
    with open('{}_not_overseg'.format(name), 'r') as f:
        data['not_over'] = set([line.rstrip() for line in f])
    make_disjoint(data)
    return data

def make_disjoint(data):
    data['both'] = data['over'].intersection(data['under'])
    data['perfect'] = data['not_over'].intersection(data['not_under'])

    data['over'].difference_update(data['both'])
    data['under'].difference_update(data['both'])
    data['not_over'].difference_update(data['perfect'])
    data['not_under'].difference_update(data['perfect'])

def lengths(data):
    print([(x, len(data[x])) for x in data])

def compare(mine, theirs):
    compared = dict()
    all_words = set().union(*mine.values())
    all_words.update(*theirs.values())
    compared['better'] = mine['perfect'].difference(theirs['perfect'])
    compared['worse_under'] = mine['under'].difference(theirs['under'],
                                                       theirs['both'])
    compared['worse_over'] = mine['over'].difference(theirs['over'],
                                                     theirs['both'])
    compared['worse_both'] = mine['both'].intersection(theirs['perfect'])
    compared['common_good'] = mine['perfect'].intersection(theirs['perfect'])
    compared['common_bad'] = all_words.difference(mine['perfect'],
                                                  theirs['perfect'])
    return compared 

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
        if len(line) == 0:
            continue
        if '\t' in line:
            w, a = line.split('\t')
        else:
            a = line
            w = a.split(', ')[0].translate(None, ' ')
        segs[w] = a
    fobj.close()
    return segs

def write_files(compared,
                mine_name, segs_mine,
                theirs_name, segs_theirs,
                segs_gold, field_len=10):
    field = '{:' + str(field_len) + 's}'
    pattern = field * 4 + '\n'
    for result_type in compared:
        with open('{}_{}_compared_{}'.format(
                mine_name, theirs_name, result_type), 'w') as f:
            f.write('# Word, My, Theirs, Gold\n')
            for word in compared[result_type]:
                f.write(pattern.format(word, segs_mine[word],
                                             segs_theirs[word],
                                             segs_gold[word]))

def main(argv):
    usage = """Usage: %prog -g goldFile -p predFile -T theirsFile -m name -t name"""

    parser = OptionParser(usage=usage)
    parser.add_option("-g", "--goldFile", dest="goldFile",
                      default = None,
                      help="gold standard segmentation file")
    parser.add_option("-p", "--predFile", dest="predFile",
                      default = None,
                      help="my predicted segmentation file")
    parser.add_option("-T", "--theirsFile", dest="theirsFile",
                      default = None,
                      help="their predicted segmentation file")
    parser.add_option("-m", "--mine-name", dest="mine_name",
                      default = None,
                      help="name of data set, for my results")
    parser.add_option("-t", "--theirs-name", dest="theirs_name",
                      default = None,
                      help="name of data set, for baseline results")
    (options, args) = parser.parse_args(argv[1:])

    if (options.goldFile == None or options.predFile == None or
        options.mine_name == None or options.theirs_name == None):
        parser.print_help()
        sys.exit()
 
    mine = read_data(options.mine_name)
    theirs = read_data(options.theirs_name)

    segs_mine = load_seg(options.predFile)
    segs_theirs = load_seg(options.theirsFile)
    segs_gold = load_seg(options.goldFile)
    field_len = max([len(x) for x in segs_gold.keys()]) + 4

    compared = compare(mine, theirs)

    write_files(compared,
                options.mine_name, segs_mine,
                options.theirs_name, segs_theirs,
                segs_gold, field_len)

if __name__ == "__main__":
    main(sys.argv)
