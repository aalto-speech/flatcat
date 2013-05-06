#!/bin/bash

if [[ $# < 1 ]]
then
	echo "Usage: $0 dataset [-n]"
	exit
fi

# -n for dryrun
if [[ $2 == "-n" ]]
then
	DRY="echo"
else
	DRY=""
fi

# Which python interpreter to use (e.g. python2, python3, pypy)
#PYTHON="python"
PYTHON="pypy"
CATMAP="../scripts/morfessor-catmap"
DATASET=$1
#if [[ $DATASET == "toy" ]]
#then
#	PPL_THRESH=10
#else
#	PPL_THRESH=
#fi
PPL_THRESH=10
EPOCH_COST="0.0025"
ITER_COST="0.005"

TESTDATA="data/morphochal07_fin_dev.goldstd.words"
GOLDSTD="data/morphochal07_fin_dev.goldstd.segmentation"
FORCESPLIT=":-"		# hyphen can't be first char
STATS="--statsfile ${DATASET}.stats.pickled"
#EXPERIMENTAL="-d log --max-iterations 10 -D data/morphochal07_fin_train.goldstd.segmentation"
EXPERIMENTAL="-d log --max-iterations 10 -A data/morphochal07_fin_train.goldstd.segmentation -w 0.1"
#EXPERIMENTAL=""

# Baseline segmentation
#$DRY $PYTHON ../morfessor.py --traindata-list -e latin-1 -t data/${DATASET}.wordcounts.gz -S data/${DATASET}.baseline.gz -f "${FORCESPLIT}"
# Train model
$DRY $PYTHON $CATMAP -e latin-1 -B data/${DATASET}.baseline.gz -p $PPL_THRESH --min-iteration-cost-gain $ITER_COST --min-epoch-cost-gain $EPOCH_COST -s ${DATASET}.pickled --logfile ${DATASET}.log -f "${FORCESPLIT}" $STATS $EXPERIMENTAL
# Segment test data
$DRY $PYTHON $CATMAP -l ${DATASET}.pickled -e latin-1 -T ${TESTDATA} -o ${DATASET}_catmap2.dev_result --output-format "{compound}\t{analysis}\n" --remove-nonmorphemes
# Calculate boundary precision recall
# (This is run using regular python regardless of choice above, due to numpy dependency)
if [[ $2 == "-n" ]]
then
	$DRY python bin/bpr_v1.11.py -g ${GOLDSTD} -p ${DATASET}_catmap2.dev_result ">" ${DATASET}_catmap2.bpr
else
	echo "Saving boundary precision recall into ${DATASET}_catmap2.bpr"
	python bin/bpr_v1.11.py -g ${GOLDSTD} -p ${DATASET}_catmap2.dev_result > ${DATASET}_catmap2.bpr
fi

# Make this filtering part of the process?
# python bin/remove_na.py morphochal07_fin_n100_catmap2.dev_result morphochal07_fin_n100_oldcatmap.dev_result > morphochal07_fin_n100_catmap2.filtered.dev_result
