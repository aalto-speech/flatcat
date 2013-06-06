#!/bin/bash

if [[ $# < 1 ]]
then
	echo "Usage: $0 run-title [-n]"
	exit
fi

# -n for dryrun
if [[ $2 == "-n" ]]
then
	DRY="echo"
else
	DRY=""
fi

function check_return {
	if (($?))
	then
		echo "Last command failed, aborting"
		exit 1
	fi
}

# Which python interpreter to use (e.g. python2, python3, pypy)
#PYTHON="python"
PYTHON="pypy"

# Location of morfessor scripts
CATMAP="../scripts/morfessor-catmap"
BASELINE="../scripts/morfessor-train"
BASELINE_TEST="../scripts/morfessor-segment"

# Location of boundary precision recall script
BPR="bin/bpr_v1.11.py"

# Location of the original data files
DATADIR="data"

# FIXME FIXME
#EXPERIMENTAL="-d log --max-iterations 10 -D data/morphochal10_fin_train.goldstd.segmentation"
#EXPERIMENTAL="-d log --max-iterations 10 -A data/morphochal10_fin_train.goldstd.segmentation -w 0.1 --annotation-supermorph-penalty 0"

RUN_TITLE=$1
DATE=`date +%Y%m%d_%H%M%S`

# Dataset (size and source). Subsampling must be performed in advance.
if [[ -z ${DATASET} ]]
then
	DATASET="toy"
	echo "WARNING: did not select dataset, using toy as fallback"
fi

ROOT_DIR="results_${DATASET}"

# Output directory for the results of this run
if [[ -z ${OUTDIR} ]]
then
	OUTDIR="${DATE}_${RUN_TITLE}"
fi
# Directory containing baseline segmentation to use (created if necessary)
if [[ -z ${BASELINEDIR} ]]
then
	BASELINEDIR="baseline_default"
fi

if [[ -z ${DRY} ]]
then
	mkdir -p "${ROOT_DIR}/${OUTDIR}"
	mkdir -p "${ROOT_DIR}/${BASELINEDIR}"
fi

TRAINDATA="${DATADIR}/${DATASET}.wordcounts.gz"
TESTDATA="${DATADIR}/morphochal10_fin_dev.goldstd.words"
GOLDSTD="${DATADIR}/morphochal10_fin_dev.goldstd.segmentation"

### Commented out variables default to empty, but should be set from job.sh

# Parameters given to all training and testing commands
#COMMON_PARAMS="-e latin-1"

# Parameters only given to baseline commands
BASELINE_OUTPUT="${ROOT_DIR}/${BASELINEDIR}/baseline.gz"
#BASELINE_TRAIN_PARAMS="--traindata-list -f ${FORCESPLIT}"
#BASELINE_TEST_PARAMS=""

# Parameters only given to catmap commands
CATMAP_MODEL="${ROOT_DIR}/${OUTDIR}/model.pickled"
#CATMAP_TRAIN_PARAMS="-p ${PPL_THRESH} --min-iteration-cost-gain ${ITER_COST} --min-epoch-cost-gain ${EPOCH_COST} -f ${FORCESPLIT}"
#CATMAP_TEST_PARAMS="--remove-nonmorphemes"

# Testing parameters
COMMON_TEST_PARAMS='--output-format {compound}\t{analysis}\n'
BASELINE_TEST_OUTPUT="${ROOT_DIR}/${BASELINEDIR}/baseline.dev_result"
BASELINE_BPR_OUTPUT="${ROOT_DIR}/${BASELINEDIR}/baseline.bpr"
CATMAP_TEST_OUTPUT="${ROOT_DIR}/${OUTDIR}/${RUN_TITLE}.catmap2.dev_result"
CATMAP_BPR_OUTPUT="${ROOT_DIR}/${OUTDIR}/${RUN_TITLE}.catmap2.bpr"

# Set to empty to disable collection of diagnostics
STATS="--statsfile ${ROOT_DIR}/${OUTDIR}/stats.pickled"
LOG="--logfile ${ROOT_DIR}/${OUTDIR}/catmap2.log"

############ Baseline
if [[ ! -e ${BASELINE_OUTPUT} ]]
then
	### Training
	${DRY} ${PYTHON} ${BASELINE} ${COMMON_PARAMS} -S ${BASELINE_OUTPUT} ${BASELINE_TRAIN_PARAMS} ${TRAINDATA}
	check_return

	### Testing
	# Segment test data using baseline
	${DRY} ${PYTHON} ${BASELINE_TEST} ${COMMON_PARAMS} ${COMMON_TEST_PARAMS} ${BASELINE_TEST_PARAMS} -L ${BASELINE_OUTPUT} -o ${BASELINE_TEST_OUTPUT} ${TESTDATA}
		check_return

	# Calculate boundary precision recall
	# (This is run using regular python regardless of choice above, due to numpy dependency)
	BPR_COMMAND="${DRY} python ${BPR} -g ${GOLDSTD} -p ${BASELINE_TEST_OUTPUT}"
	if [[ -z $DRY ]]
	then
		echo "Saving boundary precision recall into ${BASELINE_BPR_OUTPUT}"
		${BPR_COMMAND} > "${BASELINE_BPR_OUTPUT}"
		echo "run: $BASELINEDIR" >> "${BASELINE_BPR_OUTPUT}"
		echo "BASELINE_OUTPUT=${BASELINE_OUTPUT}" >> "${BASELINE_BPR_OUTPUT}"
		echo "COMMON_PARAMS=${COMMON_PARAMS}" >> "${BASELINE_BPR_OUTPUT}"
		echo "BASELINE_TRAIN_PARAMS=${BASELINE_TRAIN_PARAMS}" >> "${BASELINE_BPR_OUTPUT}"
		echo "BASELINE_TRAIN_PARAMS=${BASELINE_TRAIN_PARAMS}" >> "${BASELINE_BPR_OUTPUT}"
		echo "COMMON_TEST_PARAMS=${COMMON_TEST_PARAMS}" >> "${BASELINE_BPR_OUTPUT}"
	else
		${BPR_COMMAND} ">" ${BASELINE_BPR_OUTPUT}
	fi
fi

if [[ ! -z ${BASELINE_ONLY} ]]
then
	echo "Only making baseline."
	exit 0
fi

############ Catmap
# Train catmap model
if [[ -z ${POSTPROCESSING_ONLY} ]]
then
	TRAIN_START_TIME=`date +"%Y.%m.%d %H:%M:%S"`
	${DRY} ${PYTHON} ${CATMAP} ${COMMON_PARAMS} ${CATMAP_TRAIN_PARAMS} -B ${BASELINE_OUTPUT} -s ${CATMAP_MODEL} ${LOG} ${STATS}
	check_return
	TRAIN_END_TIME=`date +"%Y.%m.%d %H:%M:%S"`
fi

### Testing
# Segment test data using catmap
${DRY} ${PYTHON} ${CATMAP} ${COMMON_PARAMS} ${COMMON_TEST_PARAMS} ${CATMAP_TEST_PARAMS} -l ${CATMAP_MODEL} -T ${TESTDATA} -o ${CATMAP_TEST_OUTPUT} -m none
	check_return

# Calculate boundary precision recall
# (This is run using regular python regardless of choice above, due to numpy dependency)
BPR_COMMAND="${DRY} python ${BPR} -g ${GOLDSTD} -p ${CATMAP_TEST_OUTPUT}"
if [[ -z $DRY ]]
then
	echo "Saving boundary precision recall into ${CATMAP_BPR_OUTPUT}"
	${BPR_COMMAND} > "${CATMAP_BPR_OUTPUT}"
	echo "run: $RUN_TITLE" >> "${CATMAP_BPR_OUTPUT}"
	echo "train-start: ${TRAIN_START_TIME}" >> "${CATMAP_BPR_OUTPUT}"
	echo "train-end  : ${TRAIN_END_TIME}" >> "${CATMAP_BPR_OUTPUT}"
	echo "BASELINE_OUTPUT=${BASELINE_OUTPUT}" >> "${CATMAP_BPR_OUTPUT}"
	echo "COMMON_PARAMS=${COMMON_PARAMS}" >> "${CATMAP_BPR_OUTPUT}"
	echo "BASELINE_TRAIN_PARAMS=${BASELINE_TRAIN_PARAMS}" >> "${CATMAP_BPR_OUTPUT}"
	echo "CATMAP_TRAIN_PARAMS=${CATMAP_TRAIN_PARAMS}" >> "${CATMAP_BPR_OUTPUT}"
	echo "COMMON_TEST_PARAMS=${COMMON_TEST_PARAMS}" >> "${CATMAP_BPR_OUTPUT}"
	echo "CATMAP_TEST_PARAMS${CATMAP_TEST_PARAMS}" >> "${CATMAP_BPR_OUTPUT}"
else
	${BPR_COMMAND} ">" ${CATMAP_BPR_OUTPUT}
fi
