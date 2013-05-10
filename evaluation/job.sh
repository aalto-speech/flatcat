#!/bin/bash
#SBATCH --time=0-15:00:00 --mem-per-cpu=2000
#SBATCH -p batch
#SBATCH -o morphochal07_fin_n100.out

BASE=`basename $0 .sh`

if [[ $BASE == "job" ]]
then
	echo "Dont call me directly"
	exit 1
fi

if [[ ! -z $DRYFLAG && $DRYFLAG != '-n' ]]
then
	echo "DRYFLAG set to unrecognized value. Defaulting to '-n' (dryrun) for safety."
	export DRYFLAG="-n"
fi

### Default parameters

#export PYTHONPATH=/triton/ics/scratch/sgronroo/morfessor
export PYTHONPATH=~/morfessor/morfessor
export PYTHONIOENCODING="utf-8"

EVALUATE_SCRIPT="./bin/evaluate.sh"
RUN_TITLE="${BASE}"

# Force splitting on these characters
FORCESPLIT='":-"'      # hyphen can't be first char
PPL_THRESH=10
EPOCH_COST="0.0025"
ITER_COST="0.005"

# FIXME outfile for sbatch?

# Parameters given to all training and testing commands
export COMMON_PARAMS="-e latin-1"

# Parameters only given to baseline commands
#export BASELINEDIR=""
export DATASET="toy"	#FIXME
export BASELINE_TRAIN_PARAMS="--traindata-list"
export BASELINE_TEST_PARAMS=""

# Parameters only given to catmap commands
CATMAP_TRAIN_CORE="-p ${PPL_THRESH}"
CATMAP_TRAIN_SEQ="--min-iteration-cost-gain ${ITER_COST} --min-epoch-cost-gain ${EPOCH_COST}"
CATMAP_TRAIN_EXTRA=""
export CATMAP_TEST_PARAMS=""


### Experiment specific overrides

# FIXME: input sizes through DATASET
case $BASE in
	("default")
		echo "Running with default parameters..."
		;;
	("heuristics")
		# These are postprocessing variants, so all are run in the same job,
		# starting with default parameters (redundant to combine with default)
		echo "Running with variants of heuristics (4 variants + default)..."
		RUN_TITLE="heuristics_off"
		DATE=`date +%Y%m%d_%H%M%S`
		export OUTDIR="${DATE}_${BASE}"
		;;
	("ppl_thresh_A")
		PPL_THRESH=15
		CATMAP_TRAIN_CORE="-p ${PPL_THRESH}"
		echo "Running with --perplexity-threshold ${PPL_THRESH}"
		;;
	("ppl_thresh_B")
		PPL_THRESH=30
		CATMAP_TRAIN_CORE="-p ${PPL_THRESH}"
		echo "Running with --perplexity-threshold ${PPL_THRESH}"
		;;
	("ppl_thresh_C")
		PPL_THRESH=60
		CATMAP_TRAIN_CORE="-p ${PPL_THRESH}"
		echo "Running with --perplexity-threshold ${PPL_THRESH}"
		;;
	("ppl_thresh_D")
		PPL_THRESH=160
		CATMAP_TRAIN_CORE="-p ${PPL_THRESH}"
		echo "Running with --perplexity-threshold ${PPL_THRESH}"
		;;
	("ppl_thresh_E")
		PPL_THRESH=400
		CATMAP_TRAIN_CORE="-p ${PPL_THRESH}"
		echo "Running with --perplexity-threshold ${PPL_THRESH}"
		;;
	("dampening_log")
		export COMMON_PARAMS="${COMMON_PARAMS} --dampening log"
		export BASELINEDIR="baseline_${DATASET}_log"
		echo "Running with log dampening"
		;;
	("dampening_types")
		export COMMON_PARAMS="${COMMON_PARAMS} --dampening ones"
		export BASELINEDIR="baseline_${DATASET}_types"
		echo "Running with dampening to types"
		;;
	("forcesplit")
		export BASELINEDIR="baseline_${DATASET}_forcesplit"
		export BASELINE_TRAIN_PARAMS="${BASELINE_TRAIN_PARAMS} -f ${FORCESPLIT}"
		CATMAP_TRAIN_EXTRA="-f ${FORCESPLIT}"
		;;
	(*)
		echo "Unknown job ${BASE}"
                exit 1
                ;;	
esac

### Run evaluation

export CATMAP_TRAIN_PARAMS="${CATMAP_TRAIN_CORE} ${CATMAP_TRAIN_SEQ} ${CATMAP_TRAIN_EXTRA}"
$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG

if [[ $BASE == "heuristics" ]]
then
	echo "Running variant: heuristics on..."
	export POSTPROCESSING_ONLY="True"

	RUN_TITLE="heuristics_on"
	export CATMAP_TEST_PARAMS="--remove-nonmorphemes"
	$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG

	RUN_TITLE="heuristics_lts_disabled"
	export CATMAP_TEST_PARAMS="--remove-nonmorphemes --nonmorpheme-heuristics 'join-two join-all'"
	$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG

	RUN_TITLE="heuristics_jt_disabled"
	export CATMAP_TEST_PARAMS="--remove-nonmorphemes --nonmorpheme-heuristics 'longest-to-stem join-all'"
	$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG

	RUN_TITLE="heuristics_ja_disabled"
	export CATMAP_TEST_PARAMS="--remove-nonmorphemes --nonmorpheme-heuristics 'longest-to-stem join-two'"
	$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG
fi
