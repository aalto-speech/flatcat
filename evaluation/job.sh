#!/bin/bash
#SBATCH --time=0-15:00:00 --mem-per-cpu=2000
#SBATCH -p batch
#SBATCH -o morphochal07_fin_n100_%j.out

if [[ $# < 1 ]]
then
	echo "Usage: sbatch job.sh <RUN-TITLE>"
	exit 1
fi
BASE=$1

if [[ ! -z $DRYFLAG && $DRYFLAG != '-n' ]]
then
	echo "DRYFLAG set to unrecognized value. Defaulting to '-n' (dryrun) for safety."
	export DRYFLAG="-n"
fi

### Default parameters

# Dataset selection
export DATASET="toy"	#debug
#export DATASET="morphochal07_fin_n100"

export PYTHONPATH=~/morfessor/morfessor
#export PYTHONPATH=/triton/ics/scratch/sgronroo/morfessor

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
		export BASELINEDIR="baseline_log"
		echo "Running with log dampening"
		;;
	("dampening_types")
		export COMMON_PARAMS="${COMMON_PARAMS} --dampening ones"
		export BASELINEDIR="baseline_types"
		echo "Running with dampening to types"
		;;
	("forcesplit")
		export BASELINEDIR="baseline_forcesplit"
		export BASELINE_TRAIN_PARAMS="${BASELINE_TRAIN_PARAMS} -f ${FORCESPLIT}"
		CATMAP_TRAIN_EXTRA="-f ${FORCESPLIT}"
		echo "Running with forcesplit"
		;;
	("max_epochs_A")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 1 --max-epochs 1"
		echo "Running with max epochs"
		;;
	("max_epochs_B")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 2 --max-epochs 1"
		echo "Running with max epochs"
		;;
	("max_epochs_C")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 1 --max-epochs 2"
		echo "Running with max epochs"
		;;
	("max_epochs_D")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 2 --max-epochs 2"
		echo "Running with max epochs"
		;;
	("max_epochs_E")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 5 --max-epochs 2"
		echo "Running with max epochs"
		;;
	("max_epochs_F")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 1 --max-epochs 5"
		echo "Running with max epochs"
		;;
	("max_epochs_G")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 2 --max-epochs 5"
		echo "Running with max epochs"
		;;
	("max_epochs_H")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-epochs-first 5 --max-epochs 5"
		echo "Running with max epochs"
		;;
	("cost_limits_A")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.002 --min-epoch-cost-gain 0.001 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_B")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.002 --min-epoch-cost-gain 0.0025 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_C")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.002 --min-epoch-cost-gain 0.01 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_D")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.005 --min-epoch-cost-gain 0.001 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_E")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.005 --min-epoch-cost-gain 0.005 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_F")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.005 --min-epoch-cost-gain 0.01 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_G")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.05 --min-epoch-cost-gain 0.001 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_H")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.05 --min-epoch-cost-gain 0.0025 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("cost_limits_I")
		CATMAP_TRAIN_SEQ="--min-iteration-cost-gain 0.05 --min-epoch-cost-gain 0.01 --max-epochs-first 5 --max-epochs 5"
		echo "Running with cost limits"
		;;
	("exclude_split1")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --training-operations join,split,shift,resegment"
		echo "Running with first split excluded"
		;;
	("exclude_join")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --training-operations split,split,shift,resegment"
		echo "Running with join excluded"
		;;
	("exclude_split2")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --training-operations split,join,shift,resegment"
		echo "Running with second split excluded"
		;;
	("exclude_shift")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --training-operations split,join,split,resegment"
		echo "Running with shift excluded"
		;;
	("exclude_resegment")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --training-operations split,join,split,shift"
		echo "Running with resegment excluded"
		;;
	("shift_A")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 1 --min-shift-remainder 1"
		echo "Running with shift parameters"
		;;
	("shift_B")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 1 --min-shift-remainder 2"
		echo "Running with shift parameters"
		;;
	("shift_C")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 1 --min-shift-remainder 3"
		echo "Running with shift parameters"
		;;
	("shift_D")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 2 --min-shift-remainder 1"
		echo "Running with shift parameters"
		;;
	("shift_E")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 2 --min-shift-remainder 3"
		echo "Running with shift parameters"
		;;
	("shift_F")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 3 --min-shift-remainder 1"
		echo "Running with shift parameters"
		;;
	("shift_G")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 3 --min-shift-remainder 2"
		echo "Running with shift parameters"
		;;
	("shift_H")
		CATMAP_TRAIN_SEQ="${CATMAP_TRAIN_SEQ} --max-shift-distance 3 --min-shift-remainder 3"
		echo "Running with shift parameters"
		;;
	("corpus_weight_learning")
		CATMAP_TRAIN_EXTRA="--develset data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running with corpus weight learning"
		;;
	("semisupervised_auto")
		CATMAP_TRAIN_EXTRA="--annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with automatic weight"
		;;
	("semisupervised_penalty_A")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 10 --annotation-supermorph-penalty 0 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_B")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 10 --annotation-supermorph-penalty 10 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_C")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 10 --annotation-supermorph-penalty 100 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_D")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 100 --annotation-supermorph-penalty 0 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_E")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 100 --annotation-supermorph-penalty 10 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_F")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 100 --annotation-supermorph-penalty 100 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_G")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 999999 --annotation-supermorph-penalty 0 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_H")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 999999 --annotation-supermorph-penalty 10 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
		;;
	("semisupervised_penalty_I")
		CATMAP_TRAIN_EXTRA="--annotation-penalty 999999 --annotation-supermorph-penalty 100 --annotations data/morphochal10_fin_train.goldstd.segmentation"
		echo "Running semisupervised, with penalties"
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
	export CATMAP_TEST_PARAMS="--remove-nonmorphemes --nonmorpheme-heuristics join-two,join-all"
	$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG

	RUN_TITLE="heuristics_jt_disabled"
	export CATMAP_TEST_PARAMS="--remove-nonmorphemes --nonmorpheme-heuristics longest-to-stem,join-all"
	$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG

	RUN_TITLE="heuristics_ja_disabled"
	export CATMAP_TEST_PARAMS="--remove-nonmorphemes --nonmorpheme-heuristics longest-to-stem,join-two"
	$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG
fi
