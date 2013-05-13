#!/bin/bash
#SBATCH --time=0-1:00:00 --mem-per-cpu=1000
#SBATCH -p short
#SBATCH -o baseline_%j.out

if [[ $# < 1 ]]
then
	echo "Usage: sbatch job_baseline.sh <RUN-TITLE>"
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

# Parameters given to all training and testing commands
export COMMON_PARAMS="-e latin-1"

# Parameters only given to baseline commands
#export BASELINEDIR=""
export BASELINE_TRAIN_PARAMS="--traindata-list"
export BASELINE_TEST_PARAMS=""


### Experiment specific overrides

# FIXME: input sizes through DATASET
case $BASE in
	("baseline_default")
		export BASELINE_ONLY="True"
		echo "Making default baseline..."
		;;
	("baseline_log")
		export BASELINE_ONLY="True"
		export COMMON_PARAMS="${COMMON_PARAMS} --dampening log"
		export BASELINEDIR="baseline_log"
		echo "Making log dampened baseline..."
		;;
	("baseline_types")
		export BASELINE_ONLY="True"
		export COMMON_PARAMS="${COMMON_PARAMS} --dampening ones"
		export BASELINEDIR="baseline_types"
		echo "Making type dampened baseline..."
		;;
	("baseline_forcesplit")
		export BASELINE_ONLY="True"
		export BASELINEDIR="baseline_forcesplit"
		export BASELINE_TRAIN_PARAMS="${BASELINE_TRAIN_PARAMS} -f ${FORCESPLIT}"
		echo "Making forcesplit baseline..."
		;;
esac

### Run evaluation

$EVALUATE_SCRIPT $RUN_TITLE $DRYFLAG
