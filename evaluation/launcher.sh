#!/bin/bash

# Job script (contains the actual experiment parameters)
PARAM_SCRIPT="job.sh"

# How long to sleep between calls to the scheduler (seconds)
DELAY=2

if [[ $1 == "--local" ]]
then
	SCHEDULER="./"
else
	SCHEDULER="sbatch "
fi

BASELINE_MISSING=""
ROOT_DIR="results_morphochal07_fin_n100"	# FIXME
BASELINE_DIR="baseline_default"
BASELINE_FILE="baseline.gz"


function schedule_job {
	${SCHEDULER}${PARAM_SCRIPT} ${JOB}
	sleep ${DELAY}
}

function check_baseline {
	if [[ ! -e ${ROOT_DIR}/${BASELINE_DIR}/${BASELINE_FILE} ]]
	then
		BASELINE_MISSING="True"
		PARAM_SCRIPT="job_baseline.sh"
		JOB="${BASELINE_DIR}"; schedule_job
	fi
}

# Which baselines do we need
BASELINE_DIR="baseline_default"; check_baseline
BASELINE_DIR="baseline_log"; check_baseline
BASELINE_DIR="baseline_ones"; check_baseline
BASELINE_DIR="baseline_forcesplit"; check_baseline
if [[ ! -z $BASELINE_MISSING ]]
then
	echo "Some baselines were missing. Scheduled them instead."
	echo "Rerun after the baselines have been finished."
	exit 1
fi

### Which experiments to run
#JOB='default'; schedule_job
#
#JOB='heuristics'; schedule_job
#JOB='dampening_log'; schedule_job
#JOB='dampening_types'; schedule_job
#JOB='forcesplit'; schedule_job
#
#JOB='ppl_thresh_A'; schedule_job
#JOB='ppl_thresh_B'; schedule_job
#JOB='ppl_thresh_C'; schedule_job
#JOB='ppl_thresh_D'; schedule_job
#JOB='ppl_thresh_E'; schedule_job

JOB='max_epochs_A'; schedule_job
JOB='max_epochs_B'; schedule_job
JOB='max_epochs_C'; schedule_job
JOB='max_epochs_D'; schedule_job
JOB='max_epochs_E'; schedule_job
JOB='max_epochs_F'; schedule_job
JOB='max_epochs_G'; schedule_job
JOB='max_epochs_H'; schedule_job

#JOB='cost_limits_A'; schedule_job
#JOB='cost_limits_B'; schedule_job
#JOB='cost_limits_C'; schedule_job
#JOB='cost_limits_D'; schedule_job
#JOB='cost_limits_E'; schedule_job
#JOB='cost_limits_F'; schedule_job
#JOB='cost_limits_G'; schedule_job
#JOB='cost_limits_H'; schedule_job
#JOB='cost_limits_I'; schedule_job
#
#JOB='exclude_split1'; schedule_job
#JOB='exclude_join'; schedule_job
#JOB='exclude_split2'; schedule_job
#JOB='exclude_shift'; schedule_job
#JOB='exclude_resegment'; schedule_job
#
#JOB='shift_A'; schedule_job
#JOB='shift_B'; schedule_job
#JOB='shift_C'; schedule_job
#JOB='shift_D'; schedule_job
#JOB='shift_E'; schedule_job
#JOB='shift_F'; schedule_job
#JOB='shift_G'; schedule_job
#JOB='shift_H'; schedule_job
#
#JOB='corpus_weight_learning'; schedule_job
#JOB='semisupervised_auto'; schedule_job
#
#JOB='semisupervised_penalty_A'; schedule_job
#JOB='semisupervised_penalty_B'; schedule_job
#JOB='semisupervised_penalty_C'; schedule_job
#JOB='semisupervised_penalty_D'; schedule_job
#JOB='semisupervised_penalty_E'; schedule_job
#JOB='semisupervised_penalty_F'; schedule_job
#JOB='semisupervised_penalty_G'; schedule_job
#JOB='semisupervised_penalty_H'; schedule_job
#JOB='semisupervised_penalty_I'; schedule_job
