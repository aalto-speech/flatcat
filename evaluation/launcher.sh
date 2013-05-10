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

function schedule_job {
	${SCHEDULER}${PARAM_SCRIPT} ${JOB}
	sleep ${DELAY}
}

# Which experiments to run
JOB='heuristics'; schedule_job
JOB='ppl_thresh_A'; schedule_job
JOB='ppl_thresh_B'; schedule_job
JOB='ppl_thresh_C'; schedule_job
JOB='ppl_thresh_D'; schedule_job
JOB='ppl_thresh_E'; schedule_job
JOB='dampening_log'; schedule_job
JOB='dampening_types'; schedule_job
JOB='forcesplit'; schedule_job

JOB='default'; schedule_job
