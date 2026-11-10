#!/bin/sh
cd /root/autodl-tmp/cohort-rag/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 COHORT_DUTY=1.0
LOG=/root/autodl-tmp/cohort-rag/reports/icde2027_suite/run.log
nice -n 19 /root/miniconda3/bin/python -u run_icde2027_extra.py --exp basis >> $LOG 2>&1 &
PID=$!
cpulimit -p $PID -l 35 -z -q &
CL=$!
wait $PID
kill $CL 2>/dev/null
echo "$(date +%H:%M:%S) BASIS DONE" >> $LOG
