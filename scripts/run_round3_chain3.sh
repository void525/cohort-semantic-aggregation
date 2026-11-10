#!/bin/sh
# Round-3 chain v3: after chain2 finishes, rerun the (cheap) audit grid with 10 seeds over all six budgets,
# and the scheduler comparison with 10 seeds instead of 4.
cd /root/autodl-tmp/cohort-rag/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 COHORT_DUTY=1.0
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/cohort-rag/reports/logs_round3/chain.log
while kill -0 "$1" 2>/dev/null; do sleep 5; done
echo "$(date +%H:%M:%S) chain3 start" >> $LOG
nice -n 5 $PY -u run_icde2027_extra3.py --exp audit --seeds 10 >> $LOG 2>&1
echo "$(date +%H:%M:%S) audit10 done" >> $LOG
nice -n 5 $PY -u run_icde2027_suite.py --exp sched --seeds 10 >> $LOG 2>&1
echo "$(date +%H:%M:%S) sched10 done" >> $LOG
echo "$(date +%H:%M:%S) CHAIN3 DONE" >> $LOG
