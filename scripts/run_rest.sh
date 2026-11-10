#!/bin/sh
# 等第一条链（sched/bias/operator）结束后，再补跑 compose / session / denom；同样限速在 35% 单核。
cd /root/autodl-tmp/cohort-rag/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 COHORT_DUTY=1.0
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/cohort-rag/reports/icde2027_suite/run.log
until grep -q "ALL DONE" $LOG; do sleep 20; done
nice -n 19 $PY -u run_icde2027_suite.py --exp compose,session,denom --seeds 10 >> $LOG 2>&1 &
PID=$!
cpulimit -p $PID -l 35 -z -q &
CL=$!
wait $PID
kill $CL 2>/dev/null
echo "$(date +%H:%M:%S) REST DONE" >> $LOG
