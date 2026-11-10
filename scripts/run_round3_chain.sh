#!/bin/sh
# Round-3 background chain (single core; nice'd). Order = what the paper text needs first.
cd /root/autodl-tmp/cohort-rag/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 COHORT_DUTY=1.0
export LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1
PY=/root/miniconda3/bin/python
VPY=/root/autodl-tmp/venv_lotus/bin/python
LOG=/root/autodl-tmp/cohort-rag/reports/logs_round3/chain.log
echo "$(date +%H:%M:%S) chain start" >> $LOG
nice -n 15 $PY -u run_icde2027_suite.py --exp session >> $LOG 2>&1
echo "$(date +%H:%M:%S) session done" >> $LOG
nice -n 15 $PY -u run_icde2027_extra3.py --exp eg,memory >> $LOG 2>&1
echo "$(date +%H:%M:%S) eg+memory done" >> $LOG
nice -n 15 $PY -u run_icde2027_extra3.py --exp bias20 >> $LOG 2>&1
echo "$(date +%H:%M:%S) bias20 done" >> $LOG
nice -n 15 $PY -u run_icde2027_extra3.py --exp audit --seeds 10 >> $LOG 2>&1
echo "$(date +%H:%M:%S) audit done" >> $LOG
nice -n 15 $VPY -u run_icde2027_extra3.py --exp lotus >> $LOG 2>&1
echo "$(date +%H:%M:%S) lotus done" >> $LOG
echo "$(date +%H:%M:%S) CHAIN DONE" >> $LOG
