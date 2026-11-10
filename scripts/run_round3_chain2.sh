#!/bin/sh
# Round-3 chain v2: sequential (the container has a 0.5-core quota), most important results first.
cd /root/autodl-tmp/cohort-rag/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 COHORT_DUTY=1.0
export LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1
PY=/root/miniconda3/bin/python
VPY=/root/autodl-tmp/venv_lotus/bin/python
LOG=/root/autodl-tmp/cohort-rag/reports/logs_round3/chain.log
# wait for the short eg/pooled side job (pid passed as $1)
while kill -0 "$1" 2>/dev/null; do sleep 5; done
echo "$(date +%H:%M:%S) chain2 start" >> $LOG
nice -n 5 $PY -u run_icde2027_extra3.py --exp bias_real >> $LOG 2>&1
echo "$(date +%H:%M:%S) bias_real done" >> $LOG
nice -n 5 $PY -u run_icde2027_extra3.py --exp audit --seeds 5 >> $LOG 2>&1
echo "$(date +%H:%M:%S) audit done" >> $LOG
nice -n 5 $VPY -u run_icde2027_extra3.py --exp lotus >> $LOG 2>&1
echo "$(date +%H:%M:%S) lotus done" >> $LOG
nice -n 5 $PY -u run_icde2027_extra3.py --exp bias_sweep >> $LOG 2>&1
echo "$(date +%H:%M:%S) bias_sweep done" >> $LOG
echo "$(date +%H:%M:%S) CHAIN2 DONE" >> $LOG
