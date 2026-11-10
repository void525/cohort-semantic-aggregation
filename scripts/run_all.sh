#!/bin/sh
# 串行运行补充实验。容器 CPU 配额为 0.5 核：用 cpulimit 把后台任务持续限制在 ~35% 单核，
# 不触发 cgroup 限流，因而不会冻结同容器里的 IDE 服务。
cd /root/autodl-tmp/cohort-rag/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 COHORT_DUTY=1.0
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/cohort-rag/reports/icde2027_suite/run.log
run() {
  nice -n 19 $PY -u "$@" >> $LOG 2>&1 &
  PID=$!
  cpulimit -p $PID -l 35 -z -q &
  CL=$!
  wait $PID
  kill $CL 2>/dev/null
}
run run_icde2027_suite.py --exp compose,session,denom,calib --seeds 10
run run_icde2027_suite.py --exp sched --seeds 4
run run_icde2027_suite.py --exp bias
run run_icde2027_extra.py --exp operator --sessions 16
echo "$(date +%H:%M:%S) ALL DONE" >> $LOG
