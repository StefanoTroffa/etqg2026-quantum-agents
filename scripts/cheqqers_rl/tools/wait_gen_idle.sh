#!/usr/bin/env bash
# wait_gen_idle.sh: block until gen_loop8 and fw_loop (pidfiles in ${QG_T33:-$HOME/qg_t33}/gen) have exited. Pidfiles, not pgrep:
# a pgrep pattern matched a waiting parent's own command line and deadlocked panel C for 40 min (14 Nov).
G=${QG_T33:-$HOME/qg_t33}/gen
for f in gen_loop.pid generator.pid fw_loop.pid trainer.pid; do
  while [ -s $G/$f ] && kill -0 "$(cat $G/$f)" 2>/dev/null; do sleep 15; done
done
