#!/bin/bash
idle_seconds=200
program=$1
gpu_index=0
if ! [ $program ] || ! [ -f $program ] ; then
    echo "File $program does not exist"
    program="./generate_lines2.py"
    echo "Using $program"
    gpu_index=1
fi

_base=$(basename "$program" .py)
LOG_FILE="./LOG_${_base}.log"

echod() {
  local current_time=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$current_time] $@"
}


restart() {
  pkill -f python
  sleep 3

  echod "Starting $program"
  if [ -f $LOG_FILE.old ]; then
    mv $LOG_FILE.old $LOG_FILE.old.old
  fi
  if [ -f $LOG_FILE ]; then
    mv $LOG_FILE $LOG_FILE.old
  fi
  nohup python $program > $LOG_FILE 2>&1 &
  PID=$!
  echod "PID: $PID"
  echod "Started $program using GPU $gpu_index"
  sleep 60
}

restart

while true; do
  last_modified=$(stat -c %Y "$LOG_FILE")
  current_time=$(date +%s)
  diff_seconds=$((current_time - last_modified))

  # Parse the log file for date and time
  #  last_timestamp=$(grep -o -E "(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})" "$LOG_FILE" | tail -1)
  #  current_time=$(date "+%Y-%m-%d %H:%M:%S")
  #  diff_seconds=$(($(date -d "$current_time" +%s) - $(date -d "$last_timestamp" +%s)))


  # Check if the difference is more significant than your threshold, e.g., 300 seconds (5 minutes)
  if [ "$diff_seconds" -gt $idle_seconds ]; then
    echod "Model is hanging. Restarting..."
    restart
  fi

  sleep 60
done

