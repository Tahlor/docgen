#!/bin/bash

WAIT=120
GPU_THRESHOLD=40
program="./generate_lines_OTHERGPU.py"
gpu_index=0
if ! [ -f $program ] ; then
    echo "File $program does not exist"
    program="./generate_lines2.py"
    echo "Using $program"
    gpu_index=1
fi

cpu_threshold=40

echod() {
  local current_time=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$current_time] $@"
}

# Get power usage of GPU
is_gpu_power_draw_above() {
    local gpu_index="{$1:-$gpu_index}"
    local threshold_power_level="{$2:-$GPU_THRESHOLD}"
    local power_draw

    # Get power draw of the specified GPU in watts
    power_draw=$(nvidia-smi -i "$gpu_index" --query-gpu=power.draw --format=csv,noheader,nounits | awk '{print $1}')

    # Check if the power draw is above the specified power level
    awk -v pd="$power_draw" -v tpl="$threshold_power_level" 'BEGIN {if (pd > tpl) exit 0; else exit 1;}'
    return $?
}

is_cpu_usage_above() {
  local pid="${1:-$PID}" # Use 'local' keyword to restrict the scope of the variable
  #local cpu_usage=$(pidstat -p $pid | tail -1 | awk '{print $4}') # Extract CPU usage and ensure you're getting the correct line
  local cpu_percent=$(ps -p $pid -o %cpu --no-headers)
  echod "CPU: $cpu_percent%"
  awk -v usage="$cpu_percent" -v threshold="$cpu_threshold" 'BEGIN {exit !(usage > threshold)}'
  return $?
}

start() {
  return
  pkill -f python
  sleep 3

  echod "Starting $program"
  nohup python $program &
  PID=$!
  echod "PID: $PID"
  echod "Started new process, waiting 600 seconds before evaluating"
  sleep 600 # give it a headstart to set up model, download, etc.
}

# START IT
start

while true; do
    if ! is_gpu_power_draw_above "$gpu_index" "$GPU_THRESHOLD"; then
        echod "GPU power usage is under 40W, this is a warning, checking again in 3 minutes"

        usage=$(cpu_usage $PID)
        echod "CPU Usage: $usage%"

        if is_cpu_usage_above; then
          echod "CPU usage for PID $PID is over the threshold ($cpu_threshold%), waiting 30 seconds and starting over"
          sleep 30
          continue
        fi

        sleep 180
        if ! is_gpu_power_draw_above; then
          echod "GPU power usage is under 40W AGAIN, killing it"

          pkill -f "$program"
          kill -9 $PID
          pkill -f python
          sleep 5

          start
        fi
    else
        echod "GPU power usage is over 40W, seems to be running"
    fi

    echod "Sleeping for $WAIT..."
    sleep $WAIT
done