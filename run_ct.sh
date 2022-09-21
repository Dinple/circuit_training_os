#!/bin/bash 

name="$1"
collectcnt="$2"

export TF_GPU_ALLOCATOR=cuda_malloc_async
export ROOT_DIR=./logs/run_$name

# check if ROOT_DIR already exists
if [[ -d $ROOT_DIR ]] ; then
        echo "Directory $ROOT_DIR exists."
        exit 1
fi

# REVERB SERVER INFO
export REVERB_PORT=8008
export REVERB_SERVER="127.0.0.1:${REVERB_PORT}"

# INPUT INFO
export NETLIST_FILE=./circuit_training/environment/test_data/ariane/netlist.pb.txt
export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane/initial.plc

echo "NETLIST: $NETLIST_FILE"
echo "INIT_PLC: $INIT_PLACEMENT"
echo "Saving log file at $ROOT_DIR"
echo "Launching Tmux Session ..."

# Create tmux session
tmux new-session -d -s reverb_server && \
   tmux new-session -d -s train_job && \
   tmux new-session -d -s eval_job && \
   tmux new-session -d -s tb_job

# Start reverb_server
tmux send-keys -t reverb_server.0 "tf" ENTER;
tmux send-keys -t reverb_server.0 "python3 -m circuit_training.learning.ppo_reverb_server  --root_dir=${ROOT_DIR}  --port=${REVERB_PORT}" ENTER;

# Start train_job
tmux send-keys -t train_job.0 "tf" ENTER;
tmux send-keys -t train_job.0 "python3 -m circuit_training.learning.train_ppo  --root_dir=${ROOT_DIR}  --replay_buffer_server_address=${REVERB_SERVER}  --variable_container_server_address=${REVERB_SERVER}  --num_episodes_per_iteration=16  --global_batch_size=64  --netlist_file=${NETLIST_FILE} --init_placement=${INIT_PLACEMENT}" ENTER

# Start collect_job_00
for ((i=0;i<collectcnt;i++))
do
  session="collect_job_$i"
  echo "starting $session"
  tmux new-session -d -s "$session"
  tmux send-keys -t "$session.0" "tf" ENTER;
  tmux send-keys -t "$session.0" "export CUDA_VISIBLE_DEVICES=-1" ENTER;
  tmux send-keys -t "$session.0" "python3 -m circuit_training.learning.ppo_collect  --root_dir=${ROOT_DIR}  --replay_buffer_server_address=${REVERB_SERVER}  --variable_container_server_address=${REVERB_SERVER}  --task_id=$i  --netlist_file=${NETLIST_FILE}  --init_placement=${INIT_PLACEMENT}" ENTER  
done

# Start eval_job
tmux send-keys -t eval_job.0 "tf" ENTER;
tmux send-keys -t eval_job.0 "python3 -m circuit_training.learning.eval  --root_dir=${ROOT_DIR}  --variable_container_server_address=${REVERB_SERVER}  --netlist_file=${NETLIST_FILE}  --init_placement=${INIT_PLACEMENT}" ENTER

# Start tb_job
tmux send-keys -t tb_job.0 "tf" ENTER;
tmux send-keys -t tb_job.0 "tensorboard dev upload --logdir ./logs" ENTER

# Attach to train_job session
tmux attach -t  train_job
