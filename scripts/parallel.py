import sys
import os
import math

def main():
    args = sys.argv[1:]
    num_gpus = int(args[1])

    gpu = args[0]
    print(f"Running on GPU {gpu}")

    gpu_short_name = gpu
    if '-' in gpu_short_name:
        gpu_short_name = gpu_short_name.split('-')[0]

    range_start = 1
    range_end = int(args[2])

    num_jobs = range_end - range_start + 1
    jobs_per_gpu = max(int(math.ceil(num_jobs / num_gpus)), 1)

    job_start = range_start
    job_num = 1

    confirmation = input(f"Ensure that run_classification_all.sh is up-to-date.\nThis script will use slurm to run {jobs_per_gpu} jobs in each {gpu}.\nAre you sure you want to continue (y/n)? ")
    if confirmation != 'y':
        print("Aborting...")
        return

    while job_start <= range_end:
        tmux_session_name = f"{job_num}-{gpu_short_name}"
        
        os.system(f"tmux kill-session -t \"{tmux_session_name}\"")
        os.system(f"tmux new-session -d -s \"{tmux_session_name}\"")
        os.system(f"tmux send-keys -t \"{tmux_session_name}\" ./srun.sh Space {gpu} Space {job_start} Space {min(range_end, job_start + jobs_per_gpu)} Enter")
        
        job_start += jobs_per_gpu
        job_num += 1
      

# tmux new-session -d -s "sesi"
# tmux send-keys -t "HelloWorld" pwd Enter

# Check if main
if __name__ == "__main__":
    main()