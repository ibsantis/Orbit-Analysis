#!/usr/bin/env python3

#SBATCH --job-name=test_script
#SBATCH --partition=skx-normal    # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
##SBATCH --mem=62G    # need to specify memory if you set the number of tasks (--ntasks) below
#SBATCH --nodes=1    # if you specify this, the number of nodes, do not set memory (--mem) above
#SBATCH --ntasks-per-node=1    # (MPI) tasks per node
#SBATCH --ntasks=1    # (MPI) tasks total
#SBATCH --cpus-per-task=1    # (OpenMP) threads per (MPI) task
#SBATCH --time=00:15:00
#SBATCH --output=/home1/05400/ibsantis/job_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

import os
from utilities.basic import io as ut_io    # if you want to use my print diagnostics

# print run-time and CPU information
ScriptPrint = ut_io.SubmissionScriptClass('slurm')
os.system('python /home1/05400/ibsantis/scripts/orbit_analysis/test_slurm.py')
# print run-time information
ScriptPrint.print_runtime()
