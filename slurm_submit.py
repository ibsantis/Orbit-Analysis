#!/usr/bin/env python3

# Stampede2 SKX node: 48 cores, 4 GB per core, 192 GB total
#SBATCH --job-name=m12i_halo_potentials
#SBATCH --partition=skx-normal
#SBATCH --nodes=1
#SBATCH --tasks-per-node=48    # MPI tasks per node
#SBATCH --cpus-per-task=1    # OpenMP threads per MPI task
#SBATCH --time=03:00:00
#SBATCH --output=/home1/05400/ibsantis/scripts/gizmo_jobs/m12i_halo_potentials_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin
#SBATCH --account=TG-AST140064

'''
Submit a Gizmo simulation job to the SLURM queue.
Check if restart files exist - if yes, start a restart run.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

import os

from utilities.basic import io as ut_io

import os
from utilities.basic import io as ut_io    # if you want to use my print diagnostics

# print run-time and CPU information
ScriptPrint = ut_io.SubmissionScriptClass('slurm')
os.system('python /home1/05400/ibsantis/scripts/orbit_analysis/halo_potentials.py')
# print run-time information
ScriptPrint.print_runtime()
