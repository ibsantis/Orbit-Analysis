#!/usr/bin/env python3

#SBATCH --job-name=m12b_mass_profile
###SBATCH --partition=high2    # peloton node: 32 cores, 7.8 GB per core, 250 GB total
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=240G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32    # MPI tasks per node
#SBATCH --cpus-per-task=1    # OpenMP threads per MPI task
#SBATCH --time=24:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/mass_profiles/m12b_mass_profile_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin
"""
Submit a job to the SLURM queue.
@author: Isaiah Santistevan <ibsantistevan@ucdavis.edu>
"""

import os
from utilities.basic import io as ut_io

# print run-time and CPU information
ScriptPrint = ut_io.SubmissionScriptClass("slurm")
os.system("python /home/ibsantis/scripts/orbit_analysis/mass_profile_parallel.py")
# print run-time information
ScriptPrint.print_runtime()
