#!/usr/bin/python3

#SBATCH --job-name=summary_data_all
###SBATCH --partition=high2    # peloton node: 32 cores, 7.8 GB per core, 250 GB total
#SBATCH --partition=high2m    # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=3:00:00
#SBATCH --output=/home/ibsantis/scripts/jobs/summary/summary_data_all_%j.txt
#SBATCH --mail-user=ibsantistevan@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin
"""
    Run the summary_data_all.py script
"""

import os
from utilities.basic import io as ut_io

# print run-time and CPU information
ScriptPrint = ut_io.SubmissionScriptClass("slurm")
os.system("python /home/ibsantis/scripts/orbit_analysis/summary_data_submit_all.py")
# print run-time information
ScriptPrint.print_runtime()
