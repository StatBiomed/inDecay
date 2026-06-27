import os as _os
# Default to the repository root (two levels up from this file) so the package is
# portable on any clone. Override with INDECAY_MAIN_DIR / INDECAY_USER_DIR on HPC.
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
user_dir = _os.environ.get('INDECAY_USER_DIR', _REPO)
main_dir = _os.environ.get('INDECAY_MAIN_DIR', _REPO)
data_dir = f"{main_dir}/data"
somatic_dir = f"{data_dir}/somatic"
embryo_raw_dir = f"{main_dir}/zygote/"
high_dir = data_dir
pth_dir = f"{main_dir}/pl_trainer_log"
toolkit_dir = f"{main_dir}/tool"

import sys
sys.path.append(main_dir)
Indelgen = f"{main_dir}/tool/indelgentarget"
Indelana = _os.environ.get('INDECAY_INDELANA', f"{main_dir}/tool/SelfTarget/indel_analysis/")

# Singularity container for indelgentarget (used when local binary is absent)
selftarget_sif = _os.environ.get(
    'SELFTARGET_SIF',
    f"{user_dir}/containers/selftarget.sif"
)
