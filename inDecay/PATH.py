import os as _os
user_dir = _os.environ.get('INDECAY_USER_DIR', '/rds/user/wz369/hpc-work')
main_dir = _os.environ.get('INDECAY_MAIN_DIR', f"{user_dir}/inDecay")
data_dir = f"{main_dir}/data"
somatic_dir = f"{data_dir}/somatic"
embryo_raw_dir = f"{main_dir}/zygote/"
high_dir = data_dir
pth_dir = f"{main_dir}/pl_trainer_log"
toolkit_dir = f"{main_dir}/tool"

import sys
sys.path.append(main_dir)
Indelgen = f"{main_dir}/tool/indelgentarget"
Indelana = f'{user_dir}/SelfTarget/indel_analysis/'

# Singularity container for indelgentarget (used when local binary is absent)
selftarget_sif = _os.environ.get(
    'SELFTARGET_SIF',
    f"{user_dir}/containers/selftarget.sif"
)
