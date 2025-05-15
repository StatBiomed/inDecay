user_dir='/home/louisayu/ssd'
main_dir=f"{user_dir}/inDecay"
data_dir = f"{main_dir}/data"
embryo_raw_dir = f"{main_dir}/zygote/"
high_dir = data_dir
pth_dir = f"{main_dir}/pl_trainer_log"
toolkit_dir = f"{main_dir}/tool"

import sys
sys.path.append(main_dir)
Indelgen=f"{main_dir}/tool/indelgentarget"
Indelana=f'{user_dir}/SelfTarget/indel_analysis/'
