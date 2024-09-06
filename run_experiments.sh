
run_with_2GPU(){
CutOff=$1
GPU1=$2
GPU2=$3

Experiments="ST_June_2017_BOB_LV7A_DPI7
ST_June_2017_E14TG2A_LV7A_DPI7
ST_June_2017_HAP1_LV7A_DPI7"

# Cells=("iPSC" "mESC" "CHO" "K562" "HAP1")
# i=0
for e in $Experiments; do
    # cellname=${Cells[i]}
    # echo $e $cellname
    python scripts/STfeatv5_inDecay_faster.py -E $e -C ${CutOff} -G ${GPU1} --progress_bar False &
    done


Experiments="ST_June_2017_CHO_LV7A_DPI7
ST_June_2017_K562_LV7A_DPI7"

for e in $Experiments; do
    python scripts/STfeatv5_inDecay_faster.py -E $e -C ${CutOff} -G ${GPU2} --progress_bar False &
    done
}

run_with_2GPU 100 1 2 > C100_training.log &
run_with_2GPU 200 3 4 > C200_training.log &
run_with_2GPU 500 5 6 > C500_training.log &



