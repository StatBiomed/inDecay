GPU1=2
GPU2=3
e1=ST_June_2017_E14TG2A_LV7A_DPI7
e2=ST_June_2017_BOB_LV7A_DPI7

run_ETG_BOB(){
    feat_i=$1
    python scripts/STfeatv5_inDecay.py -E ${e1} -C 500 -G 0 -S "$feat_i:$((feat_i+30))" -M ST_Decay > ETG_training.log &
    python scripts/STfeatv5_inDecay.py -E ${e1} -C 500 -G 1 -S "$((feat_i+30)):$((feat_i+60))" -M ST_Decay > ETG_training.log &

    python scripts/STfeatv5_inDecay.py -E ${e2} -C 500 -G 2 -S "$feat_i:$((feat_i+30))" -M ST_Decay > ETG_training.log &
    python scripts/STfeatv5_inDecay.py -E ${e2} -C 500 -G 3 -S "$((feat_i+30)):$((feat_i+60))" -M ST_Decay > ETG_training.log &
}

for i in {0..275..60};do
    run_ETG_BOB $i
    pid=$!
    wait $pid
    echo "finished waiting ${pid} moving forward" 
    done
