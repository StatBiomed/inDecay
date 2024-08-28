Sanger_Directoriy_Name=$1
Cell_Name=$2
GPU=$3

run_dif_fix(){
    k_index=$1
    python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_pretrained.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False &
    python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_pretrained.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False --Fix_params "del_regressor[:1]" &
    python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_pretrained.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False --Fix_params "del_regressor[:2]" &
}

for i in {0..4};do
    run_dif_fix $i
    pid=$!
    wait $pid
    echo "finished , moving forward"
done