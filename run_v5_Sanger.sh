Sanger_Directoriy_Name=$1
Cell_Name=$2
GPU=$3

run_dif_fix(){
    k_index=$1
    python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_c100.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False --modelnote C100_mESC_valmsecon_l21e1_lr3e-4 &
    python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_c100.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False --Fix_params "del_regressor[:1]" --modelnote C100_mESC_valmsecon_l21e1_lr3e-4 &
    python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_c100.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False --Fix_params "del_regressor[:2]" --modelnote C100_mESC_valmsecon_l21e1_lr3e-4 &
    #python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_pretrained.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False &
    #python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_pretrained.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False --Fix_params "del_regressor[:1]" &
    #python scripts/STfeatv5_inDecay_Sanger.py -E ${Sanger_Directoriy_Name} -C 1 -G ${GPU} -P "pretrained/${Cell_Name}_featv5_pretrained.ckpt" -T ${k_index} -M ST_DeepDecay --progress_bar False --Fix_params "del_regressor[:2]" &
}

for i in {0..31};do
    run_dif_fix $i
    pid=$!
    echo "pid for $i is $pid, waiting"
    wait $pid
    echo "finished, moving forward"
done

echo " "
echo "========================="
echo "ALL training finished !!!"
echo "========================="
echo " "