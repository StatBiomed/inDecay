Experiments="ST_June_2017_BOB_LV7A_DPI7
    ST_June_2017_CHO_LV7A_DPI7
    ST_June_2017_E14TG2A_LV7A_DPI7
    ST_June_2017_HAP1_LV7A_DPI7
    ST_June_2017_K562_LV7A_DPI7"

StemCell="ST_June_2017_BOB_LV7A_DPI7
    ST_June_2017_E14TG2A_LV7A_DPI7"

OtherCell="ST_June_2017_CHO_LV7A_DPI7
    ST_June_2017_HAP1_LV7A_DPI7
    ST_June_2017_K562_LV7A_DPI7"

# for i in {0..275..50};do
#     echo $i, $((i+50))
# done
i=250
python scripts/STfeatv5_inDecay.py -E $e -C 500 -G 0 -S "$i:$((i+50))"
# for e in $StemCell;do
#     # nohup python scripts/deepDecay_ratio.py -E $e -C 500 -G 0 &
#     nohup python scripts/STfeatv5_inDecay.py -E $e -C 500 -G 0 -S 0:35 &
#     pid=$!
#     echo $e $pid
# done

# for e in $OtherCell;do
#     nohup python scripts/STfeatv5_inDecay.py -E $e -C 500 -G 3 -S 0:35 &
#     pid=$!
#     echo $e $pid
# done
