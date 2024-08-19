# "ST_June_2017_BOB_LV7A_DPI7
# ST_June_2017_CHO_LV7A_DPI7
# ST_June_2017_E14TG2A_LV7A_DPI7
# ST_June_2017_HAP1_LV7A_DPI7
# ST_June_2017_K562_LV7A_DPI7"

e=ST_June_2017_BOB_LV7A_DPI7

for i in {0..275..50};do
    python scripts/STfeatv5_inDecay.py -E $e -C 500 -G 6 -S "$i:$((i+50))"
done


