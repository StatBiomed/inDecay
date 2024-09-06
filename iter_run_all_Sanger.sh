Cell_Name=$1
GPU=$2

find data/Sanger*  -maxdepth 1 -type d |while read folder; do
    
    DataArchive=$(basename $folder)

    if [ $DataArchive = "Indelgen_result" ];
    then
        continue
    else
        echo ${DataArchive}
        bash run_v5_Sanger.sh ${DataArchive} ${Cell_Name} ${GPU}
        pid=$!
        echo "waiting for ${DataArchive}, pid is $pid"
        wait $pid
        echo "${DataArchive} finished, moving forward"
    fi

done


echo " "
echo "==================================="
echo "ALL iteration training finished !!!"
echo "==================================="
echo " "