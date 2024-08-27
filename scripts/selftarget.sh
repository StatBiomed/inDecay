main_dir=$(pwd)

cd tool/
git clone https://github.com/felicityallen/SelfTarget
cd SelfTarget
pip install -r requirements.txt

git sparse-checkout set --no-cone indel_analysis/indelmap
git checkout

# the python dependent
# cd selftarget_pyutils
# pip install -e .
# cd ../indel_prediction
# pip install -e .

cd indel_analysis/indelmap
cmake . -DINDELMAP_OUTPUT_DIR="$main_dir/tool"
make && make install
export INDELGENTARGET_EXE="$main_dir/tool"

cd $main_dir
echo 'Indelgen="'$main_dir/tool/indelgentarget'"' >> inDecay/PATH.py
echo 'Tooldir="'$main_dir/tool/indelgentarget'"' >> scripts/PATH.py

if [ -e tool/indelgen ]
then
    rm -rf tool/SelfTarget
    printf "\n=================\n"
    echo "indelgen installed !"
    printf "=================\n"
else
    echo "error auto-installing"
    echo "please install indelgen mannually"
    echo "via the instruction under https://github.com/felicityallen/SelfTarget/tree/master"
fi