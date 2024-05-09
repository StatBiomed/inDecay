main_dir=$(pwd)

cd ../
git clone https://github.com/felicityallen/SelfTarget.git
cd SelfTarget
# the python dependent
pip install -r requirements.txt
cd selftarget_pyutils
pip install -e .
cd ../indel_prediction
pip install -e .

cd ../indel_analysis/indelmap
cmake . -DINDELMAP_OUTPUT_DIR="$main_dir/tool"
make && make install
export INDELGENTARGET_EXE="$main_dir/tool"

cd $main_dir
echo 'Indelgen="'$main_dir/tool/indelgentarget'"' >> inDecay/PATH.py
echo 'Tooldir="'$main_dir/tool/indelgentarget'"' >> scripts/PATH.py