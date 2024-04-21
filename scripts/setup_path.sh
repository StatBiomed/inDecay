# under inDecay repo
main_dir=$(pwd)
echo "main_dir='$(pwd)'" | cat - scripts/PATH.py > temp && mv temp scripts/PATH.py
echo "main_dir='$(pwd)'" | cat - inDecay/PATH.py > temp && mv temp inDecay/PATH.py

cd ../SelfTarget/indel_analysis/indelmap
Tooldir=$(pwd)

cd $main_dir
echo 'Tooldir="'$Tooldir'"' >> inDecay/PATH.py
echo 'Tooldir="'$Tooldir'"' >> scripts/PATH.py