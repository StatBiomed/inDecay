# under inDecay repo
main_dir=$(pwd)

if [ -e tool ]; then
    echo "folder tool exist" 
else 
    mkdir tool
fi

echo "main_dir='$(pwd)'" | cat - scripts/PATH.py > temp && mv temp scripts/PATH.py
echo "main_dir='$(pwd)'" | cat - inDecay/PATH.py > temp && mv temp inDecay/PATH.py

cd $main_dir