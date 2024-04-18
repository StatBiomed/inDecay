read -p "Enter the path of folder you wanted to : " Data_dir

# path sanity check
if [ ! -d "$Data_dir" ]; then
  echo "$Data_dir does not exist, create a new dir"
  mkdir $Data_dir
fi

cd $Data_dir

echo "ready to download pre-training data"
curl https://figshare.com/ndownloader/articles/25133564/versions/2 --output inDecay_data.zip

echo "unzipping the downloaded data"
unzip inDecay_data.zip
rm inDecay_data.zip

# untar the folders
tar -xzf processed_dfs.tar.gz &
tar -xzf Indelgen_result.tar.gz 
rm *tar.gz

echo "finished"
echo "pre-training data saved to $(pwd)"


