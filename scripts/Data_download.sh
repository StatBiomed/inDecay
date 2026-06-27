#!/bin/bash
cd data
echo "ready to download pre-training data"
wget --user-agent="Mozilla/5.0" \
     --content-disposition \
     -O inDecay_data.zip \
     "https://figshare.com/ndownloader/articles/25133564/versions/3"

# Check if download was successful
if [ ! -f "inDecay_data.zip" ]; then
    echo "Download failed! Please check the URL and try again."
    exit 1
fi

echo "unzipping the downloaded data"
unzip inDecay_data.zip
rm inDecay_data.zip

# Check if tar files exist before extracting
if [ -f "processed_dfs.tar.gz" ]; then
    tar -xzf processed_dfs.tar.gz &
else
    echo "processed_dfs.tar.gz not found!"
fi

if [ -f "Indelgen_result.tar.gz" ]; then
    tar -xzf Indelgen_result.tar.gz
else
    echo "Indelgen_result.tar.gz not found!"
fi

# Clean up only if files exist
[ -f "processed_dfs.tar.gz" ] && rm processed_dfs.tar.gz
[ -f "Indelgen_result.tar.gz" ] && rm Indelgen_result.tar.gz

mv processed_df/ somatic/
mv Indelgen_result/ somatic/Indelgen_result/

echo "finished"
echo "pre-training data saved to $(pwd)"
