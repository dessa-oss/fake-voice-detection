apt-get update
yes | apt-get install wget
yes | apt-get install zip
yes | apt-get install unzip

cd ../data/

mkdir inference_data/
cd inference_data/
wget https://asv-audio-data-atlas.s3.amazonaws.com/realtalk.zip
unzip realtalk.zip -d .
mv realtalk/* .
rm realtalk.zip

cd ..
mkdir logical_access/
cd logical_access/

wget https://asv-audio-data-atlas.s3.amazonaws.com/preprocessed_data.zip
unzip preprocessed_data.zip -d .
rm preprocessed_data.zip



