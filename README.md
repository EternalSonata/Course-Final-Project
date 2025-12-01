Create and activate your virtual environment:

conda create --name myenv --file explicit.txt
conda activate myenv
pip install -r requirements.txt


In terms of dataset, if the network environment supports huggingface, you can directly load the data by running the script.

There are total three scripts, if you want to conduct all experiments:

cd CS5489-final
chmod +x ./run_rnn.sh
./run_rnn.sh
chmod +x ./run_transformer.sh
./run_transformer.sh
chmod +x ./run_mt_multi_tfm.sh
./run_mt_multi_tfm.sh

Here, run_rnn.sh contain all experiments about GRU and LSTM. run_transformer.sh contain all experiments about transformer, and run_mt_multi_tfm contain all multilingual experiments.

The folder structure:

CS5489-final/
│
├── MT_baselines.py
├── visualization.ipynb     
├── run_rnn.sh
├── run_transformer.sh
├── run_mt_multi_tfm.sh
├── explicit.txt          # Conda explicit environment specification
├── requirements.txt      # Pip dependencies
├── README.md             # Project setup guide
