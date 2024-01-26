# GNM_VAE
Preprocess inet-data
````
python process_inet_to_gnm.py
python data_split.py --data-dir '/home/zishuo/inet_gnm_data' --dataset-name 'inet_gnm_data' --data-splits-dir 'inet_mid_gnm'
````

Train GNM
````
python train.py --config config/gnm.yaml
````

Train GNM_VAE
````
python train.py --config config/gnm_vae.yaml
````
