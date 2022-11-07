# DATA Preprocessing

1) Download data from https://kylberg.org/kylberg-texture-dataset-v-1-0/
2) Run process_kylberg.py
```
python3 process_kylberg.py
```
3) Upload processed data to s3 cloud object storage (For example YC Object Storage)
```
aws --endpoint-url=https://storage.yandexcloud.net \
  s3 cp --recursive data/ s3://bucket-name/path_style_prefix/
```

