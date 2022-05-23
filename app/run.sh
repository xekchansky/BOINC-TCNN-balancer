#!/bin/bash

[ ! -d "data/" ] && unzip data.zip
rm data.zip

python3 send_ip.py

bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
# python3 train_horovod.py
