# BOINC-TCNN-balancer
Development of a load balancing system for volunteer distributed computing project (Texture classification CNN learning)

## Data preprocessing
Follow instructions in ```preprocessing/```

## Load Balancer deploy

## BOINC server deploy
start BOINC server:
```
sudo docker-compose pull
sudo docker-compose up -d
```

submit job:
```
docker-compose exec apache bash
bin/boinc2docker_create_work.py python:alpine python -c "print('Hello BOINC')"
bin/boinc2docker_create_work.py -it --network=host -v /mnt/share/ssh:/root/.ssh my_horovod bash app/run.sh
```