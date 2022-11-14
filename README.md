# BOINC-TCNN-balancer
Development of a load balancing system for volunteer distributed computing project (Texture classification CNN learning)

## Data preprocessing
Follow instructions in ```preprocessing/```

## Load Balancer deploy
Deployment of load balancer in yandex-cloud
1) Download [```terraform```](https://www.terraform.io/downloads.html)
2) Install [```yc-cli```](https://cloud.yandex.ru/docs/cli/quickstart#install)
3) Create service account and init terraform (follow instructions [here](https://cloud.yandex.ru/docs/tutorials/infrastructure-management/terraform-quickstart#install-terraform))
4) Create docker container registry in yandex-cloud, configure docker:
   ```commandline
   yc container registry create --name boinc-load-balancer
   yc container registry configure-docker
   ```
5) Build load balancer docker image and push it to registry:
   ```
   docker build . -t cr.yandex/<registry ID>/load_balancer
   docker push cr.yandex/<registry ID>/load_balancer
   ```
   <!---ID=crpqve2rbj2resnjl14t cmd=docker build . -t cr.yandex/crpqve2rbj2resnjl14t/load_balancer-->
6) Deploy load balancer using terraform: \
    ```
   terraform validate
   terraform plan
   terraform apply
   ```
7) Deploy BOINC server
8) Run experiments
9) ...
10) Get Logs
11) Stop load balancer:
     ```
    terraform destroy
    ```

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