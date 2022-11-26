# BOINC-TCNN-balancer
Development of a load balancing system for volunteer distributed computing project (Texture classification CNN learning)

## Data preprocessing
Follow instructions in ```preprocessing/```

### Kafka Log Broker deploy
Deployment of Apache Kafka logger
1) Deploy Kafka as a service on Yandex Cloud
2) Get certificates
   ```bash
   wget "https://storage.yandexcloud.net/cloud-certs/CA.pem" \
   --output-document YandexInternalRootCA.crt && \
   chmod 655 YandexInternalRootCA.crt
   ```
3) Specify kafka addr, login, password in ```credentials.ini```

## Load Balancer deploy
Deployment of load balancer in yandex-cloud
1) Download [```terraform```](https://www.terraform.io/downloads.html)
2) Install [```yc-cli```](https://cloud.yandex.ru/docs/cli/quickstart#install)
3) Create service account and init terraform (follow instructions [here](https://cloud.yandex.ru/docs/tutorials/infrastructure-management/terraform-quickstart#install-terraform))
4) Create docker container registry in yandex-cloud, configure docker:
   ```bash
   yc container registry create --name boinc-load-balancer
   yc container registry configure-docker
   ```
5) Build load balancer docker image and push it to registry:
   ```bash
   docker build . -f load_balancer/Dockerfile -t cr.yandex/<registry ID>/load_balancer
   docker push cr.yandex/<registry ID>/load_balancer
   ```
   <!---
   ID=crpqve2rbj2resnjl14t 
   cmd:
   docker build . -f load_balancer/Dockerfile -t cr.yandex/crpqve2rbj2resnjl14t/load_balancer
   docker push cr.yandex/crpqve2rbj2resnjl14t/load_balancer
   -->
6) Refresh env variables:
   ```bash
   export YC_TOKEN=$(yc iam create-token)
   export YC_CLOUD_ID=$(yc config get cloud-id)
   export YC_FOLDER_ID=$(yc config get folder-id)
   ```
7) Deploy load balancer using terraform: \
    ```bash
   terraform validate
   terraform plan
   terraform apply
   terraform output -json > terraform_output.json
   ```
   (debug) check server docker logs:
   ```bash
   ssh <username>@<server public ip>
   sudo journalctl -u yc-container-daemon
   ```
8) Deploy BOINC server
9) Run experiments
10) ...
11) Get Logs
12) Stop load balancer:
   ```bash
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