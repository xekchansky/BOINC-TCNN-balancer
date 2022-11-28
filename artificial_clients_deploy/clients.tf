terraform {
  required_providers {
    yandex = {
      source = "yandex-cloud/yandex"
    }
  }
  required_version = ">= 0.13"
}

provider "yandex" {
  zone = "ru-central1-a"
}

data "yandex_compute_image" "container-optimized-image" {
  family = "container-optimized-image"
}

### 1 ###
resource "yandex_compute_instance" "client0_instance" {
  #name = "load_balancer"

  service_account_id = "ajebtl7fah698jqc8v6o"

  resources {
    cores  = 8
    memory = 8
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.container-optimized-image.id
    }
  }

  network_interface {
    subnet_id = "e9bo4jvd7onk0aatsre4"
    nat       = true
  }

  metadata = {
    docker-container-declaration = file("${path.module}/declaration.yaml")
    user-data                    = file("${path.module}/cloud-config.yaml")
  }
}

### 2 ###
resource "yandex_compute_instance" "client1_instance" {
  #name = "load_balancer"

  service_account_id = "ajebtl7fah698jqc8v6o"

  resources {
    cores  = 4
    memory = 8
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.container-optimized-image.id
    }
  }

  network_interface {
    subnet_id = "e9bo4jvd7onk0aatsre4"
    nat       = true
  }

  metadata = {
    docker-container-declaration = file("${path.module}/declaration.yaml")
    user-data                    = file("${path.module}/cloud-config.yaml")
  }
}

### 3 ###
resource "yandex_compute_instance" "client2_instance" {
  #name = "load_balancer"

  service_account_id = "ajebtl7fah698jqc8v6o"

  resources {
    cores  = 2
    memory = 4
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.container-optimized-image.id
    }
  }

  network_interface {
    subnet_id = "e9bo4jvd7onk0aatsre4"
    nat       = true
  }

  metadata = {
    docker-container-declaration = file("${path.module}/declaration.yaml")
    user-data                    = file("${path.module}/cloud-config.yaml")
  }
}

### 4 ###
resource "yandex_compute_instance" "client3_instance" {
  #name = "load_balancer"

  service_account_id = "ajebtl7fah698jqc8v6o"

  resources {
    cores  = 2
    memory = 4
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.container-optimized-image.id
    }
  }

  network_interface {
    subnet_id = "e9bo4jvd7onk0aatsre4"
    nat       = true
  }

  metadata = {
    docker-container-declaration = file("${path.module}/declaration.yaml")
    user-data                    = file("${path.module}/cloud-config.yaml")
  }
}

### 5 ###
resource "yandex_compute_instance" "client4_instance" {
  #name = "load_balancer"

  service_account_id = "ajebtl7fah698jqc8v6o"

  resources {
    cores  = 4
    memory = 8
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.container-optimized-image.id
    }
  }

  network_interface {
    subnet_id = "e9bo4jvd7onk0aatsre4"
    nat       = true
  }

  metadata = {
    docker-container-declaration = file("${path.module}/declaration.yaml")
    user-data                    = file("${path.module}/cloud-config.yaml")
  }
}

### 6 ###
resource "yandex_compute_instance" "client5_instance" {
  #name = "load_balancer"

  service_account_id = "ajebtl7fah698jqc8v6o"

  resources {
    cores  = 8
    memory = 16
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.container-optimized-image.id
    }
  }

  network_interface {
    subnet_id = "e9bo4jvd7onk0aatsre4"
    nat       = true
  }

  metadata = {
    docker-container-declaration = file("${path.module}/declaration.yaml")
    user-data                    = file("${path.module}/cloud-config.yaml")
  }
}

### 7 ###
resource "yandex_compute_instance" "client6_instance" {
  #name = "load_balancer"

  service_account_id = "ajebtl7fah698jqc8v6o"

  resources {
    cores  = 2
    memory = 4
  }

  boot_disk {
    initialize_params {
      image_id = data.yandex_compute_image.container-optimized-image.id
    }
  }

  network_interface {
    subnet_id = "e9bo4jvd7onk0aatsre4"
    nat       = true
  }

  metadata = {
    docker-container-declaration = file("${path.module}/declaration.yaml")
    user-data                    = file("${path.module}/cloud-config.yaml")
  }
}

output "external_ip_address_client_node" {
  value = yandex_compute_instance.client0_instance.network_interface.0.nat_ip_address
}