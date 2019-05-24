# Docker

## 1. 容器与虚拟机的区别
![](https://www.docker.com/sites/default/files/d8/2018-11/docker-containerized-appliction-blue-border_2.png)

CONTAINERS
Containers are an abstraction at the app layer that packages code and dependencies together. Multiple containers can run on the same machine and share the OS kernel with other containers, each running as isolated processes in user space. Containers take up less space than VMs (container images are typically tens of MBs in size), can handle more applications and require fewer VMs and Operating systems.


![](https://www.docker.com/sites/default/files/d8/2018-11/container-vm-whatcontainer_2.png)

VIRTUAL MACHINES
Virtual machines (VMs) are an abstraction of physical hardware turning one server into many servers. The hypervisor allows multiple VMs to run on a single machine. Each VM includes a full copy of an operating system, the application, necessary binaries and libraries - taking up tens of GBs. VMs can also be slow to boot.

## 2. 安装

Older versions of Docker were called docker, docker.io , or docker-engine. If these are installed, uninstall them:
```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

Update the apt package index:
```
sudo apt-get update
```
Install packages to allow apt to use a repository over HTTPS:
```
 sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```
Add Docker’s official GPG key:
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```
Verify that you now have the key with the fingerprint 9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88, by searching for the last 8 characters of the fingerprint.

```
$ sudo apt-key fingerprint 0EBFCD88
    
pub   rsa4096 2017-02-22 [SCEA]
      9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid           [ unknown] Docker Release (CE deb) <docker@docker.com>
sub   rsa4096 2017-02-22 [S]
```

Use the following command to set up the stable repository. To add the nightly or test repository, add the word nightly or test (or both) after the word stable in the commands below. Learn about nightly and test channels.

```
 sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```
Update the apt package index
```
sudo apt-get update
```
Install the latest version of Docker CE and containerd:
```
sudo apt-get install docker-ce docker-ce-cli containerd.io

```
Verify that Docker CE is installed correctly by running the hello-world image..

```
sudo docker run hello-world
```

Uninstall Docker CE

```
$ sudo apt-get purge docker-ce
```
Images, containers, volumes, or customized configuration files on your host are not automatically removed. To delete all images, containers, and volumes:
```
$ sudo rm -rf /var/lib/docker
```

Nvidia docker安装：

```
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker

# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

# 3. 基本概念

![](https://docs.docker.com/engine/images/architecture.svg)

## The Docker daemon
The Docker daemon (dockerd) listens for Docker API requests and manages Docker objects such as images, containers, networks, and volumes. A daemon can also communicate with other daemons to manage Docker services.

## The Docker client
The Docker client (docker) is the primary way that many Docker users interact with Docker. When you use commands such as docker run, the client sends these commands to dockerd, which carries them out. The docker command uses the Docker API. The Docker client can communicate with more than one daemon.

## Docker registries
A Docker registry stores Docker images. Docker Hub is a public registry that anyone can use, and Docker is configured to look for images on Docker Hub by default. You can even run your own private registry. If you use Docker Datacenter (DDC), it includes Docker Trusted Registry (DTR).

When you use the docker pull or docker run commands, the required images are pulled from your configured registry. When you use the docker push command, your image is pushed to your configured registry.

## Docker objects
When you use Docker, you are creating and using images, containers, networks, volumes, plugins, and other objects. This section is a brief overview of some of those objects.

### IMAGES

An image is a read-only template with instructions for creating a Docker container. Often, an image is based on another image, with some additional customization. For example, you may build an image which is based on the ubuntu image, but installs the Apache web server and your application, as well as the configuration details needed to make your application run.

You might create your own images or you might only use those created by others and published in a registry. To build your own image, you create a Dockerfile with a simple syntax for defining the steps needed to create the image and run it. Each instruction in a Dockerfile creates a layer in the image. When you change the Dockerfile and rebuild the image, only those layers which have changed are rebuilt. This is part of what makes images so lightweight, small, and fast, when compared to other virtualization technologies.

### CONTAINERS

A container is a runnable instance of an image. You can create, start, stop, move, or delete a container using the Docker API or CLI. You can connect a container to one or more networks, attach storage to it, or even create a new image based on its current state.

By default, a container is relatively well isolated from other containers and its host machine. You can control how isolated a container’s network, storage, or other underlying subsystems are from other containers or from the host machine.

A container is defined by its image as well as any configuration options you provide to it when you create or start it. When a container is removed, any changes to its state that are not stored in persistent storage disappear.

### Example docker run command

The following command runs an ubuntu container, attaches interactively to your local command-line session, and runs /bin/bash.

```
$ docker run -i -t ubuntu /bin/bash
```
When you run this command, the following happens (assuming you are using the default registry configuration):

If you do not have the ubuntu image locally, Docker pulls it from your configured registry, as though you had run docker pull ubuntu manually.

Docker creates a new container, as though you had run a docker container create command manually.

Docker allocates a read-write filesystem to the container, as its final layer. This allows a running container to create or modify files and directories in its local filesystem.

Docker creates a network interface to connect the container to the default network, since you did not specify any networking options. This includes assigning an IP address to the container. By default, containers can connect to external networks using the host machine’s network connection.

Docker starts the container and executes /bin/bash. Because the container is running interactively and attached to your terminal (due to the -i and -t flags), you can provide input using your keyboard while the output is logged to your terminal.

When you type exit to terminate the /bin/bash command, the container stops but is not removed. You can start it again or remove it.

## SERVICES

Services allow you to scale containers across multiple Docker daemons, which all work together as a swarm with multiple managers and workers. Each member of a swarm is a Docker daemon, and the daemons all communicate using the Docker API. A service allows you to define the desired state, such as the number of replicas of the service that must be available at any given time. By default, the service is load-balanced across all worker nodes. To the consumer, the Docker service appears to be a single application. Docker Engine supports swarm mode in Docker 1.12 and higher.


# 4. 操作

## 4.1 SSH 登录

ssh服务器安装：
```
$ sudo apt-get install openssh-server
```
检查ssh服务器是否运行：

```
$ ps -e | grep ssh

```
如果看到sshd那说明ssh-server已经启动了。 


SSH登录服务器

登录：

```
ssh ubuntu@192.168.12.191 
```
退出：
```
exit
```

## 4.2 Docker基础操作

### 4.2.1 指令集

![](https://upload-images.jianshu.io/upload_images/10839544-37caa4f2eac9d994.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/554/format/webp)

镜像创建容器：docker create
镜像启动容器：docker run
拉取镜像：docker pull
推送镜像：docker push
给镜像打标签：docker tag
列出本地镜像：docker image ls/docker images
删除本地镜像：docker rmi/docker image rm
构建镜像：docker commit, docker build
查询镜像构建过程：docker history
查询镜像的详细信息：docker inspect
将镜像打成tar包：docker save
登录登出仓库：docker login/docker logout


### 4.2.2 操作示例


在远程image仓库里搜索hello-world 镜像
```
$ docker search hello-world
```

从远程仓库拉去hello-world镜像
```
$ docker pull hello-world
```
![](https://upload-images.jianshu.io/upload_images/10839544-fa05199bd4f54c16.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/554/format/webp)

查看本地镜像：

```
$ docker images
```
![](https://upload-images.jianshu.io/upload_images/10839544-f563021dea742338.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/554/format/webp)

