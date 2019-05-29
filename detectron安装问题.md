## 1. 安装错误：

修改Dockerfile

"

RUN git clone https://github.com/facebookresearch/detectron /detectron
**RUN cd /detectron && git checkout d56e267**
"

## 2. 无法连网 

<1> 首先配置docker容器的DNS

cat /etc/resolv.conf       该命令查看容器和宿主机的DNS设置，容器和宿主机的DNS设置应该一致。

docker DNS的设置可以通过启动容器时添加参数  --dns  8.8.8.8  来设置（本次有效）

或者通过 /etc/default/docker的DNS参数来进行永久设置，然后重启DOCKER生效，

     export http_proxy="http://10.1.9.100:808/"
   DOCKER_OPTS="--dns 210.83.210.155 --dns 127.0.1.1"

 

sudo service docker restart

## 3. 无法apt-get update

(1)首先更改为国内源镜像，在 /etc/apt路径下,先将sources.list文件进行备份。

cp sources.list sources.list.bak

注：如果docker容器中没有gredit、vim等文本编辑器。可以使用cat > sources.list覆盖原文件中所有的内容。

(2)更改sources.list文件中内容

https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

将网址中灰色内容全部替换sources.list内容

(3)更新源，如果不成功就继续配置接下来的步骤。

(4)在/etc/apt/路径下，将sources.list.d文件更名为sources.list.d.odd

mv sources.list.d sources.list.d.odd

然后更新源，就可以解决卡住的问题。

（更新源之后可以将文件名再改回来，具体什么原因我还没明白，如果有懂得大佬求解答）