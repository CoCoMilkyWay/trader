export clash_ip="127.0.0.1"
export clash_ip_port="7890"
export http_proxy="http://$clash_ip:$clash_ip_port"
export https_proxy="http://$clash_ip:$clash_ip_port"
export ftp_proxy="http://$clash_ip:$clash_ip_port"
export httpProxy="http://$clash_ip:$clash_ip_port"
export httpsProxy="http://$clash_ip:$clash_ip_port"
export ftpProxy="http://$clash_ip:$clash_ip_port"
export HTTP_PROXY="http://$clash_ip:$clash_ip_port"
export HTTPS_PROXY="http://$clash_ip:$clash_ip_port"
alias pip="pip --proxy http://$clash_ip:$clash_ip_port"
# for older version of python, you may need to use http mirror (not secure but works)
# pip install pip numpy pandas matplotlib urllib3 scikit-learn openpyxl -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --proxy http://127.0.0.1:7890 -t C:\Users\chuyin.wang\Desktop\share\fin\国金证券QMT交易端\bin.x64\Lib\site-packages

git config pull.rebase false
git config --global user.name "CoCoMilkyWay"
git config --global user.email "wangchuyin980321@gmail.com"
git config --global http.proxy http://$clash_ip:$clash_ip_port
git config --global https.proxy http://$clash_ip:$clash_ip_port
conda config --set proxy_servers.http http://$clash_ip:$clash_ip_port
conda config --set proxy_servers.https http://$clash_ip:$clash_ip_port
conda config --set ssl_verify false
# git config --global --unset http.proxy
# git config --global --unset https.proxy

# pip install --trusted-host pypi.python.org --upgrade pip


