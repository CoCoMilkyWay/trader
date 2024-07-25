export clash_ip="127.0.0.1"
export http_proxy="http://$clash_ip:7890"
export https_proxy="http://$clash_ip:7890"
export ftp_proxy="http://$clash_ip:7890"
export httpProxy="http://$clash_ip:7890"
export httpsProxy="http://$clash_ip:7890"
export ftpProxy="http://$clash_ip:7890"
export HTTP_PROXY="http://$clash_ip:7890"
export HTTPS_PROXY="http://$clash_ip:7890"
alias pip="pip --proxy http://$clash_ip:7890"

git config pull.rebase false
git config --global user.name "CoCoMilkyWay"
git config --global user.email "wangchuyin980321@gmail.com"
git config --global http.proxy http://$clash_ip:7890
git config --global https.proxy http://$clash_ip:7890
conda config --set proxy_servers.http http://$clash_ip:7890
conda config --set proxy_servers.https http://$clash_ip:7890
conda config --set ssl_verify false

# git config --global --unset http.proxy
# git config --global --unset https.proxy