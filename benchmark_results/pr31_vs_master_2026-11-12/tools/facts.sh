echo "\$ date -u"; date -u; echo "\$ nproc"; nproc; echo "\$ lscpu | head"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|L1d|L2|L3" ; echo "\$ free -m"; free -m; echo "\$ uname -a"; uname -a; echo "\$ df -h ~/models"; df -h ~/models | tail -2
echo "\$ sha256sum model (matches the Hugging Face x-linked-etag)"; cat ~/evidence/model.sha256
