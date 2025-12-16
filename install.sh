python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple # set mirror
python -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install .
cp doc/unix.sh jx
jx -h
echo "Recommend: Add $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) to PATH"