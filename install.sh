pip install uv
python -m uv venv --clear
python -m uv sync
cp doc/unix.sh gtools
chmod +x gtools