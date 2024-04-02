

ARCH="cuda"
VERSION="v0.0.8"
IMAGE="ghcr.io/mila-iqia/milabench:$ARCH-$VERSION"


# Download the AO config
wget https://gist.githubusercontent.com/Delaunay/ef0b2dc4aae5d42c16f00a17d17d490e/raw/acc5dabb908b3b9f918c9c5716711d4b82569da7/AO_476.yaml

mkdir configs
mv AO_476.yaml configs/AO_476.yaml

echo ">> Print report"
echo "==============="
docker run -it --rm                                                                 \
    -v $(pwd)/results:/milabench/envs/runs                                          \
    -v $(pwd)/configs:/milabench/envs/configs                                       \
    $IMAGE                                                                          \
    milabench report --config /milabench/envs/configs/AO_476.yaml --runs /milabench/envs/runs

echo "<< ============"





# results might be owned by root because of docker
chown -R $USER:$USER results

# install the right version of milabench
export PATH="$PATH:$HOME/.local/bin"
export PYTHONPATH="$PYTHONPATH:$(python -m site --user-site)"
pip install --user git+https://github.com/mila-iqia/milabench@v0.0.8

# Download the AO config
wget https://gist.githubusercontent.com/Delaunay/ef0b2dc4aae5d42c16f00a17d17d490e/raw/acc5dabb908b3b9f918c9c5716711d4b82569da7/AO_476.yaml

# get the official score
MILABENCH_BASE="." milabench report --config AO_476.yaml --runs results/

