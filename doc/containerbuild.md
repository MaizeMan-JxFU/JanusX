# Use gtools in container

## docker

```bash
# build from dockerfile
docker build -t gtools:0.1.0 .
# test
docker run --rm gtools:0.1.0 gwas --vcf ./example/mouse_hs1940.vcf.gz --pheno ./example/mouse_hs1940.pheno --out test
```

## singularity

```bash
# From docker image to build sif in linux server
docker save gtools:0.1.0 -o gtools.tar
singularity build gtools.sif docker-archive://gtools.tar
```
