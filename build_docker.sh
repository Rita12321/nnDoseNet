#!/bin/bash
now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
. vars.txt

echo "Building docker image $IMAGE"

echo "
You are running $IMAGE
$DESCRIPTION.
Please contact $MAINTAINER with any questions.

Build date: $now
" > cecardenas_docker

docker build --rm --network=host -t $IMAGE --build-arg BUILDDATE=$now --build-arg MAINTAINER="$MAINTAINER" --build-arg DESCRIPTION="$DESCRIPTION" .
echo $IMAGE > pip_freeze.txt
echo $IMAGE > conda_list.txt
docker run --rm $IMAGE pip freeze >> pip_freeze.txt
docker run --rm $IMAGE conda list >> conda_list.txt

# rm cecardenas_docker
# rm -r auto-seg-engine
# rm -r mri-brain-auto-seg
