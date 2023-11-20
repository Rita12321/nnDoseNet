#!/bin/bash
. vars.txt

echo "Building Singularity image $IMAGE"

docker push $IMAGE
#singularity pull $SIF_FILE_NAME docker://$IMAGE
