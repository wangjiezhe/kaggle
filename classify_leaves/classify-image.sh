#!/bin/bash

DATA_DIR="./data"
TRAIN_DIR="${DATA_DIR}/images_train"

while IFS="," read -r image label; do
  if [ -f "${DATA_DIR}/${image}" ]; then
    mkdir -p "${TRAIN_DIR}/${label}"
    mv "${DATA_DIR}/${image}" "${TRAIN_DIR}/${label}"
    echo "Move ${DATA_DIR}/${image} to ${TRAIN_DIR}/${label}"
  else
    echo "Warning: ${DATA_DIR}/${image} not found."
  fi
done < <(tail -n +2 "${DATA_DIR}/train.csv")
