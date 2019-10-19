#!/bin/bash

for ver in 1.2.0 1.3.0 1.4.0 1.5.0 1.6.0 1.12.0 1.13.2 1.14.0; do
  dir="versions/$ver"
  mkdir -p "$dir"

  archive="$dir/tf.tar.gz"
  if [ ! -f "$archive" ]; then
    curl -o "$archive" "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-$ver.tar.gz"
  fi
  if [ ! -f "$dir/lib/libtensorflow.so" ]; then
    tar -C "$dir" -xvf "$archive"
  fi
done
