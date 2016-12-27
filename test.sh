#!/bin/bash
/home/pkrush/caffe/.build_release/examples/cpp_classification/classification.bin \
/home/pkrush/lmdb-files/train/13294/deploy.prototxt \
/home/pkrush/lmdb-files/train/13294/snapshot_iter_844.caffemodel \
/home/pkrush/lmdb-files/train/13294/mean.binaryproto \
/home/pkrush/lmdb-files/train/13294/labels.txt \
/home/pkrush/lmdb-files/test/0/train_db/data.mdb
