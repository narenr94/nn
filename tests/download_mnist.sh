#!/bin/bash
if [ -d "./MNIST" ]; then
  echo "MNIST directory already exists!!!"
elif [ -d "../../../MNIST" ]; then
  echo "local MNIST found ... copying ..."
  cp -r ../../../MNIST ./
else
  cd ../build/examples/
  mkdir ./MNIST
  cd ./MNIST
  wget "https://pjreddie.com/media/files/mnist_train.csv"
  wget "https://pjreddie.com/media/files/mnist_test.csv"
  cd ../
fi

