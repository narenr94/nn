if [ ! -d "../build/examples/MNIST" ]; then
  cd ../build/examples/
  mkdir ./MNIST
  cd ./MNIST
  wget "https://pjreddie.com/media/files/mnist_train.csv"
  wget "https://pjreddie.com/media/files/mnist_test.csv"
  cd ../
fi



