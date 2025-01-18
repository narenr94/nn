if [ ! -d "./MNIST" ]; then
  cd ../build/examples/
  mkdir ./MNIST
  cd ./MNIST
  wget "https://pjreddie.com/media/files/mnist_train.csv"
  wget "https://pjreddie.com/media/files/mnist_test.csv"
  cd ../
else
  echo "MNIST directory already exists!!!"
fi



