if [ ! -d "./MNIST" ]; then
  mkdir ./MNIST
  cd ./MNIST
  wget "https://pjreddie.com/media/files/mnist_train.csv"
  wget "https://pjreddie.com/media/files/mnist_test.csv"
fi

rm ./*.dmp
rm ./*.log
rm ./*.o
g++ -Wall -c ../src/*.cpp ./mnist_example.c -I ../include/ -g
g++ -Wall ./*.o -o ./mnist_example -g
./mnist_example

