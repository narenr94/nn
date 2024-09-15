#ifndef NN_MATH

#include <stdlib.h>

#define NN_MATH

using namespace std;

typedef unsigned int uint;

#define EULER_NUMBER_F 2.71828182846

float sigmoidf(float n);

float derivative_sigmoidf(float n);

uint getRandomNumber(uint min, uint max);

#endif
