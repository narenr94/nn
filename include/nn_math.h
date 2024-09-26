#ifndef NN_MATH

#include <stdlib.h>

#define NN_MATH

typedef unsigned int uint;

#define EULER_NUMBER_F 2.71828182846

float get_sigmoidf(float fValue);

float find_derivative_sigmoidf(float fValue);

uint getRandomNumber(uint uMin, uint uMax);

#endif
