#ifndef NN_MATH
#define NN_MATH

#include <stdlib.h>
#include <cmath>

typedef unsigned int uint;

#define EULER_NUMBER_F 2.71828182846

float get_sigmoidf(float fValue);

float find_derivative_sigmoidf(float fValue);

uint getRandomNumber(uint uMin, uint uMax);

float get_reluf(float fValue);

float find_derivative_reluf(float fValue);

float get_leakyReluf(float fValue, float alpha);

float find_derivative_leakyReluf(float fValue, float alpha);

float get_tanhf(float fValue);

float find_derivative_tanhf(float fValue);

float get_softmaxf(float fValue, float fMax);

float find_derivative_softmaxf(float fValue);

float find_derivative_softmaxf_wrong_pred(float fValue, float fValCorr);

#endif
