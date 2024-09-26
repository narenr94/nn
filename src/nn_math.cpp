#include <math.h>
#include "nn_math.h"

float get_sigmoidf(float fVal) {
    return (1 / (1 + powf(EULER_NUMBER_F, - fVal)));
}

float find_derivative_sigmoidf(float fVal)
{
    return (fVal * (1 - fVal));
}

uint getRandomNumber(uint unMin, uint unMax) 
{
    return rand() % (unMax - unMin + 1) + unMin;
}