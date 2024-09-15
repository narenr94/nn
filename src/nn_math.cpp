#include <math.h>
#include "nn_math.h"

float sigmoidf(float n) {
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

float derivative_sigmoidf(float n)
{
    return (n * (1 - n));
}

uint getRandomNumber(uint min, uint max) 
{
    return rand() % (max - min + 1) + min;
}