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

float get_reluf(float fValue)
{
    if(fValue > 0.0f)
    {
        return fValue;
    }

    return 0.0f;
}

float find_derivative_reluf(float fValue)
{
    if(fValue > 0.0f)
    {
        return 1.0f;
    }
    return 0.0f;

}

float get_leakyReluf(float fValue, float alpha)
{
    if(fValue > 0.0f)
    {
        return fValue;
    }
    return (fValue * alpha);
}

float find_derivative_leakyReluf(float fValue, float alpha)
{
    if(fValue > 0.0f)
    {
        return 1.0f;
    }
    return alpha;
}

float get_tanhf(float fValue)
{
    return tanh(fValue);
}

float find_derivative_tanhf(float fValue)
{
    return (1 - (fValue * fValue));
}