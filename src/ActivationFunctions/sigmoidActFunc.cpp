#include "sigmoidActFunc.h"

float SigmoidActFunc::apply_act_func(float n)
{
    return get_sigmoidf(n);
}

float SigmoidActFunc::apply_act_func_derv(float fVal)
{
    return find_derivative_sigmoidf(fVal);
}