#include "tanhActFunc.h"

float TanhActFunc::apply_act_func(float n)
{
    return get_tanhf(n);
}

float TanhActFunc::apply_act_func_derv(float fVal)
{
    return find_derivative_tanhf(fVal);
}