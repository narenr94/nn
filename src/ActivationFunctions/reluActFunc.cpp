#include "reluActFunc.h"

float ReluActFunc::apply_act_func(float n)
{
    return get_reluf(n);
}

float ReluActFunc::apply_act_func_derv(float fVal)
{
    return find_derivative_reluf(fVal);
}