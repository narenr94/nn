#include "leakyReluActFunc.h"

float LeakyReluActFunc::apply_act_func(float n)
{
    return get_leakyReluf(n, m_alpha);
}

float LeakyReluActFunc::apply_act_func_derv(float fVal)
{
    return find_derivative_leakyReluf(fVal, m_alpha);
}