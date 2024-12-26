#ifndef OPTIMIZER_H
#define OPTIMIZER_H


class Optimizer{

public:

    Optimizer(){}

    virtual float apply_act_func(float n);

    virtual float apply_act_func_derv(float fVal);

    virtual ~Optimizer(){}

};

#endif