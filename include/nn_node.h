#ifndef NN_NODE

#define NN_NODE

#include "nn_logger.h"

class nn_node{

    float m_fValue; //value of node
    float m_fBias; //value of bias

    float m_fDelta; //delta value of node , used for back propogation

    
    

    public:

    /*
        nn_node : constructor, sets value and bias to 0.0
    */
    nn_node();

    /*
        set_value : Sets value of node
        @n : value to be set
    */
    void set_value(float n);
    /*
        set_bias : Sets bias of node
        @n : value of bias to be set
    */
    void set_bias(float n);
    /*
        set_delta : Sets delta of node
        @n : value of delta to be set
    */
    void set_delta(float n);
    /*
        get_delta() : returns current delta of node
    */
    float get_delta();
    /*
        get_value() : returns current value of node
    */
    float get_value();
    /*
        get_bias() : returns current bias value of node
    */
    float get_bias();

};

#endif