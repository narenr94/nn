#ifndef NN_BASE_ACC_H
#define NN_BASE_ACC_H
#include "nn_math.h"
#include "baseLossFunction.h"

class nn_layer;

class BaseAccelerator{

protected:

    nn_layer* m_pLayer;

public:

    BaseAccelerator(nn_layer* pLayer){}

    virtual void do_forwardpass_dense_layer() = 0;
    virtual void do_backwardpass_dense_layer_output_layer(float* pfExpOut, BaseLossFunction* lossFunc) = 0;
    virtual void do_backwardpass_dense_layer() = 0;

    /*todo:

        out_lyr->apply_act_func_all_nodes();
        m_pOptimizer->correct_weights_biases();

        //batch processing stuff
        
    */ 

   virtual ~BaseAccelerator(){}

};
#endif