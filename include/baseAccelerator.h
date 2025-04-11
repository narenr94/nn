#ifndef NN_BASE_ACC_H
#define NN_BASE_ACC_H
#include "nn_math.h"

class NeuralNet;

class BaseAccelerator{

protected:

    NeuralNet* m_pNN;

public:

    BaseAccelerator(){}

    virtual void do_forwardpass_to_current_layer(uint unInLayerIdx) = 0;
    virtual void find_delta_of_current_layer_nodes(float* pfExpOut, uint idx) = 0;

    /*todo:

        out_lyr->apply_act_func_all_nodes();
        m_pOptimizer->correct_weights_biases();

        //batch processing stuff
        
    */ 

   virtual ~BaseAccelerator(){}

};
#endif