#ifndef NN_BASE_ACC_H
#define NN_BASE_ACC_H
#include "nn_math.h"

class NeuralNet;

class BaseAccelerator{

protected:

    NeuralNet* m_pNN;

public:

    BaseAccelerator(NeuralNet* pNN);

    virtual void do_forwardpass_to_next_layer(uint unInLayerIdx);
    virtual void find_delta_of_all_nodes(float* pfExpOut);

    /*todo:

        out_lyr->apply_act_func_all_nodes();
        m_pOptimizer->correct_weights_biases();

        //batch processing stuff
        
    */ 

   virtual ~BaseAccelerator();

};
#endif