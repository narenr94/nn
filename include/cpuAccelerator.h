#ifndef NN_CPU_ACC_H
#define NN_CPU_ACC_H
#include "baseAccelerator.h"

class NeuralNet;

class CpuAccelerator : public BaseAccelerator{

protected:

    NeuralNet* m_pNN;

public:

    CpuAccelerator(NeuralNet* pNN);

    void do_forwardpass_to_next_layer(uint unInLayerIdx) override;
    void find_delta_of_all_nodes(float* pfExpOut) override;

    /*todo:

        out_lyr->apply_act_func_all_nodes();
        m_pOptimizer->correct_weights_biases();

        //batch processing stuff
        
    */ 

   virtual ~CpuAccelerator();

};
#endif