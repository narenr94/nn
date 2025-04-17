#ifndef NN_CPU_ACC_H
#define NN_CPU_ACC_H
#include "baseAccelerator.h"


class CpuAccelerator : public BaseAccelerator{


public:

    CpuAccelerator(nn_layer* pNLyr);

    void do_forwardpass_dense_layer() override;
    void do_backwardpass_dense_layer_output_layer(float* pfExpOut, BaseLossFunction* lossFunc) override;
    void do_backwardpass_dense_layer() override;

    /*todo:

        out_lyr->apply_act_func_all_nodes();
        m_pOptimizer->correct_weights_biases();

        //batch processing stuff
        
    */ 

   virtual ~CpuAccelerator();

};
#endif