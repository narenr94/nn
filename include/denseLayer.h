#ifndef DenseLayer_H
#define DenseLayer_H

#include "baseLayer.h"


class DenseLayer : public BaseLayer{

    

    
    public:

    /*
        DenseLayer() : constructor for layer
        @n_nodes : number of nodes in layer
        @prevLyr : pointer to previous layer, nullptr for input layer
    */
    DenseLayer(sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1);

    /*
        ~DenseLayer() : destruct and frees layer resources
    */
    ~DenseLayer();

    void SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr) override;

    void do_forwardpass_to_current_layer() override;

    void set_transform_matrix_parameter(uint in_idx, uint out_idx, float wt) override;

    void set_transform_matrix_parameter(uint Idx, float fWt) override;

    void set_all_transform_matrix_parameter(float* wt) override;
    

};

#endif