#ifndef ConvLayer_H
#define ConvLayer_H

#include "baseLayer.h"


class ConvLayer : public BaseLayer{

     
    public:

    /*
        ConvLayer() : constructor for layer
        @n_nodes : number of nodes in layer
        @prevLyr : pointer to previous layer, nullptr for input layer
    */
    ConvLayer(sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1);

    /*
        ~ConvLayer() : destruct and frees layer resources
    */
    ~ConvLayer();

    
    void SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr) override;

    void do_forwardpass_to_current_layer() override;

    void set_transform_matrix_parameter(uint in_idx, uint out_idx, float wt) override;

    void set_transform_matrix_parameter(uint Idx, float fWt) override;

    void set_all_transform_matrix_parameter(float* wt) override;
    

};

#endif