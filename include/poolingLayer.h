#ifndef PoolingLayer_H
#define PoolingLayer_H

#include "baseLayer.h"
#include "nn_defines.h"


class PoolingLayer : public BaseLayer{


    ePooling_type m_pooling_type = ePooling_type::AVERAGE; 
    uint m_stride = 2;
    ePoolingKernelSize m_pooling_kernel_size = ePoolingKernelSize::Sz2x2;
    
    public:

    /*
        PoolingLayer() : constructor for layer
        @n_nodes : number of nodes in layer
        @prevLyr : pointer to previous layer, nullptr for input layer
    */
    PoolingLayer(sLayer_Dimensions t_dims, ePooling_type t_pooling_type, ePoolingKernelSize t_kernel_size, uint t_stride);

    PoolingLayer(std::string load_data);

    /*
        ~PoolingLayer() : destruct and frees layer resources
    */
    ~PoolingLayer();

    void do_forwardpass_to_current_layer() override;

    void set_transform_matrix_parameter(uint in_idx, uint out_idx, float wt) override;

    void set_transform_matrix_parameter(uint Idx, float fWt) override;

    void set_all_transform_matrix_parameter(float* wt) override;

    std::string get_serialized_save_data() override;

    std::pair<uint,uint> PoolingLayer::get_kernel_rows_cols();

};

#endif