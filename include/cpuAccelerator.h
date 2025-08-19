#ifndef NN_CPU_ACC_H
#define NN_CPU_ACC_H
#include "baseAccelerator.h"


class CpuAccelerator : public BaseAccelerator{


public:

    CpuAccelerator(BaseLayer* pNLyr);

    void do_forwardpass_dense_layer() override;
    void do_backwardpass_from_output_layer(float* pfExpOut, BaseLossFunction* lossFunc) override;
    void do_backwardpass_dense_layer() override;

    void do_forwardpass_conv_layer(sLayer_Dimensions t_tims) override;
    void do_backwardpass_conv_layer(sLayer_Dimensions t_tims) override;

    void do_forwardpass_pooling_layer(ePooling_type t_pooling_type, uint t_stride) override;

    virtual ~CpuAccelerator();

private:

    uint find_max_in_window_at(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx);

    float find_avg_in_window_at(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx);

};
#endif