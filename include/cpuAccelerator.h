#ifndef NN_CPU_ACC_H
#define NN_CPU_ACC_H
#include "baseAccelerator.h"


class CpuAccelerator : public BaseAccelerator{


public:

    CpuAccelerator(BaseLayer* pNLyr);

    void do_forwardpass_dense_layer() override;
    void do_backwardpass_from_output_layer(std::vector<float>& pfExpOut, BaseLossFunction* lossFunc) override;
    void do_backwardpass_dense_layer() override;

    void do_forwardpass_conv_layer() override;
    void do_backwardpass_conv_layer() override;

    void do_forwardpass_pooling_layer(ePooling_type t_pooling_type) override;
    void do_backwardpass_pooling_layer(ePooling_type t_pooling_type) override;

    virtual ~CpuAccelerator();

private:

    float find_conv_at_window_all_channels(std::pair<uint, uint> row_col_pos, sLayer_Parsed_Dim& t_prev_parsed_dims, uint t_kernel_num);

    float find_conv_at_window_one_channels(uint mtx_num, std::pair<uint, uint> row_col_pos, sLayer_Parsed_Dim& t_prev_parsed_dims, uint t_kernel_num);

    uint find_max_in_window_at(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx);

    float find_avg_in_window_at(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx);

    std::vector<uint> find_all_elements_idx_in_window(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx);

    uint find_max_element_idx_in_window(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx);

    void reset_layer_deltas();


};
#endif