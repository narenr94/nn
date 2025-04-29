#ifndef DenseLayer_H
#define DenseLayer_H

#include "baseLayer.h"

#define RAND_MIN_WEIGHT_BIAS 1

#define RAND_MAX_WEIGHT_BIAS 9


class DenseLayer : public BaseLayer{

    

    
    public:

    /*
        DenseLayer() : constructor for layer
        @n_nodes : number of nodes in layer
        @prevLyr : pointer to previous layer, nullptr for input layer
    */
    DenseLayer(uint n_nodes, eAct_func eActFunc, float actParam1);

    /*
        ~DenseLayer() : destruct and frees layer resources
    */
    ~DenseLayer();

    eAct_func get_act_func() override;

    float get_act_param() override;

    void apply_act_func_all_nodes() override;
    
    void get_delta_all_nodes(float * fVal) override;

    /*
        get_num_nodes() : returns total number of nodes in layer
    */
    uint get_num_nodes() override;

    /*
        set_node_value() : sets value of node in particular index

        @value : value to be set in node
        @index : the index represnting the node where value is to be set
    */
    bool set_node_value(float value, uint index) override;

    /*
        set_all_node_values() : sets value of all nodes in layer

        @value : array containing values to be added
        
    */
    bool set_all_node_values(float* value) override;

    /*
        set_all_node_biases() : sets value of all nodes in layer

        @bias : array containing bias to be added
        
    */
    bool set_all_node_biases(float* bias) override;

    /*
        set_node_bias() : sets bias of node in particular index

        @bias : bias to be set in node
        @index : the index represnting the node where bias is to be set
    */
    bool set_node_bias(float bias, uint index) override;

    /*
        set_node_delta() : sets delta of node in particular index

        @delta : delta to be set in node
        @index : the index represnting the node where bias is to be set
    */
    bool set_node_delta(float delta, uint index) override;

    /*
        get_node_value_idx() : returns value of node in specified index

        @idx : the index represnting the node from where value is to be got
    */
    float get_node_value_idx(uint idx) override;
    /*
        get_node_bias_idx() : returns bias of node in specified index

        @idx : the index represnting the node from where bias is to be got
    */
    float get_node_bias_idx(uint idx) override;

    /*
        get_node_delta_idx() : returns delta of node in specified index

        @idx : the index represnting the node from where delta is to be got
    */
    float get_node_delta_idx(uint idx) override;
    
    /*
        populateBiasesWithRandomNumbers() : assign random numbers to biases
    */
    void populateBiasesWithRandomNumbers() override;

    const float* get_transform_matrix() override;

    void SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr) override;

    BaseLayer* GetPreviousLayer() override;

    BaseLayer* GetNextLayer() override;

    void do_forwardpass_to_current_layer() override;

    void do_backwardpass_to_previous_layer() override;

    void do_backwardpass_to_previous_layer_output_layer(float* fExpOut, BaseLossFunction* lossFunc) override;

    /*
        set_weight() : sets weight of connection between particulat input and output node

        @in_idx : input node index
        @out_idx : output node index
        @wt : value of weight
    */
    void set_transform_matrix_parameter(uint in_idx, uint out_idx, float wt) override;

    void set_transform_matrix_parameter(uint Idx, float fWt) override;

    /*
        set_all_weight() : sets all weight in matrix

        @wt : array of values of weights
    */
    void set_all_transform_matrix_parameter(float* wt) override;

    /*
        get_weight() : gets weight of connection between particulat input and output node

        @in_idx : input node index
        @out_idx : output node index
    */
    float get_transform_matrix_parameter(uint in_idx, uint out_idx) override;

    float get_transform_matrix_parameter(uint Idx) override;

    /*
        get_transform_matrix_parameter_size() : gets size of matrix
    */
    uint get_transform_matrix_parameter_size() override;

    /*
        populate_transform_matrix_parameter_with_random_numbers() : assign random numbers to weights
    */
    void populate_transform_matrix_parameter_with_random_numbers() override;


};

#endif