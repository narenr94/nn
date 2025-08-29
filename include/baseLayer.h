#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "nn_math.h"
#include "baseActivationFunction.h"
#include "baseLossFunction.h"

#define RAND_MIN_PARAMETER_BIAS 1

#define RAND_MAX_PARAMETER_BIAS 9

#include "nn_defines.h"

class BaseAccelerator; //Forward Declaration

class BaseLayer{

protected:
    //node stuff

    float* m_pfValues = nullptr; //value of node
    float* m_pfBiases = nullptr; //value of bias
    float* m_pfDeltas = nullptr; //delta value of node , used for back propogation


    //dimentions
    sLayer_Dimensions m_Dimensions;
 
    //end of node stuff

    uint m_unNumNodes; //total number of nodes

    //Activation Function
    eAct_func m_eActFunc; //activation function to be used
    BaseActivationFunction* m_pActFunc;
    float m_actParam1;

    float* m_pfTransformParameters = nullptr;

    BaseAccelerator* m_pAccelerator;

    BaseLayer* m_pPrevLyr = nullptr;

    BaseLayer* m_pNextLyr = nullptr;

    bool m_bPrevNxtLyrsSet;

    uint m_unTransformMatrixSize = 0;

    eLayer_type m_layer_type;


public:

    BaseLayer(eLayer_type t_layer_type, sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1);

    BaseLayer(eLayer_type t_layer_type, std::string load_data);

    virtual ~BaseLayer();

    //---------- Layer APIs-------------

    eAct_func get_act_func();

    float get_act_param();

    void apply_act_func_all_nodes();
    
    void get_delta_all_nodes(std::vector<float>& fVal);

    eLayer_type get_layer_type();

    sLayer_Dimensions get_layer_dimensions();

    /*
        get_num_nodes() : returns total number of nodes in layer
    */
    uint get_num_nodes();

    /*
        set_node_value() : sets value of node in particular index

        @value : value to be set in node
        @index : the index represnting the node where value is to be set
    */
    bool set_node_value(float value, uint index);

    /*
        set_all_node_values() : sets value of all nodes in layer

        @value : array containing values to be added
        
    */
    bool set_all_node_values(std::vector<float>& value);

    /*
        set_all_node_biases() : sets value of all nodes in layer

        @bias : array containing bias to be added
        
    */
    bool set_all_node_biases(float* bias);

    /*
        set_node_bias() : sets bias of node in particular index

        @bias : bias to be set in node
        @index : the index represnting the node where bias is to be set
    */
    bool set_node_bias(float bias, uint index);

    /*
        set_node_delta() : sets delta of node in particular index

        @delta : delta to be set in node
        @index : the index represnting the node where bias is to be set
    */
    bool set_node_delta(float delta, uint index);

    /*
        get_node_value_idx() : returns value of node in specified index

        @idx : the index represnting the node from where value is to be got
    */
    float get_node_value_idx(uint idx);
    /*
        get_node_bias_idx() : returns bias of node in specified index

        @idx : the index represnting the node from where bias is to be got
    */
    float get_node_bias_idx(uint idx);

    /*
        get_node_delta_idx() : returns delta of node in specified index

        @idx : the index represnting the node from where delta is to be got
    */
    float get_node_delta_idx(uint idx);
    
    /*
        populateBiasesWithRandomNumbers() : assign random numbers to biases
    */
    void populateBiasesWithRandomNumbers();

    const float* get_transform_matrix();

    BaseLayer* GetPreviousLayer();

    BaseLayer* GetNextLayer();    

    void do_backwardpass_to_previous_layer();

    void do_backwardpass_to_previous_layer_output_layer(std::vector<float>& fExpOut, BaseLossFunction* lossFunc);

    float get_transform_matrix_parameter(uint Idx);

    uint get_transform_matrix_parameter_size();

    void populate_transform_matrix_parameter_with_random_numbers();

    std::string get_serialized_general_layer_data();

    std::string get_serialized_biases_data();

    std::string get_serialized_transform_mtx_data();

    void SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr);

    sLayer_Parsed_Dim get_prev_layer_parsed_output_dims();

    //---------- End Layer APIs-------------

    //---------- Pure Virtual APIs------------

    virtual void set_transform_matrix_parameter(uint in_idx, uint out_idx, float wt) = 0;

    virtual void set_transform_matrix_parameter(uint Idx, float fWt) = 0;

    virtual void set_all_transform_matrix_parameter(float* wt) = 0;
    
    virtual void do_forwardpass_to_current_layer() = 0;

    virtual std::string get_serialized_save_data() = 0;
    

    // virtual void deserialize_set_load_data(std::string data) = 0;

    //---------- Pure Virtual APIs------------

    private:
    
    void apply_parsed_load_data(eLayer_type t_layer_type, std::vector<std::pair<std::string, std::vector<std::string>>>& parsed_lines);

    void setup_layer(eLayer_type t_layer_type, sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1);

    void extract_apply_biases(std::vector<std::string>& biases_string);

    void extract_apply_transform_param(std::vector<std::string>& wt_string);

    void setup_accelerator();

    void setup_activation_function(eAct_func eActFunc);
};


#endif