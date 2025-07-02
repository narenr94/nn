#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "nn_math.h"
#include "baseActivationFunction.h"
#include "baseLossFunction.h"

#define RAND_MIN_PARAMETER_BIAS 1

#define RAND_MAX_PARAMETER_BIAS 9

/*
    list of activation functions
    Note: keep TANH at last to keep test scripts intact
*/
enum eAct_func{
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SOFTMAX,
    TANH    
};

/*
    list of layer type
    Note: keep DENSE at last to keep test scripts intact
*/
enum eLayer_type{
    CONV,
    DENSE
};

struct sLayer_Dimensions{

    uint unInputRows;
    uint unInputColumns;
    uint unOutputRows;
    uint unOutputColumns;
    uint unTransformParametersRows;
    uint unTransformParametersColumns;
    uint unNoTransformParameterMtx;

    sLayer_Dimensions(uint in_rows, uint in_columns, uint out_rows, uint out_cols, uint trans_rows, uint trans_cols, uint no_trans_mtx)
    {
        unInputRows = in_rows;
        unInputColumns = in_columns;
        unOutputRows = out_rows;
        unOutputColumns = out_cols;
        unTransformParametersRows = trans_rows;
        unTransformParametersColumns = trans_cols;
        unNoTransformParameterMtx = no_trans_mtx;
    }

};

class BaseAccelerator; //Forward Declaration

class BaseLayer{

protected:
    //node stuff

    float* m_pfValues; //value of node
    float* m_pfBiases; //value of bias
    float* m_pfDeltas; //delta value of node , used for back propogation


    //dimentions
    sLayer_Dimensions m_Dimensions;
 
    //end of node stuff

    uint m_unNumNodes; //total number of nodes

    //Activation Function
    eAct_func m_eActFunc; //activation function to be used
    BaseActivationFunction* m_pActFunc;
    float m_actParam1;

    float* m_pfTransformParameters;

    BaseAccelerator* m_pAccelerator;

    BaseLayer* m_pPrevLyr;

    BaseLayer* m_pNextLyr;

    bool m_bPrevNxtLyrsSet;

    uint m_unTransformMatrixSize;

    eLayer_type m_layer_type;


public:

    BaseLayer(uint unNumNodesnodes, eAct_func eActFunc, float actParam1):
    m_pNextLyr(nullptr),
    m_pPrevLyr(nullptr),
    m_pfTransformParameters(nullptr),
    m_bPrevNxtLyrsSet(false),
    m_pfValues(nullptr),
    m_pfBiases(nullptr),
    m_pfDeltas(nullptr),
    m_Dimensions(0,0,0,0,0,0,0){}

    virtual ~BaseLayer(){}

    //---------- Layer APIs-------------

    virtual eAct_func get_act_func() = 0;

    virtual float get_act_param() = 0;

    virtual void apply_act_func_all_nodes() = 0;
    
    virtual void get_delta_all_nodes(float * fVal) = 0;

    virtual eLayer_type get_layer_type() = 0;

    sLayer_Dimensions get_layer_dimensions();

    /*
        get_num_nodes() : returns total number of nodes in layer
    */
    virtual uint get_num_nodes() = 0;

    /*
        set_node_value() : sets value of node in particular index

        @value : value to be set in node
        @index : the index represnting the node where value is to be set
    */
    virtual bool set_node_value(float value, uint index) = 0;

    /*
        set_all_node_values() : sets value of all nodes in layer

        @value : array containing values to be added
        
    */
    virtual bool set_all_node_values(float* value) = 0;

    /*
        set_all_node_biases() : sets value of all nodes in layer

        @bias : array containing bias to be added
        
    */
    virtual bool set_all_node_biases(float* bias) = 0;

    /*
        set_node_bias() : sets bias of node in particular index

        @bias : bias to be set in node
        @index : the index represnting the node where bias is to be set
    */
    virtual bool set_node_bias(float bias, uint index) = 0;

    /*
        set_node_delta() : sets delta of node in particular index

        @delta : delta to be set in node
        @index : the index represnting the node where bias is to be set
    */
    virtual bool set_node_delta(float delta, uint index) = 0;

    /*
        get_node_value_idx() : returns value of node in specified index

        @idx : the index represnting the node from where value is to be got
    */
    virtual float get_node_value_idx(uint idx) = 0;
    /*
        get_node_bias_idx() : returns bias of node in specified index

        @idx : the index represnting the node from where bias is to be got
    */
    virtual float get_node_bias_idx(uint idx) = 0;

    /*
        get_node_delta_idx() : returns delta of node in specified index

        @idx : the index represnting the node from where delta is to be got
    */
    virtual float get_node_delta_idx(uint idx) = 0;
    
    /*
        populateBiasesWithRandomNumbers() : assign random numbers to biases
    */
    virtual void populateBiasesWithRandomNumbers() = 0;

    virtual const float* get_transform_matrix() = 0;

    virtual void SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr) = 0;

    virtual BaseLayer* GetPreviousLayer() = 0;

    virtual BaseLayer* GetNextLayer() = 0;

    virtual void do_forwardpass_to_current_layer() = 0;

    void do_backwardpass_to_previous_layer();

    virtual void do_backwardpass_to_previous_layer_output_layer(float* fExpOut, BaseLossFunction* lossFunc) = 0;

    //---------- End Layer APIs-------------

    //---------- Transform Parameters APIs------------

    virtual void set_transform_matrix_parameter(uint in_idx, uint out_idx, float wt) = 0;

    virtual void set_transform_matrix_parameter(uint Idx, float fWt) = 0;

    virtual void set_all_transform_matrix_parameter(float* wt) = 0;

    virtual float get_transform_matrix_parameter(uint Idx) = 0;

    /*
        get_transform_matrix_parameter_size() : gets size of matrix
    */
    virtual uint get_transform_matrix_parameter_size() = 0;

    virtual void populate_transform_matrix_parameter_with_random_numbers() = 0;

    //---------- Transform Parameters APIs------------

};


#endif