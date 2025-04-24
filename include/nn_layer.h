#ifndef NN_LAYER
#define NN_LAYER

#include "baseActivationFunction.h"

#include "baseLossFunction.h"

class nn_l2l_weight_matrix; //forward declaration

class BaseAccelerator; //Forward Declaration

#define RAND_MIN_WEIGHT_BIAS 1

#define RAND_MAX_WEIGHT_BIAS 9

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

class nn_layer{

    //node stuff

    float* m_pfValues; //value of node
    float* m_pfBiases; //value of bias
    float* m_pfDeltas; //delta value of node , used for back propogation

    //end of node stuff

    uint m_unNumNodes; //total number of nodes

    //Activation Function
    eAct_func m_eActFunc; //activation function to be used
    BaseActivationFunction* m_pActFunc;
    float m_actParam1;

    nn_l2l_weight_matrix* m_pWtMtx;

    nn_layer* m_pPrevLyr;

    nn_layer* m_pNextLyr;

    bool m_bPrevNxtLyrsSet;

    BaseAccelerator* m_pAccelerator;
    
    public:

    /*
        nn_layer() : constructor for layer
        @n_nodes : number of nodes in layer
        @prevLyr : pointer to previous layer, nullptr for input layer
    */
    nn_layer(uint n_nodes, eAct_func eActFunc, float actParam1);

    /*
        ~nn_layer() : destruct and frees layer resources
    */
    ~nn_layer();

    eAct_func get_act_func();

    float get_act_param();

    void apply_act_func_all_nodes();
    
    void get_delta_all_nodes(float * fVal);

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
    bool set_all_node_values(float* value);

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

    nn_l2l_weight_matrix* GetWeightMatrix();

    void SetPreviousNextLayers(nn_layer* prevLyr, nn_layer* nxtLyr);

    nn_layer* GetPreviousLayer();

    nn_layer* GetNextLayer();

    void do_forwardpass_to_current_layer();

    void do_backwardpass_to_previous_layer();

    void do_backwardpass_to_previous_layer_output_layer(float* fExpOut, BaseLossFunction* lossFunc);

};

#endif