#ifndef NN_LAYER
#define NN_LAYER

#include "nn_node.h"

/*
    list of layer types
*/
enum eLyr_type{
    INPUT_LYR,
    HIDDEN_LYR,
    OUTPUT_LYR
};

#define RAND_MIN_WEIGHT_BIAS 1

#define RAND_MAX_WEIGHT_BIAS 9

class nn_layer{

    uint m_unNumNodes; //total number of nodes
    nn_node** m_ppNodes; //starting address from nodes can be accessed
    bool m_bInitialized; //is layer initialized?
    eLyr_type eLyrType; //layer type


    
    public:

    /*
        nn_layer() : constructor for layer
        @n_nodes : number of nodes in layer
    */
    nn_layer(uint n_nodes);

    /*
        ~nn_layer() : destruct and frees layer resources
    */
    ~nn_layer();

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
    bool set_node_delta(float bias, uint index);

    /*
        is_initialized() : returns if layer is initialized or not
    */
    bool is_initialized();

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
        set_layer_type() : sets layer type

        @lt : layer type to be set
    */
    void set_layer_type(eLyr_type lt);

    /*
        get_layer_type() : gets layer's type
    */
    eLyr_type get_layer_type();
    
    /*
        populateBiasesWithRandomNumbers() : assign random numbers to biases
    */
    void populateBiasesWithRandomNumbers();


};

#endif