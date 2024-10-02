#ifndef NN_L2L_WT_MTX
#define NN_L2L_WT_MTX

#include "nn_layer.h"

class nn_l2l_weight_matrix{
    
    nn_layer* m_pInputLayer; //input layer for matrix
    nn_layer* m_pOutputLayer; //output layer for matrix

    uint m_unSize; //size of matrix

    float* m_pfWeightMatrix; //starting address from where weights of matrix are stored

    bool m_bInitialized; //is weight matrix initialized?

    
    public:

    /*
        nn_l2l_weight_matrix() : constructor of layer to layer weight matrix
        @in : pointer to input/left layer
        @ot : pointer to output/right layer
    */
    nn_l2l_weight_matrix(nn_layer* in, nn_layer* ot);

    /*
        ~nn_l2l_weight_matrix() : destructs and frees resources of layer to layer weight matrix
    */
    ~nn_l2l_weight_matrix();

    /*
        set_weight() : sets weight of connection between particulat input and output node

        @in_idx : input node index
        @out_idx : output node index
        @wt : value of weight
    */
    bool set_weight(uint in_idx, uint out_idx, float wt);

    /*
        set_all_weight() : sets all weight in matrix

        @wt : array of values of weights
    */
    bool set_all_weight(float* wt);

    /*
        get_weight() : gets weight of connection between particulat input and output node

        @in_idx : input node index
        @out_idx : output node index
    */
    float get_weight(uint in_idx, uint out_idx);

    /*
        get_size() : gets size of matrix
    */
    uint get_size();

    /*
        populateWeightsWithRandomNumbers() : assign random numbers to weights
    */
    void populateWeightsWithRandomNumbers();


};

#endif