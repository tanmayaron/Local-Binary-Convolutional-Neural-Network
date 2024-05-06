//`include "neuron.sv"

module neuron1 #(parameter NUM_INPUTS=8)
(
    input signed [15:0] inputs [0:NUM_INPUTS-1], 
    input signed [15:0] weights [0:NUM_INPUTS-1], 
    input clk,
    input rst,
    output reg signed [15:0] opt 

);

// Internal variables
reg signed [15:0] sum=0; // Accumulator for weighted sum
reg [3:0] i=0;

always @(posedge clk) begin
    if(rst==0) begin
        sum = 0; // Initialize the sum to zero
        i=0;
    end
    else begin
        if(i<NUM_INPUTS) begin
            sum = sum + (inputs[i] * weights[i]);
            i = i + 1;
        end
        else begin

            if (sum[15] == 1'b1)
                opt = 8'd0;
            else
                opt = sum; 
            //i<=0;
        end
    end
    
end

endmodule




module fc_layer #(parameter IP_LAYER=8, NUM_INP=8)
(
    input signed [15:0] inputs [0:(IP_LAYER * NUM_INP)-1],   // Inputs to the layer (from previous layer)
    input signed [15:0] weights [0:(IP_LAYER * NUM_INP)-1],  // Weights for each neuron
    input signed [15:0] weights_dense [0:7],  // Weights for each neuron

    input signed [15:0] dense_wt [0:1],
    input signed [15:0] bias,             // Bias for each neuron
    output reg signed [15:0] ot       // Output from the fully connected layer
);


    // Internal signals
    reg signed [15:0] neuron_outputs [0:IP_LAYER-1];
    reg signed [15:0] dense_op[0:1];

    // Instantiate neurons
    //INPUT LAYER
    genvar i;
    generate
        for (i = 0; i < IP_LAYER; i = i + 1) begin : neuron_inst
            neuron1  #(.NUM_INPUTS(NUM_INP) ) ip_layer
            (
                .inputs(inputs[(NUM_INP*i) +: NUM_INP]),
                .weights(weights[(NUM_INP*i) +: NUM_INP]),
                .clk(clk),
                .rst(rst),
                .opt(neuron_outputs[i])
            );
        end
    endgenerate

    //DENSE LAYER-1

            neuron1  #(.NUM_INPUTS(NUM_INP) ) inst_neuron_dense_1
            (
                .inputs(neuron_outputs),
                .weights(weights_dense),
                .clk(clk),
                .rst(rst),
                .opt(dense_op[0])
            );
            neuron1  #(.NUM_INPUTS(NUM_INP) ) inst_neuron_dense_2
            (
                .inputs(neuron_outputs),
                .weights(weights_dense),
                .clk(clk),
                .rst(rst),
                .opt(dense_op[1])
            );


    //OUTPUT NEURON
    neuron1 #(.NUM_INPUTS(2)) dense_layer
    (
     .inputs(dense_op) ,
     .weights(dense_wt) ,
     .clk(clk),
     .rst(rst),
     .opt(ot)
    );

    

endmodule
