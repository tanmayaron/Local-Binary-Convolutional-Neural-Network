`timescale 1ns / 1ps

module fc_layer_tb;

    // Parameters
    parameter IP_LAYER = 8;
    parameter NUM_INP = 8;

    // Inputs
    reg signed [15:0] inputs [0:(IP_LAYER * NUM_INP)-1];
    reg signed [15:0] weights [0:(IP_LAYER * NUM_INP)-1];
    reg signed [15:0] dense_wt [IP_LAYER];
    reg signed [15:0] bias;
    reg clk;
    reg rst;

    // Output
    wire signed [15:0] ot;

    // Instantiate the DUT (Design Under Test)
    fc_layer #(
        .IP_LAYER(IP_LAYER),
        .NUM_INP(NUM_INP)
    ) dut (
        .inputs(inputs),
        .weights(weights),
        .dense_wt(dense_wt),
        .bias(bias),
        .ot(ot)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Reset generation
    initial begin
        rst = 1;
        #10;
        rst = 0;
    end

    // Stimulus
    initial begin
        // Provide inputs
        for (int i = 0; i < (IP_LAYER * NUM_INP); i = i + 1)
            inputs[i] = $random % 100;
        for (int i = 0; i < (IP_LAYER * NUM_INP); i = i + 1)
            weights[i] = $random % 100;
        for (int i = 0; i < IP_LAYER; i = i + 1)
            dense_wt[i] = $random % 100;
        bias = $random % 256;

        // Wait for some time
        #500;

        // Display output
        $display("Output: %d", ot);
    end

endmodule
