
//1X1 convolution (inspired from Googlenet)
module bitmap_conv #(
    parameter NUM_CHANNELS = 3,
    parameter IMG_SIZE = 4
)(
    input signed [15:0] ifmap[IMG_SIZE-1:0][IMG_SIZE-1:0][NUM_CHANNELS-1:0],
    input signed [15:0] filter[NUM_CHANNELS-1:0], 
    input clk,
    output reg signed [15:0] ofmap[IMG_SIZE-1:0][IMG_SIZE-1:0] 
);

    genvar i, j,k;
    
    generate
        for (i = 0; i < IMG_SIZE; i = i + 1) begin
            for (j = 0; j < IMG_SIZE; j = j + 1) begin
                for (k = 0; k < NUM_CHANNELS; k = k + 1) begin
                    always @(posedge clk) begin
                        
                        if(k==0) begin
                            ofmap[i][j]=0;
                        end
 
                        ofmap[i][j] = ofmap[i][j] + (ifmap[i][j][k] * filter[k]); 

                        

                        

                    end
                end      
            end
        end
    endgenerate
endmodule



//Multi-channel ReLu
module relu_multi1 #(parameter NUM_INSTANCES=5, CHANNELS=1)
(
    input signed [15:0] ifmap [NUM_INSTANCES-1:0][NUM_INSTANCES-1:0][CHANNELS-1:0],
    input clk,
    output reg signed [15:0] ofmap [NUM_INSTANCES-1:0][NUM_INSTANCES-1:0][CHANNELS-1:0]
);

genvar i, j, k, l;

generate
    for (i = 0; i < NUM_INSTANCES; i = i + 1) begin
        for (j = 0; j < NUM_INSTANCES; j = j + 1) begin
                for (l = 0; l < CHANNELS; l = l + 1) begin
                    always @(posedge clk) begin
                        if (ifmap[i][j][l][15] == 1'b0) begin
                            ofmap[i][j][l] <= ifmap[i][j][l];
                        end
                        else begin
                            ofmap[i][j][l] <= 8'd0;
                        end
                    end
                
            end
        end
    end
endgenerate

endmodule


//Multi-channel Ternary convolution
//Replacing MAC operation with adders and subtractors
module ternary_conv #(parameter IMG_SIZE = 5, KER_SIZE = 2, NUM_CHANNELS=2)
(
    input signed [15:0] ifmap[IMG_SIZE-1:0][IMG_SIZE-1:0], 
    input signed [15:0] filter[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0],  
    input clk,
    output reg signed [15:0] ofmap[IMG_SIZE-KER_SIZE:0][IMG_SIZE-KER_SIZE:0][NUM_CHANNELS-1:0]
);



integer i, j, k, idx_j,res,k_idx,ifmap_val,l_idx;


generate
    for (genvar idx_l = 0; idx_l < NUM_CHANNELS; idx_l = idx_l + 1) begin
        for (genvar idx_i = 0; idx_i < (IMG_SIZE-KER_SIZE+1)*(IMG_SIZE-KER_SIZE+1); idx_i = idx_i + 1) begin
            for (genvar idx_k = 0; idx_k < (KER_SIZE)*(KER_SIZE); idx_k = idx_k + 1) begin
                always@(posedge clk) begin
                    
                    if(idx_k==0)begin
                        res=0;
                    end


                    k_idx = idx_k / KER_SIZE;
                    l_idx = idx_k % KER_SIZE;
                    i = idx_i / (IMG_SIZE - KER_SIZE + 1);
                    j = idx_i % (IMG_SIZE - KER_SIZE + 1);
                    idx_j = (i+k_idx)*(IMG_SIZE) + (j+l_idx);
                    ifmap_val = ifmap[idx_j/IMG_SIZE][idx_j%IMG_SIZE];
                    if (filter[k_idx][l_idx][idx_l] == 1) begin
                        res = res + ifmap_val;
                    end
                    else if (filter[k_idx][l_idx][idx_l] == -1) begin
                        res = res - ifmap_val;
                    end



                    if(idx_k==((KER_SIZE)*(KER_SIZE)-1))begin                       
                        ofmap[i][j][idx_l] = res;
                    end
                end
            end
        end
    end
endgenerate
endmodule


//Per layer operations
//Each layer performs Ternary convolution, ReLu, and then 1X1 convolution to output a single channel image
//Input image is a single channel image which is split into multiple channels through ternary convolution, and then combined to form single channel output
module layer #(parameter IMG_SIZE=5, NUM_CHANNELS=1, KER_SIZE=3)
(
    input signed [15:0] ifmap[IMG_SIZE-1:0][IMG_SIZE-1:0],
    input signed [15:0] filter1[NUM_CHANNELS-1:0],
    input signed [15:0] filter[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0],
    input clk,
    //output reg signed [15:0] ternary_conv_output[IMG_SIZE-KER_SIZE:0][IMG_SIZE-KER_SIZE:0][NUM_CHANNELS-1:0],
    //output reg signed [15:0] relu_output[IMG_SIZE-KER_SIZE:0][IMG_SIZE-KER_SIZE:0][NUM_CHANNELS-1:0],
    output reg signed [15:0] ofmap[IMG_SIZE-KER_SIZE:0][IMG_SIZE-KER_SIZE:0]
);

reg signed [15:0] ternary_conv_output[IMG_SIZE-KER_SIZE:0][IMG_SIZE-KER_SIZE:0][NUM_CHANNELS-1:0];
reg signed [15:0] relu_output[IMG_SIZE-KER_SIZE:0][IMG_SIZE-KER_SIZE:0][NUM_CHANNELS-1:0];


// Instantiate conv2_final_multi module
ternary_conv #(
    .IMG_SIZE(IMG_SIZE),
    .KER_SIZE(3), // Set appropriate kernel size
    .NUM_CHANNELS(NUM_CHANNELS)
) conv_inst1 (
    .ifmap(ifmap),
    .filter(filter), // Provide appropriate filter for conv2
    .clk(clk),
    .ofmap(ternary_conv_output)
);

relu_multi1 #(
    .NUM_INSTANCES(IMG_SIZE-KER_SIZE+1),
    .CHANNELS(NUM_CHANNELS)
) relu_inst (
    .ifmap(ternary_conv_output),
    .clk(clk),
    .ofmap(relu_output)
);

// Instantiate convolution module
bitmap_conv #(
    .NUM_CHANNELS(NUM_CHANNELS),
    .IMG_SIZE(IMG_SIZE-KER_SIZE+1)
) conv_inst2 (
    .ifmap(relu_output),
    .filter(filter1), // Provide appropriate filter for conv1
    .clk(clk),
    .ofmap(ofmap)
);





endmodule

//Flattening module
module flatten #(parameter IMG_SIZE=8)
(
    input signed [15:0] input_data [0:IMG_SIZE-1][0:IMG_SIZE-1], // Input data
    input clk,
    output reg signed [15:0] flattened_data [(IMG_SIZE*IMG_SIZE)-1:0] // Flattened output data
);


integer idx;

always @(posedge clk) begin
    idx = 0;
    for (integer i = 0; i < IMG_SIZE; i = i + 1) begin
        for (integer j = 0; j < IMG_SIZE; j = j + 1) begin
            flattened_data[idx] = input_data[i][j];
            idx = idx + 1;
        end
    end
end

endmodule




//LBCNN module 
//contains 4 layers
module lbcnn #(
    parameter IMG_SIZE = 15,
    parameter NUM_CHANNELS = 1,
    parameter KER_SIZE = 3
)(
    input signed [15:0] ifmap[IMG_SIZE-1:0][IMG_SIZE-1:0],
    input signed [15:0] filter_1[NUM_CHANNELS-1:0],
    input signed [15:0] filter1[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0],
    input signed [15:0] filter2[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0],
    input signed [15:0] filter3[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0],
    input signed [15:0] filter4[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0],
    input clk,
    output reg signed [15:0] output_flat[(IMG_SIZE-4*KER_SIZE+4)*(IMG_SIZE-4*KER_SIZE+4)-1:0],
    output reg signed [15:0] output_feature_map[IMG_SIZE-4*KER_SIZE+3:0][IMG_SIZE-4*KER_SIZE+3:0]
);


reg signed [15:0] layer1_output[IMG_SIZE-KER_SIZE:0][IMG_SIZE-KER_SIZE:0];
reg signed [15:0] layer2_output[IMG_SIZE-2*KER_SIZE+1:0][IMG_SIZE-2*KER_SIZE+1:0];
reg signed [15:0] layer3_output[IMG_SIZE-3*KER_SIZE+2:0][IMG_SIZE-3*KER_SIZE+2:0];


// Instantiate 4 layers
layer #(
    .IMG_SIZE(IMG_SIZE),
    .NUM_CHANNELS(NUM_CHANNELS),
    .KER_SIZE(KER_SIZE)
) layer1 (
    .ifmap(ifmap),
    .filter1(filter_1),
    .filter(filter1), // Assuming same filter for simplicity
    .clk(clk),
    .ofmap(layer1_output)
);

layer #(
    .IMG_SIZE(IMG_SIZE-KER_SIZE+1),
    .NUM_CHANNELS(NUM_CHANNELS),
    .KER_SIZE(KER_SIZE)
) layer2 (
    .ifmap(layer1_output),
    .filter1(filter_1), // Assuming same filter for simplicity
    .filter(filter2),
    .clk(clk),
    .ofmap(layer2_output)
);

layer #(
    .IMG_SIZE(IMG_SIZE-2*KER_SIZE+2),
    .NUM_CHANNELS(NUM_CHANNELS),
    .KER_SIZE(KER_SIZE)
) layer3 (
    .ifmap(layer2_output),
    .filter1(filter_1), // Assuming same filter for simplicity
    .filter(filter3),
    .clk(clk),
    .ofmap(layer3_output)
);

layer #(
    .IMG_SIZE(IMG_SIZE-3*KER_SIZE+3),
    .NUM_CHANNELS(NUM_CHANNELS),
    .KER_SIZE(KER_SIZE)
) layer4 (
    .ifmap(layer3_output),
    .filter1(filter_1), // Assuming same filter for simplicity
    .filter(filter4),
    .clk(clk),
    .ofmap(output_feature_map)
);

flatten #(.IMG_SIZE(IMG_SIZE-4*KER_SIZE+4))
flatten1 (
    .input_data(output_feature_map),
    .clk(clk),
    .flattened_data(output_flat)
);



endmodule




