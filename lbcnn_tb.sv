module lbcnn_tb;

    // Parameters
    parameter IMG_SIZE = 15;
    parameter NUM_CHANNELS = 1;
    parameter KER_SIZE = 3;
    parameter FC_INPUT_SIZE = (IMG_SIZE-4*KER_SIZE+4)*(IMG_SIZE-4*KER_SIZE+4);

    // Inputs
    logic signed [15:0] ifmap[IMG_SIZE-1:0][IMG_SIZE-1:0];
    logic signed [15:0] filter_1[NUM_CHANNELS-1:0];
    logic signed [15:0] filter1[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0];
    logic signed [15:0] filter2[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0];
    logic signed [15:0] filter3[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0];
    logic signed [15:0] filter4[KER_SIZE-1:0][KER_SIZE-1:0][NUM_CHANNELS-1:0];
    logic clk;
    logic rst;
    
    // Outputs
    logic signed [15:0] output_feature_map[IMG_SIZE-4*KER_SIZE+3:0][IMG_SIZE-4*KER_SIZE+3:0];
    logic signed [15:0] output_flat[(IMG_SIZE-4*KER_SIZE+4)*(IMG_SIZE-4*KER_SIZE+4)-1:0];

    // Instantiate the module under test
    lbcnn #(
        .IMG_SIZE(IMG_SIZE),
        .NUM_CHANNELS(NUM_CHANNELS),
        .KER_SIZE(KER_SIZE)
    ) dut (
        .ifmap(ifmap),
        .filter_1(filter_1),
        .filter1(filter1),
        .filter2(filter2),
        .filter3(filter3),
        .filter4(filter4),
        .clk(clk),
        .output_flat(output_flat),
        .output_feature_map(output_feature_map)
    );
    
    // Clock generation
    always #5 clk = ~clk;

    // Reset generation
    initial begin
        rst = 1;
        #10;
        rst = 0;
    end

    // Test stimulus
    initial begin
        // Initialize clock
        clk = 0;
        // Example input feature map
        for (int i = 0; i < IMG_SIZE; i = i + 1) begin
            for (int j = 0; j < IMG_SIZE; j = j + 1) begin
                ifmap[i][j] = i+1;
            end
        end
        
        // Example filter (3x3, single channel)
        filter1[0][0][0] = 1; filter1[0][1][0] = 0; filter1[0][2][0] = 1;
        filter1[1][0][0] = 0; filter1[1][1][0] = -1; filter1[1][2][0] = 0;
        filter1[2][0][0] = 1; filter1[2][1][0] = 0; filter1[2][2][0] = 1;

        filter2[0][0][0] = 1; filter2[0][1][0] = 0; filter2[0][2][0] = 1;
        filter2[1][0][0] = 0; filter2[1][1][0] = -1; filter2[1][2][0] = 0;
        filter2[2][0][0] = 1; filter2[2][1][0] = 0; filter2[2][2][0] = 1;

        filter3[0][0][0] = 1; filter3[0][1][0] = 0; filter3[0][2][0] = 1;
        filter3[1][0][0] = 0; filter3[1][1][0] = -1; filter3[1][2][0] = 0;
        filter3[2][0][0] = 1; filter3[2][1][0] = 0; filter3[2][2][0] = 1;

        filter4[0][0][0] = 1; filter4[0][1][0] = 0; filter4[0][2][0] = 1;
        filter4[1][0][0] = 0; filter4[1][1][0] = -1; filter4[1][2][0] = 0;
        filter4[2][0][0] = 1; filter4[2][1][0] = 0; filter4[2][2][0] = 1;

        // Example filter 1
        for (int i = 0; i < NUM_CHANNELS; i = i + 1) begin
            filter_1[i] = 16'd2;
        end

        
        // Apply stimulus for some cycles
        #1000;

        // Print output feature map
        $display("Output Feature Map:");
        for (int i = 0; i < IMG_SIZE-4*KER_SIZE+3; i = i + 1) begin
            for (int j = 0; j < IMG_SIZE-4*KER_SIZE+3; j = j + 1) begin
                $write("%d ", output_feature_map[i][j]);
            end
            $write("\n");
        end

        // Print flattened output
        $display("Flattened Image:");
        for (int i = 0; i < FC_INPUT_SIZE; i = i + 1) begin
            $write("%d ", output_flat[i]);
        end
        $write("\n");


        

        // End of simulation
        $finish;
    end

endmodule
