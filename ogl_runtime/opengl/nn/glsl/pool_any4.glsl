uniform sampler2D input_image;

// pool params
uniform int kernel_size;
uniform int stride_size;
uniform int padding;
uniform int pool_type;

// height, width and channel of input and output
uniform ivec3 input_shape;
uniform ivec3 output_shape;
out vec4 color;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))
// color: shape(NH, WC4, 4)
// shape: (height, width, channel)
// image shape: (n*h, w*in_4, in4)
// output shape: (n*h, w*out_4, out4)
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    int index = pos.x + pos.y * MAX_TEXTURE_SIZE;
    int out_4_dims = UP_DIV(output_shape.z, 4);
    int input_base = UP_DIV(input_shape.z, 4)*input_shape.y;

    int out_4_ind = index%out_4_dims;
    index = index/out_4_dims;
    int output_index_x = index%output_shape.y;
    index = index/output_shape.y;
    int output_index_y = index%output_shape.x;
    int batch_ind = index/output_shape.x;

    // init color value
    bool first = true;
    color = vec4(0.0);

    for(int i=0;i<kernel_size;++i){
        for (int j=0;j<kernel_size;++j) {
            int input_index_x = output_index_x*stride_size+i-padding;
            int input_index_y = output_index_y*stride_size+j-padding;
            if(input_index_x<0||input_index_x>=input_shape.y){
                continue;
                // when out of boundary
            }
            if(input_index_y<0||input_index_y>=input_shape.x){
                continue;
            }
            // channel of output is consisted with input
            int in_4_ind = out_4_ind;
            int input_pos_y = batch_ind*input_shape.x+input_index_y;
            int input_pos_x = input_index_x*UP_DIV(input_shape.z, 4)+in_4_ind;
            int input_index = input_pos_x+input_pos_y*input_base;

            vec4 item = texelFetch(input_image, ivec2(input_index%MAX_TEXTURE_SIZE, input_index/MAX_TEXTURE_SIZE), 0);
            if(pool_type==0){
                // max pool
                if(first){
                    color = item;
                    first = false;
                }else{
                    color=max(color, item);
                }
            }else{
                color +=item;
            }

        }
    }
    if(pool_type!=0){
        // average pool
        color = color/(float(kernel_size*kernel_size));
    }
}
