// (nh, wc/4, 4)
// nhwc4
uniform  PRECISION sampler2D   input_image;

// (h*w*o/4, 1, o4)
// hwio4
uniform  PRECISION sampler2D   input_filter;

// (1, o/4, 4)
// any4
uniform PRECISION sampler2D   input_bias;

// conv2d params
uniform int stride_size;
uniform int kernel_size;
uniform int group;
uniform int dilation;
uniform int padding;
// use int type to represent bool type
uniform int use_bias;

// fused op with activation(means clip here)
uniform int act;
uniform float min_value;
uniform float max_value;

// height, width and channel of input and output
uniform ivec3 input_shape;
uniform ivec3 output_shape;
out vec4 color;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// filter shape: (h*w* out_4, in_4*in4, out4)
// image shape: (n*h, w*in_4, in4)
// output shape: (n*h, w*out_4, out4)
// bias shape: (1, out/4, out4)
// where in4=out4 = 4
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    int out_4_dims = UP_DIV(output_shape.z, 4);
    int in_4_dims = UP_DIV(input_shape.z, 4);

    // decompose pos
    // pos = (w*out_4_i, nh_i)
    // output_shape=(h,w,c)
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;

    int out_4_ind = pos.x%out_4_dims;
    int output_index_x = pos.x/out_4_dims;

#ifdef USE_BIAS
    color = texelFetch(input_bias, ivec2(out_4_ind%MAX_TEXTURE_SIZE,   out_4_ind/MAX_TEXTURE_SIZE), 0);
#else
    color = vec4(0.0);
#endif

    // make sure out_group_size=1
    /* int out_group_size = output_shape.z/group; */
    /* int in_group_size = input_shape.z/group; */
    int input_index_x_base = output_index_x*stride_size-padding;
    int input_index_y_base = output_index_y*stride_size-padding;

    for (int j=0;j<kernel_size;++j) {
        for(int i=0;i<kernel_size;++i){
            int input_index_x = i*dilation+input_index_x_base;
            int input_index_y = j*dilation+input_index_y_base;
            if(input_index_x<0||input_index_x>=input_shape.y){
                continue;
                // when out of boundary
            }
            if(input_index_y<0||input_index_y>=input_shape.x){
                continue;
            }

            int input_pos_y = batch_ind*input_shape.x+input_index_y;
            int input_pos_x = input_index_x*in_4_dims+out_4_ind;

            // (h*w*o/4, 1, o4)
            int filter_pos_y = (j*kernel_size+i)*out_4_dims+out_4_ind;
            vec4 k = texelFetch(input_filter, ivec2(0, filter_pos_y), 0);

            color+=k*texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
        }
    }

#ifdef USE_CLIP
    /* color = max(vec4(min_value), color); */
    /* color = min(vec4(max_value), color); */
    color = clamp(color, vec4(min_value), vec4(max_value));
#endif

#ifdef USE_RELU
    color = max(vec4(0.0), color);
#endif
}
