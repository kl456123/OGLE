uniform sampler2D input_image;
uniform sampler2D input_filter;
uniform sampler2D input_bias;
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
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))

// filter shape: (h*w* out_4, in_4*in4, out4)
// image shape: (n*h, w*in_4, in4)
// output shape: (n*h, w*out_4, out4)
// bias shape: (1, out_4, out4)
// where in4=out4 = 4
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // decompose pos
    // pos = (w*out_4_i, nh_i)
    // output_shape=(h,w,c)
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;

    int out_4_ind = pos.x%UP_DIV(output_shape.z, 4);
    int output_index_x = pos.x/UP_DIV(output_shape.z, 4);

    int out_group_size = output_shape.z/group;
    int in_group_size = input_shape.z/group;

#ifdef USE_BIAS
    color = texelFetch(input_bias, ivec2(out_4_ind%MAX_TEXTURE_SIZE,   out_4_ind/MAX_TEXTURE_SIZE), 0);
#else
    color = vec4(0.0);
#endif
    float res[4];
    for(int c_i=0;c_i<4;++c_i){
        int out_c_ind = out_4_ind*4+c_i;
        int grp_ind = out_c_ind/out_group_size;
        float value=0.0;
        for(int i=0;i<kernel_size;++i){
            for (int j=0;j<kernel_size;++j) {
                int input_index_x = output_index_x*stride_size+i*dilation-padding;
                int input_index_y = output_index_y*stride_size+j*dilation-padding;
                if(input_index_x<0||input_index_x>=input_shape.y){
                    continue;
                    // when out of boundary
                }
                if(input_index_y<0||input_index_y>=input_shape.x){
                    continue;
                }
                // loop in channel dim
                for(int k=0;k<in_group_size;++k){
                    // get input value
                    // w*c/4
                    int in_c_ind = k+grp_ind*in_group_size;
                    int input_pos_x = input_index_x*UP_DIV(input_shape.z, 4)+in_c_ind/4;
                    // n*h
                    int input_pos_y = batch_ind*input_shape.x+input_index_y;
                    vec4 input_vec = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
                    float input_value;
                    if(in_c_ind%4==0){
                        input_value = input_vec.x;
                    }else if(in_c_ind%4==1){
                        input_value = input_vec.y;
                    }else if(in_c_ind%4==2){
                        input_value = input_vec.z;
                    }else if(in_c_ind%4==3){
                        input_value = input_vec.w;
                    }
                    // get filter value
                    // (h*w* out_4, in_4*in4, out4)
                    // in_4*in4
                    int f_c_ind = k;
                    int f_n_ind = out_c_ind;
                    int filter_pos_x = k;
                    // h*w*out_4
                    int filter_pos_y=(j*kernel_size+i)*UP_DIV(output_shape.z,4)+f_n_ind/4;
                    float filter_value;
                    vec4 filter_vec = texelFetch(input_filter, ivec2(filter_pos_x, filter_pos_y), 0);
                    if(out_c_ind%4==0){
                        filter_value = filter_vec.x;
                    }else if(out_c_ind%4==1){
                        filter_value = filter_vec.y;
                    }else if(out_c_ind%4==2){
                        filter_value = filter_vec.z;
                    }else if(out_c_ind%4==3){
                        filter_value = filter_vec.w;
                    }
                    value+=input_value*filter_value;
                }
            }
        }

        res[c_i] = value;
    }
    color.x +=res[0];
    color.y +=res[1];
    color.z +=res[2];
    color.w +=res[3];

#ifdef USE_CLIP
    color = max(vec4(min_value), color);
    color = min(vec4(max_value), color);
#endif

#ifdef USE_RELU
    color = max(vec4(min_value), color);
#endif
}
