/**
 * group=1, dialtion=1, kernel=1, stride=1, padding=0
 */
// nhwc4
uniform PRECISION sampler2D input_image;
// hwo4i4
// (h*w*o/4, i/4*i4, o4)
uniform PRECISION sampler2D   input_filter;
uniform PRECISION sampler2D  input_bias;
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
    // decompose pos
    // pos = (w*out_4_i, nh_i)
    // output_shape=(h,w,c)
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;

    int out_4_ind = pos.x%UP_DIV(output_shape.z, 4);
    int output_index_x = pos.x/UP_DIV(output_shape.z, 4);

#ifdef USE_BIAS
    color = texelFetch(input_bias, ivec2(out_4_ind%MAX_TEXTURE_SIZE,   out_4_ind/MAX_TEXTURE_SIZE), 0);
#else
    color = vec4(0.0);
#endif
    int in_4_dim = UP_DIV(input_shape.z, 4);

    int input_index_x = output_index_x;
    int input_index_y = output_index_y;
    int tmp = batch_ind*input_shape.x;
    // loop in channel dim
    for(int in_4_ind=0;in_4_ind<in_4_dim;++in_4_ind){
        // get input image
        int input_pos_y = tmp+input_index_y;
        int input_pos_x = input_index_x*in_4_dim+in_4_ind;

        // get input filter
        // filter shape: (out_4, in_4*in4, out4)
        int filter_pos_y = out_4_ind;
        int filter_pos_x = in_4_ind*4;
        vec4 k0 = texelFetch(input_filter, ivec2(filter_pos_x,   filter_pos_y), 0);
        vec4 k1 = texelFetch(input_filter, ivec2(filter_pos_x+1, filter_pos_y), 0);
        vec4 k2 = texelFetch(input_filter, ivec2(filter_pos_x+2, filter_pos_y), 0);
        vec4 k3 = texelFetch(input_filter, ivec2(filter_pos_x+3, filter_pos_y), 0);
        /* vec4 k0=vec4(1.0); */
        /* vec4 k1=vec4(2.0); */
        /* vec4 k2=vec4(3.0); */
        /* vec4 k3=vec4(4.0); */

        // kernel matrix
        mat4 k = mat4(k0, k1, k2, k3);

        // a 4-elements tuple in output channel dim
        color+=k*texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
    }

#ifdef USE_CLIP
    /* color = max(vec4(min_value), color); */
    /* color = min(vec4(max_value), color); */
    color = clamp(color, vec4(0.0), vec4(max_value));
#endif

#ifdef USE_RELU
    color = max(vec4(0.0), color);
#endif
}
