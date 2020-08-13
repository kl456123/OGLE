uniform sampler2D input_image;

// other inputs
uniform sampler2D input_gamma;
uniform sampler2D input_beta;
uniform sampler2D input_mean;
uniform sampler2D input_var;

// some attr params
uniform float momentum;
uniform float eps;
uniform ivec3 output_shape;
out vec4 color;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))
// color: shape(NH, WC4, 4)
// shape: (height, width, channel)
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // decompose output index
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;
    int out_4_ind = pos.x%UP_DIV(output_shape.z, 4);
    int output_index_x = pos.x/UP_DIV(output_shape.z, 4);

    vec4 mean = texelFetch(input_mean, ivec2(out_4_ind, 0), 0);
    vec4 var = texelFetch(input_var, ivec2(out_4_ind, 0), 0);
    vec4 beta = texelFetch(input_beta, ivec2(out_4_ind, 0), 0);
    vec4 gamma = texelFetch(input_gamma, ivec2(out_4_ind, 0), 0);

    color = (texelFetch(input_image, pos, 0)-mean)/sqrt(var+eps)*gamma+beta;
}
