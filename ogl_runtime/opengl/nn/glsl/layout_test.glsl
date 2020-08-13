// (1, (nchw)/4, 4)
uniform sampler2D input_image;

// nchw
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (h*w* out_4, in_4*in4, out4)
// chw: (out4, h*w* out_4, in_4*in4)
// zyx
out vec4 color;

void main(){
    // w, h, c
    ivec2 input_texture_size = textureSize(input_image, 0);
    int spatial_texture_size = input_texture_size.x*input_texture_size.y;

    ivec2 pos = ivec2(gl_FragCoord.xy);
    color.x = 1.0;
    color.y = 2.0;
}
