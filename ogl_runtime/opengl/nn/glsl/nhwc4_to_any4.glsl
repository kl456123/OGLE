// (nh, wc/4, 4)
uniform sampler2D input_image;

// from nhwc4 to any4
// output shape is equal to input shape
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (1, n * h * w * c/4, 4)
out vec4 color;

void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);

    int index = pos.x+pos.y*MAX_TEXTURE_SIZE;
    int output_num_elements = output_shape.x * output_shape.y
                    * output_shape.z * UP_DIV(output_shape.w, 4);
    if(index>=output_num_elements){
        return;
    }

    int base = UP_DIV(output_shape.w, 4)*output_shape.z;

    int input_pos_y = index/base;
    int input_pos_x = index%base;

    color = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
}
