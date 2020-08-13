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
    int output_pos_x = pos.x;
    int output_pos_y = pos.y;
    int out_n4_ind = output_pos_y % UP_DIV(output_shape.x, 4);
    output_pos_y =  output_pos_y / UP_DIV(output_shape.x, 4);
    int out_w_ind = output_pos_y%output_shape.w;
    output_pos_y = output_pos_y/output_shape.w;
    int out_h_ind = output_pos_y%output_shape.z;
    int out_c_ind = output_pos_x;


    float res[4];
    for(int i=0;i<4;++i){
        if(out_n4_ind*4+i>=output_shape.x || out_c_ind>=output_shape.y){
            continue;
        }
        int index = (((out_n4_ind*4+i)*output_shape.y+out_c_ind)*output_shape.z+out_h_ind)*output_shape.w+out_w_ind;
        int offset = index/4;
        if(index%4==0){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
        }else if(index%4==1){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
        }else if(index%4==2){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
        }else if(index%4==3){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
        }
    }

    color.x = res[0];
    color.y = res[1];
    color.z = res[2];
    color.w = res[3];
}
