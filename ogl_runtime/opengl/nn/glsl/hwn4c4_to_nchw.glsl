// (hwo/4, i/4*4, 4)
uniform sampler2D input_image;

// nchw
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (1, (nchw)/4, 4)
out vec4 color;

void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);

    float res[4];
    for(int i=0;i<4;++i){
        int output_index = (pos.x+pos.y*MAX_TEXTURE_SIZE)*4+i;
        int out_w_ind = output_index%output_shape.w;
        output_index = output_index/output_shape.w;
        int out_h_ind = output_index%output_shape.z;
        output_index = output_index/output_shape.z;
        int out_c_ind = output_index%output_shape.y;
        int out_n_ind = output_index/output_shape.y;

        int input_pos_x = out_c_ind;
        int input_pos_y = (out_h_ind*output_shape.w+out_w_ind)*UP_DIV(output_shape.x, 4)+out_n_ind/4;
        if(out_n_ind%4==0){
            res[i] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).x;
        }else if(out_n_ind%4==1){
            res[i] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).y;
        }else if(out_n_ind%4==2){
            res[i] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).z;
        }else if(out_n_ind%4==3){
            res[i] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).w;
        }
    }

    color.x = res[0];
    color.y = res[1];
    color.z = res[2];
    color.w = res[3];
}
