// (nh, wc/4, 4)
uniform sampler2D input_image;

// 4
uniform ivec4 perm;

// hwc
uniform ivec4 input_shape;
// hwc
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))
out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    int index = pos.x;
    int out_c4_i = index % UP_DIV(output_shape.w, 4);
    index = index/UP_DIV(output_shape.w, 4);
    int out_w_i = index%output_shape.z;
    index = index/output_shape.z;
    int out_h_i = index%output_shape.y;
    index = index/output_shape.y;
    int out_n_i = index%output_shape.x;

    int perms[4];
    perms[0] = perm.x;
    perms[1] = perm.y;
    perms[2] = perm.z;
    perms[3] = perm.w;
    int input_base = UP_DIV(input_shape.w, 4)*input_shape.z;

    int input_coord[4];
    float res[4];
    for(int ind=0;ind<4;++ind){
        int out_c_i = out_c4_i*4+ind;
        int output_coord[4];
        output_coord[0]= out_n_i;
        output_coord[1] = out_h_i;
        output_coord[2] = out_w_i;
        output_coord[3] = out_c_i;
        // transpose coord
        for(int i=0;i<4;++i){
            input_coord[perm[i]] = output_coord[i];
        }
        // input_coord to offset
        // nh, wc4, 4
        /* int input_pos_x = input_coord[2]*UP_DIV(input_shape.w, 4)+input_coord[3]/4; */
        /* int input_pos_y = input_coord[0]*input_shape.y+input_coord[1]; */
        /* res[ind] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0); */

        // nhw*c/4 , 4
        int offset = ((input_coord[0]*input_shape.y+input_coord[1])*input_shape.z+input_coord[2])*UP_DIV(input_shape.w, 4)+input_coord[3]/4;


        int input_pos_y = offset/input_base;
        int input_pos_x = offset%input_base;


        if(input_coord[3]%4==0){
            res[ind] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).x;
        }else if(input_coord[3]%4==1){
            res[ind] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).y;
        }else if(input_coord[3]%4==2){
            res[ind] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).z;
        }else if(input_coord[3]%4==3){
            res[ind] = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).w;
        }
    }

    color.x = res[0];
    color.y = res[1];
    color.z = res[2];
    color.w = res[3];
}
