// (1, n1, 4)
uniform sampler2D input_image;
uniform sampler2D origin_image;

// nhwc
uniform ivec4 input_shape;
uniform ivec4 output_shape;

uniform int axis;
uniform int axis_offset;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // find the index of output
    // decompose output index to output coords
    int index = pos.x + pos.y*MAX_TEXTURE_SIZE;
    int out_c_i = index % UP_DIV(output_shape.w, 4);
    index = index/UP_DIV(output_shape.w, 4);
    int out_w_i = index%output_shape.z;
    index = index/output_shape.z;
    int out_h_i = index%output_shape.y;
    index = index/output_shape.y;
    int out_n_i = index%output_shape.x;

    if(axis==0){
        if(out_n_i<axis_offset||out_n_i>=axis_offset+input_shape.x){
            color = texelFetch(origin_image, pos, 0);
        }else{
            int offset = (((out_n_i-axis_offset)*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }
    }else if(axis==1){
        if(out_h_i<axis_offset||out_h_i>=axis_offset+input_shape.y){
            color = texelFetch(origin_image, pos, 0);
        }else{
            int offset = ((out_n_i*input_shape.y+out_h_i-axis_offset)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }
    }else if(axis==2){
        if(out_w_i<axis_offset||out_w_i>=axis_offset+input_shape.z){
            color = texelFetch(origin_image, pos, 0);
        }else{
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i-axis_offset)*UP_DIV(input_shape.w, 4)+out_c_i;
            color = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }
    }else{
        // assign seperately
        if(out_c_i*4+0<axis_offset||out_c_i*4+0>=axis_offset+input_shape.w){
            color.x = texelFetch(origin_image, pos, 0).x;
        }else{
            int in_c_i = out_c_i*4+0-axis_offset;
            int offset = (((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.x = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.x = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.x = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.x = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }

        if(out_c_i*4+1<axis_offset||out_c_i*4+1>=axis_offset+input_shape.w){
            color.y = texelFetch(origin_image, pos, 0).y;
        }else{
            int in_c_i = out_c_i*4+1-axis_offset;
            int offset = (((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.y = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.y = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.y = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.y = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }

        if(out_c_i*4+2<axis_offset||out_c_i*4+1>=axis_offset+input_shape.w){
            color.z = texelFetch(origin_image, pos, 0).z;
        }else{
            int in_c_i = out_c_i*4+2-axis_offset;
            int offset = (((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.z = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.z = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.z = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.z = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }

        if(out_c_i*4+3<axis_offset||out_c_i*4+1>=axis_offset+input_shape.w){
            color.w = texelFetch(origin_image, pos, 0).w;
        }else{
            int in_c_i = out_c_i*4+3-axis_offset;
            int offset = (((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.w = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.w = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.w = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.w = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }
    }
}
