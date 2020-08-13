// (1, n1, 4)
uniform sampler2D input_image;
// (1, n2, 4)
uniform sampler2D other_image;

// nhwc
uniform ivec4 input_shape;
uniform ivec4 other_shape;
uniform ivec4 output_shape;

uniform int axis;
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
        if(out_n_i<input_shape.x){
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }else{
            int offset = (((out_n_i-input_shape.x)*other_shape.y+out_h_i)*other_shape.z+out_w_i)*UP_DIV(other_shape.w, 4)+out_c_i;
            color = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }
    }else if(axis==1){
        if(out_h_i<input_shape.y){
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }else{
            int offset = ((out_n_i*other_shape.y+out_h_i-input_shape.y)*other_shape.z+out_w_i)*UP_DIV(other_shape.w, 4)+out_c_i;
            color = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }
    }else if(axis==2){
        if(out_w_i<input_shape.z){
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }else{
            int offset = ((out_n_i*other_shape.y+out_h_i)*other_shape.z+out_w_i-input_shape.z)*UP_DIV(other_shape.w, 4)+out_c_i;
            color = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }
    }else{
        // assign seperately
        if(out_c_i*4+0<input_shape.w){
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color.x = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
        }else{
            int in_c_i = out_c_i*4+0-input_shape.w;
            int offset = (((out_n_i*other_shape.y+out_h_i)*other_shape.z+out_w_i)*UP_DIV(other_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.x = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.x = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.x = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.x = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }

        if(out_c_i*4+1<input_shape.w){
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color.y = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
        }else{
            int in_c_i = out_c_i*4+1-input_shape.w;
            int offset = (((out_n_i*other_shape.y+out_h_i)*other_shape.z+out_w_i)*UP_DIV(other_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.y = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.y = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.y = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.y = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }

        if(out_c_i*4+2<input_shape.w){
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color.z = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
        }else{
            int in_c_i = out_c_i*4+2-input_shape.w;
            int offset = (((out_n_i*other_shape.y+out_h_i)*other_shape.z+out_w_i)*UP_DIV(other_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.z = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.z = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.z = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.z = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }

        if(out_c_i*4+3<input_shape.w){
            int offset = ((out_n_i*input_shape.y+out_h_i)*input_shape.z+out_w_i)*UP_DIV(input_shape.w, 4)+out_c_i;
            color.w = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
        }else{
            int in_c_i = out_c_i*4+3-input_shape.w;
            int offset = (((out_n_i*other_shape.y+out_h_i)*other_shape.z+out_w_i)*UP_DIV(other_shape.w, 4)+in_c_i/4);
            if(in_c_i%4==0){
                color.w = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
            }else if(in_c_i%4==1){
                color.w = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
            }else if(in_c_i%4==2){
                color.w = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
            }else {
                color.w = texelFetch(other_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
            }
        }
    }
}
