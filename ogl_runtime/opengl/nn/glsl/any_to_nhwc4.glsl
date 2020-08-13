// (1, (nhwc)/4, 4)
uniform sampler2D input_image;

// from any to any4
// output shape is equal to input shape
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (nh, wc/4, 4)
out vec4 color;

vec4 get_vec4(int index){
    vec4 res;
    int offset = index/4;
    if(index%4==0){
        res= texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
    }else if(index%4==1){
        res.xyz= texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).yzw;
        offset+=1;
        res.w = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
    }else if(index%4==2){
        res.xy= texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).zw;
        offset+=1;
        res.zw = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).xy;
    }else if(index%4==3){
        res.x= texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
        offset+=1;
        res.yzw = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).xyz;
    }
    return res;
}

void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);

    int out_h_i = pos.y%output_shape.y;
    int out_n_i = pos.y/output_shape.y;
    int out_c_i = pos.x%UP_DIV(output_shape.w, 4);
    int out_w_i = pos.x/UP_DIV(output_shape.w, 4);

    int offset = ((out_n_i*output_shape.y+out_h_i)*output_shape.z+out_w_i)*output_shape.w+out_c_i*4;
    color = get_vec4(offset);
    int remain = output_shape.w - out_c_i*4;
    if(remain>=4){
    }else if(remain==1){
        color.yzw = vec3(0.0);
    }else if(remain==2){
        color.zw = vec2(0.0);
    }else if(remain==3){
        color.w = 0.0;
    }
}
