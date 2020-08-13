// (1, (nhwc)/4, 4)
uniform sampler2D input_image;

// from any to any4
// output shape is equal to input shape
uniform ivec4 output_shape;
// uniform int output_num_elements;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (1, nhwc/4, 4)
out vec4 color;

vec4 get_vec4(int index){
    // get vec4 from any dformat
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

    int out_4_dim = UP_DIV(output_shape.w, 4)*4;

    int output_num_elements = output_shape.x * output_shape.y
                    * output_shape.z * UP_DIV(output_shape.w, 4);
    if(pos.x+pos.y*MAX_TEXTURE_SIZE>=output_num_elements){
        return;
    }

    int output_4_index = (pos.x+pos.y*MAX_TEXTURE_SIZE)*4;
    int remain =  output_shape.w - output_4_index%out_4_dim;
    int index = output_4_index/out_4_dim*output_shape.w+output_4_index%out_4_dim;
    color = get_vec4(index);
    if(remain>=4){
    }else if(remain==1){
        color.yzw = vec3(0.0);
    }else if(remain==2){
        color.zw = vec2(0.0);
    }else if(remain==3){
        color.w = 0.0;
    }
}
