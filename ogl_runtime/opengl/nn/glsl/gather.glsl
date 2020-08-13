uniform sampler2D input_image;
uniform sampler2D input_index;
uniform int axis;

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // assume it is scalar and refers to nchw dformat
    int selected_index = int(texelFetch(input_index, ivec2(0), 0).x);
    if(selected_index==0){
        color.x = texelFetch(input_image, ivec2(0), 0).x;
    }else if(selected_index==1){
        color.x = texelFetch(input_image, ivec2(0), 0).w;
    }else if(selected_index==2){
        color.x = texelFetch(input_image, ivec2(0), 0).y;
    }else{
        color.x = texelFetch(input_image, ivec2(0), 0).z;
    }
}
