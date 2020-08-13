uniform sampler2D input_image;
uniform ivec3 input_shape;

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    color = texelFetch(input_image, pos, 0);
    color = max(vec4(0.0), color);
}
