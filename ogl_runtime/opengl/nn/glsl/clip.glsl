uniform sampler2D input_image;
uniform float min_value;
uniform float max_value;

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    color = texelFetch(input_image, pos, 0);
    color = max(vec4(min_value), color);
    color = min(vec4(max_value), color);
}
