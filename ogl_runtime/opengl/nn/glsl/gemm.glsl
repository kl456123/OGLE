
// M, K
uniform sampler2D A;
// K, N
uniform sampler2D B;
// M, N
uniform sampler2D C;

uniform float alpha;
uniform float beta;

uniform int use_bias;
uniform int transb;


// M, K
uniform ivec2 input_shape;
// M, N
uniform ivec2 output_shape;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    int out_dim4 = UP_ROUND(output_shape.y, 4);

    float res[4];

    for(int k=0;k<4;++k)
    {
        int output_4_index = (pos.x+pos.y*MAX_TEXTURE_SIZE)*4+k;
        int output_index = output_4_index/out_dim4*output_shape.y+ output_4_index%out_dim4;
        int j = output_index%output_shape.y;
        int i = output_index/output_shape.y;
        for(int m=0;m<input_shape.y;++m){
            int a_index = i*input_shape.y+m;
            int b_index;
            if(transb==1){
                b_index = j*input_shape.y+m;
            }else{
                b_index = m*output_shape.y+j;
                }
            float a;
            float b;
            {
                int tmp = a_index%input_shape.y;
                int offset = a_index/input_shape.y*UP_DIV(input_shape.y, 4)+tmp/4;
                if(tmp%4==0){
                    a = texelFetch(A, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
                }else if(tmp%4==1){
                    a = texelFetch(A, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
                }else if(tmp%4==2){
                    a = texelFetch(A, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
                }else if(tmp%4==3){
                    a = texelFetch(A, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
                }

            }

            {
                int last_dim;
                if(transb==0){
                    last_dim= output_shape.y;
                }else{
                    last_dim = input_shape.y;
                }

                int tmp = b_index % last_dim;
                int offset = b_index / last_dim * UP_DIV(last_dim, 4)+tmp/4;
                if(tmp%4==0){
                    b = texelFetch(B, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
                }else if(tmp%4==1){
                    b = texelFetch(B, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
                }else if(tmp%4==2){
                    b = texelFetch(B, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
                }else if(tmp%4==3){
                    b = texelFetch(B, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
                }
            }
            res[k] += alpha*a*b;
        }


    }

    color.x = res[0];
    color.y = res[1];
    color.z = res[2];
    color.w = res[3];

    if(use_bias==1){
        color+=beta*texelFetch(C, pos, 0);
    }
}
