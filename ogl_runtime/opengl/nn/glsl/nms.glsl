// (num_batches* num_samples, 4)
uniform PRECISION sampler2D boxes;
/* uniform PRECISION sampler2D scores; */



uniform int topk;
uniform float nms_threshold;

// (num_batches, num_samples, 4)
uniform ivec3 input_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))
float iou(vec4 box_a, vec4 box_b){
    float w = max(0.0f, min(box_a.z, box_b.z) - max(box_a.x, box_b.x));
    float h = max(0.0f, min(box_a.w, box_b.w) - max(box_a.y, box_b.y));
    float i = w * h;
    float u = (box_a.z - box_a.x) * (box_a.w - box_a.y) + (box_b.z - box_b.x) * (box_b.w - box_b.y) - i;

    if (u <= 0.0) return 0.0f;
    else          return i/u;
}

// (num_batches*num_samples*num_samplers/4, 4)
out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    int index = pos.x+pos.y*MAX_TEXTURE_SIZE;
    int j4 = index%UP_DIV(input_shape.y, 4);
    index = index/UP_DIV(input_shape.y, 4);
    int i = index%input_shape.y;
    int batch_ind = index/input_shape.y;

    vec4 box_i = texelFetch(boxes, ivec2(index%MAX_TEXTURE_SIZE, index/MAX_TEXTURE_SIZE), 0);

    int index_j;
    float ious[4];
    for(int i=0;i<4;++i){
        index_j = j4*4+i;
        vec4 box_j = texelFetch(boxes, ivec2(index_j%MAX_TEXTURE_SIZE, index_j/MAX_TEXTURE_SIZE), 0);
        ious[i] = iou(box_i, box_j);
    }
    color.x = ious[0];
    color.y = ious[1];
    color.z = ious[2];
    color.w = ious[3];
    /* color = texelFetch(boxes, pos, 0); */
}
