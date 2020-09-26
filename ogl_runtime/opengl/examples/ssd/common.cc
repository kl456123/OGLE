#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <dirent.h>
#include "opengl/utils/logging.h"

static std::vector<std::string> class_names{"bg", "person", "pet_cat", "pet_dog",\
    "sofa", "table", "bed", "excrement", "wire", "key"};

void drawBoxes(const std::vector<BoxInfo>& finalBoxInfos, cv::Mat& raw_image){
    // visualize bbox
    for (auto& face: finalBoxInfos)
    {
        cv::Rect vis_box;
        vis_box.x = (int) face.box.x;
        vis_box.y = (int) face.box.y;
        vis_box.width  = (int) face.box.width;
        vis_box.height = (int) face.box.height;
        auto color = cv::Scalar(0,0,255);
        cv::rectangle(raw_image, vis_box, color, 2);
        // show txt
        auto font = cv::FONT_HERSHEY_SIMPLEX;
        std::stringstream ss;
        ss.precision(3);
        CHECK_GE(face.class_name, 0);
        CHECK_LT(face.class_name, class_names.size());
        ss<<class_names[face.class_name]<<" "<<face.score;
        auto txt = ss.str();
        int baseline;
        float font_scale = 0.5;
        cv::Size cat_size = cv::getTextSize(txt, font, font_scale, 2, &baseline);

        // get color for specific class
        cv::rectangle(raw_image,
                cv::Rect(face.box.x, face.box.y - cat_size.height - 2+face.box.height,
                    cat_size.width, cat_size.height), color, -1);
        int thickness = 1;
        cv::putText(raw_image, txt, cv::Point(face.box.x, face.box.y - 2+face.box.height),
                font, font_scale, cv::Scalar(0,0,0), thickness, cv::LINE_AA);
    }

}

void *open_video_stream(const std::string& f, int c, int w, int h, int fps)
{
    cv::VideoCapture *cap;
    if(!f.empty()) cap = new cv::VideoCapture(f.c_str());
    else cap = new cv::VideoCapture(c);
    if(!cap->isOpened()) return NULL;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, h);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return cap;
}

float iou(cv::Rect box0, cv::Rect box1)
{
    float xmin0 = box0.x;
    float ymin0 = box0.y;
    float xmax0 = box0.x + box0.width;
    float ymax0 = box0.y + box0.height;

    float xmin1 = box1.x;
    float ymin1 = box1.y;
    float xmax1 = box1.x + box1.width;
    float ymax1 = box1.y + box1.height;

    float w = fmax(0.0f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
    float h = fmax(0.0f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));

    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;

    if (u <= 0.0) return 0.0f;
    else          return i/u;
}

float get_size(float w, float h){
    float pad  = (w+h)*0.5;
    return std::sqrt((w+pad)*(h+pad));
}

void softmax(float* data_in,float* data_out, int num){
    float exp[num];
    float total_sum=0;
    for(int i=0;i<num;i++){
        exp[i] = std::exp(data_in[i]);
        total_sum+=exp[i];
    }
    for(int i=0;i<num;i++){
        data_out[i] = exp[i]/total_sum;
    }
}

void ReadFilesFromDir(const std::string& path_to_dir,
        std::vector<std::string>* image_name_list){
    DIR *dir;
    dir = opendir(path_to_dir.c_str());
    struct dirent *ent;
    CHECK_NOTNULL(dir);
    while ((ent = readdir (dir)) != NULL) {
        auto name= std::string(ent->d_name);
        // ignore "." ".."
        if(name.size()<4){
            continue;
        }
        auto suffix = name.substr(name.size()-4, 4);
        if(suffix==".png"||suffix==".jpg"){
            // filter image
            image_name_list->emplace_back(name);
        }
    }

    closedir(dir);
}
