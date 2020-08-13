#include "opengl/examples/ssd/detector.h"
#include "opengl/core/fbo_session.h"
#include "opengl/core/functor.h"
#include "opengl/utils/util.h"
#include <stdio.h>


namespace opengl{
    namespace{
        class Counter{
            public:
                static void Increase(){count_++;}
                static int Get(){return count_;}
                static Counter* Global(){
                    static auto counter = new Counter;
                    return counter;
                }
            private:
                Counter()=default;
                static int count_;
        };
        int Counter::count_=0;
    }
    /*static*/ std::unique_ptr<Detector> Detector::Create(const string& model_name,
            const TensorNameList& input_names, const TensorNameList& output_names,
            const IntList& shape){
        // use default options here
        DetectorOptions options;
        options.model_name = model_name;
        // remain params keep default
        // wh
        options.input_width = shape[0];
        options.input_height = shape[1];
        options.topk = 100;
        options.nms_threshold=0.45;
        options.score_threshold = 0.3;
        options.variances = {0.1, 0.1, 0.2, 0.2};
        options.input_names = input_names;
        options.output_names = output_names;
        return Create(options);
    }

    /*static*/ std::unique_ptr<Detector> Detector::Create(const DetectorOptions& options){
        auto detector = std::unique_ptr<Detector>(new Detector(options));
        detector->PrintDetectorInfo();
        return detector;
    }

    void Detector::PrintDetectorInfo(){
        std::cout<<"Detector Info: "<<std::endl;
        std::cout<<"variances: "<<variances_[0]<<" "<<variances_[1]
            <<" "<<variances_[2]<<" "<<variances_[3]<<std::endl;
        std::cout<<"input_height: "<<input_sizes_[0]<<std::endl;
        std::cout<<"input_width: "<<input_sizes_[1]<<std::endl;
        std::cout<<"mode_name: "<<model_name_<<std::endl;
        std::cout<<"input_names: "<<input_names_[0]<<std::endl;
        std::cout<<"output_names: "<<output_names_[0]<<std::endl;
        std::cout<<"score_threshold: "<<score_threshold_<<std::endl;
        std::cout<<"nms_threshold: "<<nms_threshold_<<std::endl;
        std::cout<<"topk: "<<topk_<<std::endl;
    }

    Detector::Detector(const DetectorOptions& options){
        variances_ = options.variances;
        model_name_ = options.model_name;
        input_names_ = options.input_names;
        output_names_ = options.output_names;

        score_threshold_ = options.score_threshold;
        nms_threshold_= options.nms_threshold;
        topk_ = options.topk;

        session_.reset(new FBOSession);
        session_->LoadGraph(options.model_name);
        if(session_->allow_resized()){
            input_sizes_.push_back(options.input_height);
            input_sizes_.push_back(options.input_width);
        }else{
            LOG(WARNING)<<"resized input is not allowed in the session"
                << "default input shape is used instead";
            input_sizes_.clear();
            input_sizes_ = session_->default_input_sizes();
        }

        CHECK_EQ(input_sizes_.size(), 2);

        // allocate memory for input tensor
        ::opengl::IntList input_shape({1, input_sizes_[0], input_sizes_[1], 3});
        ::opengl::DataFormat input_dformat = ::dlxnet::TensorProto::NHWC;
        ::opengl::Tensor* image_ptr = Tensor::Empty(Tensor::DT_FLOAT,
                input_shape, input_dformat);

        input_tensors_.emplace_back(image_ptr);
        output_tensors_.clear();
    }



    void Detector::Preprocess(const cv::Mat& raw_image, cv::Mat& image){
        cv::cvtColor(raw_image,image, cv::COLOR_BGR2RGB);

        origin_input_sizes_ = {raw_image.rows, raw_image.cols};
        // order? hw or wh
        cv::resize(image, image, cv::Size(input_sizes_[1], input_sizes_[0]));

        image.convertTo(image, CV_32FC3);
        const float mean_vals[3] = { 123.f, 117.f, 104.f};
        image = image - cv::Scalar(mean_vals[0], mean_vals[1], mean_vals[2]);
    }




    void Detector::GetTopK(std::vector<BoxInfo>& input, int top_k)
    {
        std::sort(input.begin(), input.end(),
                [](const BoxInfo& a, const BoxInfo& b)
                {
                return a.score > b.score;
                });

        if(top_k<input.size()){
            input.erase(input.begin()+top_k, input.end());
        }
    }

    void Detector::NMS(std::vector<BoxInfo>& tmp_faces, std::vector<BoxInfo>& faces, float nms_threshold){
        int N = tmp_faces.size();
        std::vector<int> labels(N, -1);
        for(int i = 0; i < N-1; ++i)
        {
            for (int j = i+1; j < N; ++j)
            {
                cv::Rect pre_box = tmp_faces[i].box;
                cv::Rect cur_box = tmp_faces[j].box;
                float iou_ = iou(pre_box, cur_box);
                if (iou_ > nms_threshold) {
                    labels[j] = 0;
                }
            }
        }

        for (int i = 0; i < N; ++i)
        {
            if (labels[i] == -1)
                faces.push_back(tmp_faces[i]);
        }
    }

    void Detector::GenerateBoxInfo(std::vector<BoxInfo>& boxInfos, float score_threshold){
        // (num_batches, num_samples, num_classes)
        CHECK_EQ(output_tensors_[0]->shape().size(), 3);
        // (num_batches, num_samples, 4)
        CHECK_EQ(output_tensors_[1]->shape().size(), 3);

        /////////////////////////////
        // Debug
        // char fname[100];
        // auto counter = Counter::Global();
        // sprintf(fname, "demo.txt.%d", counter->Get());

        // // get cpu outputs from device
        // TensorList outputs_cpu;
        // session_->GetOutputs({"780"}, {"ANY"}, &outputs_cpu);
        // DumpTensor(outputs_cpu[0], fname);
        // if(counter->Get()>=2){
        // char fname2[100];
        // sprintf(fname2, "demo.txt.%d", counter->Get()-2);
        // CompareTXT(fname, fname2);
        // }
        // counter->Increase();

        auto tensors_host = output_tensors_;

        auto boxes_data = tensors_host[0]->host<float>();
        auto scores_data = tensors_host[1]->host<float>();
        int num_boxes = tensors_host[0]->shape()[1];

        int raw_image_width = origin_input_sizes_[1];
        int raw_image_height = origin_input_sizes_[0];
        num_classes_ = tensors_host[1]->shape()[2];
        int num_cols = num_classes_;

        for(int i = 0; i < num_boxes; ++i)
        {

            // probability decoding, softmax
            float total_sum = exp(scores_data[i*num_cols + 0]);
            // init
            int max_id = 0;
            float max_prob=0;

            for(int j=1;j<num_classes_;j++){
                float logit = exp(scores_data[i*num_cols + j]);
                total_sum  += logit;
                if(max_prob<logit){
                    max_prob = logit;
                    max_id = j;
                }
            }
            float xmin = boxes_data[i*4]*raw_image_width;
            float ymin = boxes_data[i*4+1]*raw_image_height;
            float xmax = boxes_data[i*4+2]*raw_image_width;
            float ymax = boxes_data[i*4+3]*raw_image_height;

            max_prob /= total_sum;


            if (max_prob > score_threshold) {
                BoxInfo tmp_face;
                tmp_face.box.x = xmin;
                tmp_face.box.y = ymin;
                tmp_face.box.width  = xmax - xmin;
                tmp_face.box.height = ymax - ymin;

                // center
                tmp_face.cx = (xmin+xmax)/2.0;
                tmp_face.cy = (ymin+ymax)/2.0;

                tmp_face.height = tmp_face.box.height;
                tmp_face.width = tmp_face.box.width;

                tmp_face.score = max_prob;
                tmp_face.class_name = static_cast<CLASS_NAME>(max_id);
                boxInfos.push_back(tmp_face);
            }
        }
    }

    void Detector::LoadToInputTensors(const cv::Mat& image){
        void* data = input_tensors_[0]->host();
        ::memcpy(data, image.data, input_tensors_[0]->num_elements()*sizeof(float));
    }

    void Detector::Run(const cv::Mat& image){
        // load to cpu input tensor
        LoadToInputTensors(image);
        session_->Run({{"input", input_tensors_[0]}});
        LoadToOutputTensors(output_names_);
    }

    void Detector::LoadToOutputTensors(const TensorNameList& names){
        CHECK_EQ(names.size(), 3);
        Tensor* cls_logits = session_->FindTensorByName(names[0]);
        Tensor* box_preds = session_->FindTensorByName(names[1]);
        Tensor* anchors = session_->FindTensorByName(names[2]);

        Tensor* boxes=nullptr;
        tmp_tensors_.clear();
        if(tmp_tensors_.empty()){
            // cache here
            boxes = new Tensor(Tensor::DT_FLOAT, box_preds->shape(),
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::ANY4);
            tmp_tensors_.emplace_back(boxes);
        }else{
            boxes = tmp_tensors_[0];
        }

        functor::SSDBBoxDecoder()(session_->context(), box_preds, anchors, boxes,
                variances_);

        output_tensors_.clear();
        if(output_tensors_.empty()){
            Tensor* boxes_cpu = Tensor::Empty(Tensor::DT_FLOAT,
                    boxes->shape(), dlxnet::TensorProto::ANY);
            Tensor* cls_logits_cpu = Tensor::Empty(Tensor::DT_FLOAT,
                    cls_logits->shape(), dlxnet::TensorProto::ANY);

            output_tensors_.emplace_back(boxes_cpu);
            output_tensors_.emplace_back(cls_logits_cpu);
        }

        session_->context()->CopyDeviceTensorToCPU(boxes, output_tensors_[0]);
        session_->context()->CopyDeviceTensorToCPU(cls_logits, output_tensors_[1]);
    }

    void Detector::Detect(const cv::Mat& raw_image, std::vector<BoxInfo>& finalBoxInfos){
        // preprocess
        cv::Mat image;
        VLOG(1)<<"Preprocessing ";
        Preprocess(raw_image, image);


        VLOG(1)<<"Running ";
        Run(image);

        VLOG(1)<<"Postprocessing ";
        // postprocess
        std::vector<BoxInfo> boxInfos;
        GenerateBoxInfo(boxInfos, score_threshold_);
        // top k
        GetTopK(boxInfos, topk_);
        // nms
        NMS(boxInfos, finalBoxInfos, nms_threshold_);

        // handle corner case
    }


    Detector::~Detector(){
        // clear input and output tensors
        for(auto tensor:output_tensors_){
            if(tensor){
                delete tensor;
            }
        }

        for(auto tensor:input_tensors_){
            if(tensor){
                delete tensor;
            }
        }
        for(auto tensor:tmp_tensors_){
            if(tensor){
                delete tensor;
            }
        }
    }
} // namespace opengl
