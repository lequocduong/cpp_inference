#ifndef ANDROIDIPCPRO_PW_AI_CALL_H
#define ANDROIDIPCPRO_PW_AI_CALL_H

#include "base_model.h"
#include <string>

namespace AIBox {

struct InferOptions
{
    Device device;
    Precision precision;
    int iter_count;
};

struct InferStatistic
{
    int frame_width;
    int frame_height;
    int frame_count;
    double fps;
    double model_load_time;
    double pre_cost;
    double post_cost;
    double aver_infer_cost;
    double max_infer_cost;
    double min_infer_cost;
    double one_frame_cost;
    double total_cost;
};

int imageRestore(const std::string &model_path, const std::string &input_path, const std::string &save_path, 
                 const InferOptions &opt, InferStatistic &data);

}

#endif //ANDROIDIPCPRO_PW_AI_CALL_H
