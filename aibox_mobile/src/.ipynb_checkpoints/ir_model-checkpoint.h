#ifndef _AI_BOX_IR_MODEL_H_
#define _AI_BOX_IR_MODEL_H_

#include "base_model.h"

namespace AIBox {

struct TimeCost
{
    double pre;
    double post;
    double infer;
};

class IRModel : public BaseModel
{
public:
    IRModel(int scale, int tile);
    cv::Mat predict(const cv::Mat &im0, TimeCost &time_cost);

private:
    cv::Mat preprocess(const cv::Mat &im0, cv::Size &size);

    int upscale_;
    int window_size_;
    int tile_;
    int tile_pad_;
};

} // namespace AIBOX

#endif // _AI_BOX_IR_MODEL_H_