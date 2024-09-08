#include "pw_ai_call.h"

using namespace AIBox;

int main(int argc, char **argv)
{
    AIBox::InferOptions opt;
    opt.device = Device::GPU;
    opt.precision = Precision::FP32;
    opt.iter_count = 10;
    AIBox::InferStatistic data = {0};
    AIBox::imageRestore(argv[1], argv[2], "./test.jpg", opt, data);
    return 0;
}