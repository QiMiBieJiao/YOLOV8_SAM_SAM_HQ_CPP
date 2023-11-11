#ifndef ORTYOLONET_H
#define ORTYOLONET_H
#include<baseortnet.h>
class YOLOOrtNetPrivate;
class ORTNET_EXPORT YOLOOrtNet:public BaseOrtNet
{
public:
    enum Type
    {
        Classify,
        Detect,
        Pose,
        Segment,
        FastSAM
    };
    struct Task
    {
        Type taskType;
        int classCount=-1;
        std::vector<int64_t> kptShape;
    };
    YOLOOrtNet(const Task &cfg);
    Task getTask();
    std::vector<NetOutputResult> predict(const cv::Mat &src,const PromptFilter &filter);
    std::vector<std::vector<NetOutputResult>> predictBatch(const std::vector<cv::Mat> &srcVec,const PromptFilter &filter);
protected:
    virtual bool netSanityCheck();;
private:
    std::unique_ptr<YOLOOrtNetPrivate> d;
    friend class YOLOOrtNetPrivate;

};

#endif // ORTYOLONET_H
