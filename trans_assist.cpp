#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <sstream>

#include "cnnspp_spotter.h"

using namespace std;
using namespace cv;

#define NONE 0
#define UP 1
#define DOWN 2

#define NUM_TO_HIGHLIGHT 10

#define MIN_WINDOW 32
#define MAX_WINDOW 264
#define STEP 16

struct CallBackParams
{
    int x1,x2,y1,y2,state,curx,cury;
    bool change;
};
bool checkParams(CallBackParams params, int x, int y)
{
    bool toReturn =  params.x1>=0 && params.y1>=0 && 
            params.x1<x && params.y1<y && params.x2<x && params.y2<y;
    if (!toReturn)
        cout<<"Check failed"<<endl;
    return toReturn;
}


void mouseCallBackFunc(int event, int x, int y, int flags, void* page_p)
{

    CallBackParams* params = (CallBackParams*) page_p;
    //x/=params->scale;
    //y/=params->scale;
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
         //cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        params->state=DOWN;
        params->x1=x;
        params->y1=y;
        params->x2=-1;
        params->y2=-1;
        params->change=true;
    }
    else if  ( event == cv::EVENT_LBUTTONUP )
    {
         //cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;

        params->state=UP;
        params->x2=x;
        params->y2=y;
        params->change=true;
    }
    else if  ( event == cv::EVENT_MBUTTONDOWN )
    {
         //cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if ( event == cv::EVENT_MOUSEMOVE )
    {
         //cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

        params->curx=x;
        params->cury=y;

        params->change=true;
    }
}
void mouseCallBackFunc2(int event, int x, int y, int flags, void* page_p)
{

    CallBackParams* params = (CallBackParams*) page_p;
    //x/=params->scale;
    //y/=params->scale;
    if  ( event == cv::EVENT_LBUTTONUP )
    {
         //cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;

        params->state=UP;
        params->x1=x;
        params->y1=y;
        params->x2=x;
        params->y2=y;
        params->change=true;
    }
}

void debug(const multimap<float, SubwordSpottingResult >& res)
{
    set<int> ids;
    for (auto iter=res.begin(); iter!=res.end(); iter++)
    {
        int id=iter->second.imIdx;
        assert(ids.find(id)==ids.end());
        ids.insert(id);
    }
}

map<int,SubwordSpottingResult> drawRes(const multimap<float, SubwordSpottingResult >& res, const map<string,Rect>& imageLocs, const vector<tuple<int,int,int,int,string> >& words, float resizeScale, const set<int>& searchedWords, Mat& draw, int bestWord, int xStart, int xEnd, int tly, int bry, int bxOff, int byOff, int N)
{
    float best = res.begin()->first;
    float worst = res.rbegin()->first;
    int sumH=0;
    int maxW=0;
    auto iter=res.begin();
    for (int i=0; i<N+1; i++, iter++)
    {
        //Mat roi = draw(Rect((get<0>(words[res[i].imIdx])+res[i].startX)*resizeScale,
        //                  get<1>(words[res[i].imIdx])*resizeScale,
        //                  (res[i].endX-res[i].startX+1)*resizeScale,
        //                  (get<3>(words[iter->second.imIdx])-get<1>(words[iter->second.imIdx])+1)*resizeScale));
        Rect thisLoc = imageLocs.at(get<4>(words[iter->second.imIdx]));
        int xOff = thisLoc.x;
        int yOff = thisLoc.y;
        Mat roi = draw(Rect((get<0>(words[iter->second.imIdx])+iter->second.startX)*resizeScale+xOff,
                          get<1>(words[iter->second.imIdx])*resizeScale+yOff,
                          (iter->second.endX-iter->second.startX+1)*resizeScale,
                          (get<3>(words[iter->second.imIdx])-get<1>(words[iter->second.imIdx])+1)*resizeScale));
        if (i<N/2+1)
        {
            sumH+=(get<3>(words[iter->second.imIdx])-get<1>(words[iter->second.imIdx])+1)*resizeScale +1;
            maxW=max(maxW,(int)((get<2>(words[iter->second.imIdx])-get<0>(words[iter->second.imIdx])+1)*resizeScale));
        }
        //cout<<
        Scalar highlightColor;
        if (searchedWords.find(iter->second.imIdx) != searchedWords.end())
            highlightColor=Scalar(210, 140, 140);
        else
            highlightColor=Scalar(0, 195, 195);
        Mat color(roi.size(), CV_8UC3, highlightColor);
        float alpha = 0.5*(iter->second.score-worst)/(best-worst);
        addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi);

    }

    rectangle(draw,Point((get<0>(words[bestWord])+xStart)*resizeScale+bxOff,tly*resizeScale+byOff),Point((get<0>(words[bestWord])+xEnd)*resizeScale+bxOff,bry*resizeScale+byOff),Scalar(0,24,230),1);

    map<int,SubwordSpottingResult> topLocations;
    Mat resWindow = Mat::zeros(sumH,maxW,CV_8UC3);
    int curR=0;
    iter=res.begin();
    for (int i=0; i<N/2+1; i++, iter++)
    {
        if (searchedWords.find(iter->second.imIdx) != searchedWords.end())
            continue;
        Rect thisLoc = imageLocs.at(get<4>(words[iter->second.imIdx]));
        int xOff = thisLoc.x;
        int yOff = thisLoc.y;
        Mat roi = draw(Rect(get<0>(words[iter->second.imIdx])*resizeScale+xOff,
                          get<1>(words[iter->second.imIdx])*resizeScale+yOff,
                          (get<2>(words[iter->second.imIdx])-get<0>(words[iter->second.imIdx])+1)*resizeScale,
                          (get<3>(words[iter->second.imIdx])-get<1>(words[iter->second.imIdx])+1)*resizeScale));
        roi.copyTo(resWindow(Rect(0,curR,roi.cols,roi.rows)));
        curR+=1+roi.rows;
        topLocations[curR]=iter->second;
        //cout<<"topRes["<<curR<<"]: ("<<iter->second.imIdx<<") "<<get<4>(words[iter->second.imIdx])<<endl;
    }

    imshow("page",draw);
    //int dispX = (tlx+brx)/2 - maxW/2;
    //int dispY = bry+2;
    //moveWindow("res", dispX,dispY);
    imshow("res",resWindow);
    return topLocations;
}


int main(int argc, char** argv)
{
    if (argc<2)
    {
        cout<<"Usage: "<<argv[0]<<" imageDir pageImageList.txt gt.gtp featurizer.prototxt embedder.prototxt trained.caffemodel saveDir height width gpu"<<endl;
        exit(1);
    }

    string imageDir = argv[1];
    if (imageDir[imageDir.length()-1]!='/')
        imageDir+="/";
    string imageListFile = argv[2];
    string gtFile = argv[3];
    string featFile = argv[4];
    string embdFile = argv[5];
    string wghtFile = argv[6];
    string saveDir = argv[7];
    if (saveDir[saveDir.length()-1]!='/')
        saveDir+="/";
    int height = atoi(argv[8]);
    int width = atoi(argv[9]);
    int gpu = atoi(argv[10]);

    string line;
    //vector<string> imageList;
    //vector<Mat> images;
    map<string,Mat> images;
    ifstream imageListRead(imageListFile);
    while (getline(imageListRead,line))
    {
        int li = line.find_last_of('/');
        //string imageName;
        //if (li != string::npos)
        //    imageName = line.substr(li+1);
        //else
        //    imageName = line;
        //imageList.push_back(imageName);
        //images.push_back(imread(line,0));
        images[line]=imread(imageDir+line);//read BGR
    }
    int N = NUM_TO_HIGHLIGHT*images.size();

    vector<tuple<int,int,int,int,string> > words;
    ifstream gt(gtFile);
    while (getline(gt,line))
    { 
        stringstream ss(line);
        string part;
        getline(ss,part,' ');

        string pathIm=string(part);
        //int li = pathIm.find_last_of('/');
        //string imageName;
        //if (li != string::npos)
        //    imageName = pathIm.substr(li+1);
        //else
        //    imageName = pathIm;
        for (auto& p : images)
            if (p.first.compare(pathIm)==0)
            {
        
                getline(ss,part,' '); 
                int x1=max(0,stoi(part));//;-1;
                getline(ss,part,' '); 
                int y1=max(0,stoi(part));//;-1;
                getline(ss,part,' ');
                int x2=min(p.second.cols-1,stoi(part));//;-1;
                getline(ss,part,' ');
                int y2=min(p.second.rows-1,stoi(part));//;-1;

                words.emplace_back(x1,y1,x2,y2,pathIm);
                break;
            }
    }
    gt.close();

    CNNSPPSpotter spotter(featFile,embdFile,wghtFile,"none",gpu,true,0.25,4,saveDir+"cnnspp_spotter");
    //string imageDir = "./";
    //int loc = imageFile.rfind('/');
    //if (loc != string::npos)
    //    imageDir = imageFile.substr(0,loc);
    //GWDataset dataset(gtFile,imageDir);
    GWDataset dataset(words,imageDir);
    spotter.setCorpus_dataset(&dataset);
    //spotter.getCorpusFeaturization();
    for (int window=MIN_WINDOW; window<=MAX_WINDOW; window+=STEP)
        spotter.getEmbedding(window);
   
    int pageH=0;//pages.begin()->second->getImg()->rows;
    int pageW=0;//pages.begin()->second->getImg()->cols;
    for (auto p : images)
    {
        pageH = max(pageH,p.second.rows);
        pageW = max(pageW,p.second.cols);
    }
    float resizeScale=.001;
    int nAcross, nDown;
    for (float scale=0.5; scale>.001; scale-=.001)
    {
        nAcross=floor(width/(pageW*scale));
        nDown=floor(height/(pageH*scale));
        if (nAcross*nDown>=images.size())
        {
            resizeScale=scale;
            break;
        }
    }
    cv::Mat draw = cv::Mat::zeros(height,width,CV_8UC3);
    int xPos=0;
    int yPos=0;
    int across=0;
    map<string,Rect> imageLocs;
    for (auto& p : images)
    {
    //cv::resize(workingIm,workingIm,cv::Size(workingIm.cols*resizeScale,workingIm.rows*resizeScale));
        //cout <<"page dims: "<<workingIm.rows<<", "<<workingIm.cols<<"  at: "<<xPos<<", "<<yPos<<endl;
        //cv::imshow("test",workingIm);
        //cv::waitKey();
        cv::Mat workingIm;
        resize(p.second,workingIm,cv::Size(),resizeScale,resizeScale);   
        assert(xPos>=0 && yPos>=0 && xPos+workingIm.cols<=draw.cols && yPos+workingIm.rows<=draw.rows);
        workingIm.copyTo(draw(cv::Rect(xPos,yPos,workingIm.cols,workingIm.rows)));
        imageLocs[p.first] = Rect(xPos,yPos,workingIm.cols,workingIm.rows);
        xPos+=workingIm.cols;
        if (++across >= nAcross)
        {
            xPos=0;
            yPos+=workingIm.rows;
            across=0;
        }
    }
 
    //float scale = height/(0.0+im.rows);
    //cv::resize(im,im, cv::Size(), scale,scale);
    //for (auto& p : images)
    //{
    //    cv::resize(p.second,p.second,cv::Size(), resizeScale,resizeScale);
    //}
    cv::namedWindow("page");
    CallBackParams params;
    params.state=NONE;
    params.change=false;
    cv::setMouseCallback("page", mouseCallBackFunc, &params);

    namedWindow("res");
    CallBackParams params2;
    params2.state=NONE;
    params2.change=false;
    setMouseCallback("res", mouseCallBackFunc2, &params2);

    //Mat draw;
    //cvtColor(im,draw,CV_GRAY2BGR);
    multimap<float, SubwordSpottingResult > res;
    set<int> searchedWords;
    map<int,SubwordSpottingResult> topResByLoc;
    int xStart,xEnd,tly,bry, bxOff, byOff, bestWord;

    Mat drawOrig = draw.clone();
    while(1)
    {
        cv::imshow("page",draw);
        cv::waitKey(4);
        if (params.change ||  params2.change)
        {

            if (params.state==DOWN)
            {
                res.clear();
                searchedWords.clear();
                topResByLoc.clear();
                //cvtColor(im,draw,CV_GRAY2BGR);
                if (checkParams(params,draw.rows*2,draw.cols*2))
                {
                    draw=drawOrig.clone();
                    rectangle(draw,Point(params.x1,params.y1),Point(params.curx,params.cury),Scalar(0,24,230),1);
                }
            }
            else if (params.state==UP)
            {
                //cvtColor(im,draw,CV_GRAY2BGR);
                draw=drawOrig.clone();
                if (params.x1==params.x2 || params.y1==params.y2 || !checkParams(params,draw.rows*2,draw.cols*2))
                {
                    params.x1=params.x2=params.y1=params.y2=-1;
                }
                else
                {
                    string imageName;
                    Rect loc;
                    //cout<<"params.x1="<<params.x1<<", params.x2="<<params.x2<<endl;
                    //cout<<"params.y1="<<params.y1<<", params.y2="<<params.y2<<endl;
                    for (auto p : imageLocs)
                    {
                        //cout<<"p.second.x="<<p.second.x<<", p.second.x(2)="<<p.second.x+p.second.width<<endl;
                        //cout<<"p.second.y="<<p.second.y<<", p.second.y(2)="<<p.second.y+p.second.height<<endl;
                        if (    params.x1>=p.second.x && params.x1<p.second.x+p.second.width &&
                                params.x2>=p.second.x && params.x2<p.second.x+p.second.width &&
                                params.y1>=p.second.y && params.y1<p.second.y+p.second.height &&
                                params.y2>=p.second.y && params.y2<p.second.y+p.second.height )
                        {
                            imageName=p.first;
                            loc=p.second;
                            break;
                        }
                    }
                    //cout<<"on image "<<imageName<<endl;
                    if (imageName.length()==0)
                    {
                        params.state=NONE;
                        params.change=false;
                        continue;
                    }
                    int tlx = (min(params.x1,params.x2)-loc.x)/resizeScale+1;
                    tly = (min(params.y1,params.y2)-loc.y)/resizeScale+1;
                    int brx = (max(params.x1,params.x2)-loc.x)/resizeScale-1;
                    bry = (max(params.y1,params.y2)-loc.y)/resizeScale-1;
                    //find word
                    bestWord=-1;
                    float bestDistSum=999999;
                    for (int i=0; i<words.size(); i++)
                    {
                        if (get<4>(words[i]).compare(imageName)==0)
                        {
                            float distSum = 0;
                            if (tlx<get<0>(words[i]) || tlx>get<2>(words[i]))
                                distSum += pow(tlx-get<0>(words[i]),2);
                            if (tly<get<1>(words[i]) || tly>get<3>(words[i]))
                                distSum += pow(tly-get<1>(words[i]),2);
                            if (brx<get<0>(words[i]) || brx>get<2>(words[i]))
                                distSum += pow(brx-get<2>(words[i]),2);
                            if (bry<get<1>(words[i]) || bry>get<3>(words[i]))
                                distSum += pow(bry-get<3>(words[i]),2);
                            if (distSum<bestDistSum)
                            {
                                bestDistSum=distSum;
                                bestWord=i;
                            }
                        }
                    }
                    if (bestWord==-1)
                    {
                        params.state=NONE;
                        params.change=false;
                        continue;
                    }
                    ///
                    //cout<<"Best match was: "<<dataset.labels()[bestWord]<<"["<<bestWord<<"]"<<endl;
                    ///
                    tly = get<1>(words[bestWord]);
                    bry = get<3>(words[bestWord]);
                    string bName = get<4>(words[bestWord]);
                    bxOff = imageLocs[bName].x;
                    byOff = imageLocs[bName].y;
                    xStart = max(tlx-get<0>(words[bestWord]),0);
                    xEnd = min(brx-get<0>(words[bestWord]),get<2>(words[bestWord])-get<0>(words[bestWord]));
                    xStart += (xStart%STEP)<(STEP/2)?-1*(xStart%STEP):STEP-(xStart%STEP);
                    xEnd += (xEnd%STEP)<=(STEP/2)?-1*(xEnd%STEP):STEP-(xEnd%STEP);
                    if (xEnd-xStart<MIN_WINDOW)
                    {
                        int dif = (MIN_WINDOW-(xEnd-xStart))/4;
                        xStart -= STEP*(dif/2);
                        xEnd += STEP*(dif/2);
                        if (dif%2==1)
                        {
                            if (xStart >tlx-get<0>(words[bestWord]))
                                xStart-=STEP;
                            else
                                xEnd+=STEP;
                        }
                    }
                    else if (xEnd-xStart>MAX_WINDOW)
                    {
                        int dif = ((xEnd-xStart)-MAX_WINDOW)/4;
                        xStart += STEP*(dif/2);
                        xEnd -= STEP*(dif/2);
                        if (dif%2==1)
                        {
                            if (xStart <tlx-get<0>(words[bestWord]))
                                xStart+=STEP;
                            else
                                xEnd-=STEP;
                        }
                    }

                    if (res.size()==0)
                    {
                        spotter.getEmbedding(xEnd-xStart);
                        vector< SubwordSpottingResult > newRes = spotter.subwordSpot(bestWord,xStart,0.3,xEnd-xStart);
                        searchedWords.insert(bestWord);
                        for (SubwordSpottingResult r : newRes)
                            res.emplace(r.score,r);
                    }
                    topResByLoc = drawRes(res, imageLocs, words, resizeScale, searchedWords, draw, bestWord, xStart, xEnd,tly, bry, bxOff, byOff, N);
                }

                params.state=NONE;
            }
            else if (params2.state==UP)
            {
                if (checkParams(params2,3000,3000) && topResByLoc.size()>0) 
                {
                    draw=drawOrig.clone();
                    //which pressed?
                    auto iter = topResByLoc.begin();
                    while (iter!= topResByLoc.end() && iter->first<params2.y1)
                        iter++;
                    if (iter != topResByLoc.end())
                    {
                        //cout<<"selected ("<<iter->second.imIdx<<") "<<get<4>(words[iter->second.imIdx])<<endl;
                        int wordIdx = iter->second.imIdx;
                        int startX = iter->second.startX;
                        int endX = iter->second.endX;
                        startX += (startX%STEP)<(STEP/2)?-1*(startX%STEP):STEP-(startX%STEP);
                        endX += (endX%STEP)<=(STEP/2)?-1*(endX%STEP):STEP-(endX%STEP);
                        assert(startX>=0);
                        if (endX-startX<MIN_WINDOW)
                        {
                            int dif = (MIN_WINDOW-(endX-startX))/4;
                            startX -= STEP*(dif/2);
                            endX += STEP*(dif/2);
                            if (startX<0)
                            {
                                endX-=startX;
                                startX=0;
                            }
                            if (dif%2==1)
                            {
                                if (startX-STEP>=0)
                                    startX-=STEP;
                                else
                                    endX+=STEP;
                            }
                        }
                        else if (endX-startX>MAX_WINDOW)
                        {
                            int dif = ((endX-startX)-MAX_WINDOW)/4;
                            startX += STEP*(dif/2);
                            endX -= STEP*(dif/2);
                            if (dif%2==1)
                            {
                                //if (endX <tlx-get<0>(words[bestWord]))
                                    startX+=STEP;
                                //else
                                //    endX-=STEP;
                            }
                        }

                        //new spotting
                        spotter.getEmbedding(endX-startX);
                        vector< SubwordSpottingResult > newRes = spotter.subwordSpot(wordIdx,startX,0.3,endX-startX);
                        searchedWords.insert(wordIdx);

                        //merge results
                        for (SubwordSpottingResult r : newRes)
                        {
                            bool matchFound=false;
                            for (auto iter=res.begin(); iter!=res.end(); iter++)
                            {
                                if (r.imIdx == iter->second.imIdx)
                                {
                                    assert((0.0+min(r.endX,iter->second.endX)-max(r.startX,iter->second.startX)/max(r.endX-r.startX,iter->second.endX-iter->second.startX))>0.4); //UPDATE_OVERLAP_THRESH
                                //{
                                    if (matchFound)
                                    {
                                        res.erase(iter);
                                    }
                                    else
                                    {
                                        matchFound=true;
                                        if (r.score < iter->second.score)
                                        {
                                            res.erase(iter);
                                            res.emplace(r.score,r);
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!matchFound)
                            {
                                res.emplace(r.score,r);
                            }
                            //debug(res);
                        }

                        topResByLoc = drawRes(res, imageLocs, words, resizeScale, searchedWords, draw, bestWord, xStart, xEnd, tly, bry, bxOff, byOff, N);
                    }
                }
                params2.state=NONE;
            }
            params.change=false;
            params2.change=false;
        }
    }
}
