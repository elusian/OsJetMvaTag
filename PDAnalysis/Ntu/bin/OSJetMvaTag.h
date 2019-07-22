#ifndef OSJetMvaTag_H
#define OSJetMvaTag_H

#include <vector>
#include <set>
#include <map>
#include <string>

#include "PDAnalyzerUtil.h"

#include "TString.h"

#include <tensorflow/core/framework/tensor.h>

#include <Math/Vector4Dfwd.h>

class TF1;

namespace tf = tensorflow;

class OSJetMvaTag : public virtual PDAnalyzerUtil,
	public virtual AlbertoUtil
{
public:
	OSJetMvaTag();
	~OSJetMvaTag();

	void inizializeOSJetTagVars();

	bool makeOSJetTagging();
	int selectOSJet(); 

	int getOSJetTag(){ return OSJetTagDecision_; }
	float getOSJetTagMvaValue(){ return OSJetTagMvaValue_; }
	float getOSJetTagMistagProbRaw(){ return OSJetTagMistagProbRaw_; }
	float getOSJetTagMistagProbCalProcess(){ return OSJetTagMistagProbCalProcess_; }
	float getOSJetTagMistagProbCalProcessBuBs(){ return OSJetTagMistagProbCalProcessBuBs_; }

	void setVtxOSJetTag(int iB, int iPV) { ssIndex_ = iB; pvIndex_ = iPV;}
	void setOSJetDzCut(float dzCut);
	void inizializeOSJetMvaReader(TString, TString);
	bool inizializeOSJetCalibration(TString process, TString processBuMC, TString processBsMC, TString methodPath);

private:
	struct VarNorm
	{
		float scale;
		float offset;
	}
	struct NormalizationManager
	{
		NormalizationManager(){};
		
		void setNorm(TString norm_file)
		
		map<string, VarNorm> variables;
	};
	
	TString methodNameFromWeightName();
	void computeOSJetTagVariables();
	

	std::unique_ptr<tf::Session> session_;

	std::unordered_map<std::string, tf::Tensor> inputs_;
	std::vector<tf::Tensor> outputs_;
	
// I won't write all of that explicitly
#define declareNormVar(var) \
	float var##Offset; \
	float var##Scale;
	
	declareNormVar(trkPt)
	declareNormVar(trkEta)
	declareNormVar(trkPhi)
	declareNormVar(trkDxySigned)
	declareNormVar(trkDz)
	declareNormVar(trkExy)
	declareNormVar(trkEz)
	declareNormVar(trkDrJet)
	declareNormVar(jetPt)
	declareNormVar(jetEta)
	declareNormVar(jetPhi)
	declareNormVar(jetDrB)
	declareNormVar(jetProbb)
	declareNormVar(ssbPt)
	declareNormVar(ssbEta)
	declareNormVar(ssbPhi)
#undef declareNormVar

	int ssIndex_;
	int pvIndex_;
	int OSJetIndex_;
	int OSJetTagDecision_;

	float OSJetTagMvaValue_;
	float OSJetTagMistagProbRaw_;
	float OSJetTagMistagProbCalProcess_;
	float OSJetTagMistagProbCalProcessBuBs_;

	float trkDzCut_;
	float PFIsoCut_;
	float jetDrCut_;
	float jetBTagCut_;
	float jetDzCut_;
	
	tf::Tensor& trk_input_;
	tf::Tensor& charge_input_;
	tf::Tensor& jet_input_;
	tf::Tensor& ssb_input_;

	//MISTAG VARIABLES
	TF1 *wCalProcess_;
	TF1 *wCalBuMC_;
	TF1 *wCalBsMC_;
	TF1 *wCalBuBs_;
	
	bool ptOrder(int iTrk1, int iTrk2);	// pt ordering for trk indexes
	bool isAccepted(int iTrk);		// check acceptance of a track
	
	tf::Status load_graph(TString graph_file_name);
	
	static size_t maxtracks = 25;
	
	// while refactoring it inside the utils
	PxPyPzEVector get4VecFromJPsiX(int iSvt);
};

#endif
