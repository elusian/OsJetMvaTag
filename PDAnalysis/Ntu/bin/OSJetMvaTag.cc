#include <stdexcept>
#include <fstream>

#include <TGraph.h>
#include <TF1.h>

#include <Math/Vector4D.h>

using namespace std;

OSJetMvaTag::OSJetMvaTag():
	OSJetIndex_(-1),
	OSJetTagDecision_(0), 
	OSJetTagMvaValue_(-1.), 
	OSJetTagMistagProbRaw_(-1.), 
	OSJetTagMistagProbCalProcess_(-1.), 
	OSJetTagMistagProbCalProcessBuBs_(-1.), 
	dzCut_(1.), 
	nMuonsSel_(0)
{}

OSJetMvaTag::~OSJetMvaTag() {}

// =====================================================================================
void OSJetMvaTag::inizializeOSJetTagVars()
{
	ssIndex_ = -1;
	pvIndex_ = -1;
	OSJetIndex_ = -1;
	OSJetTagDecision_ = 0;
	OSJetTagMvaValue_ = -1;
	OSJetTagMistagProbRaw_ = -1;
	OSJetTagMistagProbCalProcess_ = -1;
	OSJetTagMistagProbCalProcessBuBs_ = -1;
}

void OSJetMvaTag::setOSJetDzCut(float dzCut = 1.)
{
	dzCut_ = dzCut;
}

void OSJetMvaTag::inizializeOSJetMvaReader(
	TString methodName
,	TString methodPath
,	TString normFileName
,	TString normFilePath
	)
{
	TString graph_path = methodPath + methodName;
	Status load_graph_status = load_graph(graph_path);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		throw runtime_error("OSJetMvaTag: Unable to open dnn graph");
	}
	TString norm_path = normFilePath + normFileName;
	
	NormalizationManager norm;
	norm.setNorm(norm_path);
	
#define initNormVar(var) \
	var##Offset = norm.variables[#var].offset; \
	var##Scale = norm.variables[#var].scale;
	
	initNormVar(trkPt)
	initNormVar(trkEta)
	initNormVar(trkPhi)
	initNormVar(trkDxySigned)
	initNormVar(trkDz)
	initNormVar(trkExy)
	initNormVar(trkEz)
	initNormVar(trkDrJet)
	initNormVar(jetPt)
	initNormVar(jetEta)
	initNormVar(jetPhi)
	initNormVar(jetProbb)
	initNormVar(jetDrB)
	initNormVar(ssbPt)
	initNormVar(ssbEta)
	initNormVar(ssbPhi)
#undef initNormVar
}

bool OSJetMvaTag::inizializeOSJetCalibration( 
	TString process = "BuJPsiKData2018"
,   TString processBuMC = "BuJPsiKMC2018"
,   TString processBsMC = "BsJPsiPhiMC2018"
,   TString methodPath = ""  
)
{
	if(methodPath == "") methodPath = methodPath_;
	auto *f   = new TFile(methodPath + "OSJetTaggerCalibration" + process + ".root");
	auto *fBu = new TFile(methodPath + "OSJetTaggerCalibration" + processBuMC + ".root");
	auto *fBs = new TFile(methodPath + "OSJetTaggerCalibration" + processBsMC + ".root");

	if(f->IsZombie()){ cout<<"f IsZombie"<<endl;return false; }
	if(fBu->IsZombie()){ cout<<"fBu IsZombie"<<endl;return false; }
	if(fBs->IsZombie()){ cout<<"fBs IsZombie"<<endl;return false; }

	wCalProcess_ = (TF1*)f->Get("OSJetCal");
	wCalBuMC_	= (TF1*)fBu->Get("OSJetCal");
	wCalBsMC_	= (TF1*)fBs->Get("OSJetCal");

	wCalBuBs_ = new TF1("OSJetCalBuBs","[0]-[1]*[2]/[3]+[2]/[3]*x",0.,1.);
	float qs = wCalBsMC_->GetParameter(0);
	float ms = wCalBsMC_->GetParameter(1);
	float qu = wCalBuMC_->GetParameter(0);
	float mu = wCalBuMC_->GetParameter(1);
	wCalBuBs_->SetParameters(qs, qu, ms, mu);

	delete f;
	delete fBu;
	delete fBs;
	return true;  
}

bool OSJetMvaTag::makeOSJetTagging()
{
	if(ssIndex_ < 0){ cout<<"SS NOT INITIALIZED"<<endl; return 1; }
	selectOSJet();
	OSJetTagDecision_ = -1*trkCharge->at(OSJetTrackIndex_); 

	computeOSJetTagVariables();
	
	auto out = session_->Run(inputs_, {"ouput/Sigmoid"}, &outputs_);
	
	OSJetTagMvaValue_ = out[0].flat().(0);
	OSJetTagMistagProbRaw_ = 1 - OSJetTagMvaValue_;
	OSJetTagMistagProbCalProcess_ = wCalProcess_->Eval(OSJetTagMistagProbRaw_);
	OSJetTagMistagProbCalProcessBuBs_ = wCalBuBs_->Eval(OSJetTagMistagProbCalProcess_);

	return 0;
}

PxPyPzEVector OSJetMvaTag::get4VecFromJPsiX(int iSvt)
{
	int iJPsi = subVtxFromSV(iSvt)[0];
	vector<int>& tkJpsi = tracksFromSV(iJPsi);
	vector<int>& tkSsB = tracksFromSV(iSvt);
	
	PxPyPzEVector t(0,0,0,0);
	
	for(auto iSvTrk: tkSsB)
	{
		// this is broken for BdJPsiKst
		float mass = MassK;
		if (iSvTrk == tkJpsi[0] or iSvTrk == tkJpsi[1]) m = MassMu;
		PxPyPzMVector a( trkPt->at(iSvTrk), trkEta->at(iSvTrk), trkPhi->at(iSvTrk), m );
		t += a;
	}

	return t;
}

int OSJetMvaTag::selectOSJet()
{
	vector<int>& tkSsB = tracksFromSV(ssIndex_);
	auto tB = get4VecFromJPsiX(ssIndex_);
	
	float bestJetTag = cutBTag;
	
	for (int iJet = 0; iJet < nJets; ++iJet)
	{
		if (goodJet(iJet) != true) continue;
		if (abs(jetEta->at(iJet)) > 2.5) continue;
		if (jetPt->at(iJet) < minPtJet) continue;

		float bTag = GetJetProbb(iJet);
		if (bTag < cutBTag_ ) continue;

		if ( deltaR(jetEta->at(iJet), jetPhi->at(iJet), tB.Eta(), tB.Phi()) < jetDrCut_ ) continue;

		vector<int>& jet_tks = tracksFromJet( iJet );

		bool skip = false; 
		for (auto it: jet_tks)
		{
			if(std::find(tkSsB.begin(),tkSsB.end(),it) != tkSsB.end()){
				skip = true;
				break;
			}
		}
		if (skip) continue; // skip jet if contains signal tracks

		int nTrkNearPV = 0;
		for (auto it: jet_tks)
		{
			if(fabs(dZ(it, ssbPVT)) >= jetDzCut_) continue;
			if( !isTrkHighPurity(it) ) continue;
			nTrkNearPV++;
		}

		if (nTrkNearPV < 2) continue; // skip jet if less than 2 tracks are near the PV

		if (bTag > bestJetTag)
		{ // take the jet with the highest btag prob
			bestJetTag = bTag;
			OSJetIndex_ = iJet;
		}
	}
	
	return OSJetIndex_;
}

void OSJetMvaTag::computeOSJetTagVariables()
{
	auto nValidTrks = trkPt->end() - trkPt->begin();
	
	// index list, this will be sorted
	// numpy code ~= 
	// 	indexes = trkPt.argsort()
	std::vector<int> indexes(nValidTrks);
	iota(indexes.begin(), indexes.end(), 0);
	
	// filter index by accepted events
	// filtering by index is easier considering we have parallel vectors
	indexes.erase(
		remove_if(
			indexes.begin(), 
			indexes.end(), 
			[this](int iTrk)
			{ 
				return !this->isAccepted(iTrk);
			}
		), 
		indexes.end()
	);
	
	
	sort(
		indexes.begin(), 
		indexes.end(), 
		[this](int i1, int i2)
		{
			return this->ptOrder(i1, i2);
		}
	);
	
#define getRescVar(var, value) \
	(value + var##Offset)*var##Scale
	
	// copied from DeepJet
	// works because the tensor is column major
	auto trk_input = inputs[0].second;
	auto trkTensor = trk_input.tensor<float, 3>();
	trkTensor.setZero();
	auto charge_input = inputs[1].second;
	auto chargeTensor = charge_input.tensor<float, 3>();
	chargeTensor.setZero();
	size_t fillSize = min(25l, nValidTrks);
	size_t zeroedSize = maxtracks - fillSize;
	for (int iTrk = 0; iTrk  < fillSize; iTrk++)
	{
		size_t fillIndex = iTrk + zeroedSize;
		//clog << fillIndex << endl;
		
		auto trkRow = &trkTensor(0, fillIndex, 0);
		auto chargeRow = &chargeTensor(0, fillIndex, 0);
		
		auto trkIndex = indexes[iTrk];
		
		float dxy = dSign(it, jetPx->at(iJet), jetPy->at(iJet))*abs(trkDxy->at(it));
		float dz = dZ(it, ssbPVT);
		float drJet = deltaR(jetEta->at(iJet), jetPhi->at(iJet), trkEta->at(it), trkPhi->at(it));
		
		*(  trkRow) = getRescVar(trkPt, (*trkPt)[trkIndex]);
		*(++trkRow) = getRescVar(trkEta, (*trkEta)[trkIndex]);
		*(++trkRow) = getRescVar(trkPhi, (*trkPhi)[trkIndex]);
		*(++trkRow) = getRescVar(trkDxySigned, dxy);
		*(++trkRow) = getRescVar(trkDz, dz);
		*(++trkRow) = getRescVar(trkExy, (*trkExy)[trkIndex]);
		*(++trkRow) = getRescVar(trkEz, (*trkEz)[trkIndex]);
		*(++trkRow) = getRescVar(trkDrJet, drJet);
		
		*chargeRow = (*trkCharge)[trkIndex];
	}
	
	auto tB = get4VecFromJPsiX(ssIndex_);
	
	auto jet_input = inputs[2].second;
	auto jetTensor = jet_input.tensor<float, 2>();
	jetTensor.setZero();
	auto jetRow = &jetTensor(0, 0);
	
	*(  jetRow) = getRescVar(jetPt, (*jetPt)[OSJetIndex_]);
	*(++jetRow) = getRescVar(jetEta, (*jetPt)[OSJetIndex_]);
	*(++jetRow) = getRescVar(jetPhi, (*jetPt)[OSJetIndex_]);
	*(++jetRow) = getRescVar(jetProbb, GetJetProbb(OSJetIndex_););
	*(++jetRow) = getRescVar(deltaR(jetEta->at(iJet), jetPhi->at(iJet), tB.Eta(), tB.Phi()););
	
	auto ssb_input = inputs[3].second;
	auto ssbTensor = ssb_input.tensor<float, 2>();
	ssbTensor.setZero();
	auto ssbRow = &ssbTensor(0, 0);
	
	*(  ssbRow) = getRescVar(ssbPt, tB.Pt());
	*(++ssbRow) = getRescVar(ssbEta, tB.Eta());
	*(++ssbRow) = getRescVar(ssbPhi, tB.Phi());
#undef getRescVar
}

bool OSJetMvaTag::isAccepted(int iTrk)
{
	vector<int>& jet_tks = tracksFromJet( OSJetIndex_ );
	
	bool accept = isTrkHighPurity(it);
	
	if(std::find(jet_tks.begin(), jet_tks.end(), it) == jet_tks.end())
	{
		accept = accept and (fabs(dZ(it, ssbPVT)) < jetDzCut);
		accept = accept and (deltaR(jetEta->at(OSJetIndex_), jetPhi->at(OSJetIndex_), trkEta->at(iTrk), trkPhi->at(iTrk)) < 0.5);
	}
	
	accept = accept and !isnan(trkDxySigned[iTrk]) and !isinf(trkEz[iTrk]) and trkDz[iTrk] < 1;
	
	return accept;
}

// utility to order indexes by growing Pt
bool OSJetMvaTag::ptOrder(int iTrk1, int iTrk2)
{
	return trkPt[iTrk1] < trkPt[iTrk2];
}

// taken from TF example
tf::Status OSJetMvatag::load_graph(TString graph_file_name)
{
	tf::GraphDef graph_def;
	tf::Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}
	
	// WARNING this locks tensorflow in the least possible number of thread, to reduce overhead
	// if parallelism is ever introduced this should be removed
	graph = graph_def;
	tf::SessionOptions options;
	tf::ConfigProto& config = options.config;
	config.set_inter_op_parallelism_threads(1);
	config.set_intra_op_parallelism_threads(1);
	config.set_use_per_session_threads(true);

	session.reset(tensorflow::NewSession(options));
	tf::Status session_create_status = session->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return tf::Status::OK();
}

//==================================================================================================================

void OSJetMvatag::NormalizationManager::setNorm(TString norm_file)
{
	ifstream norm_stream(norm_file);
	
	string var;
	float offset;
	float scale;
	while(norm_stream >> var >> offset >> scale)
	{
		variables[var].offset = offset;
		variables[var].scale = scale;
		clog << var << " " << offset << " " << scale << endl;
	}
}
