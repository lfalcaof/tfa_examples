// make_time_dependent_acceptance_v3.C
// =====================================================================
// Builds a time-dependent acceptance from a TH3F/TH3D:
//   - Z slices (t/τ) -> TH2D per slice (XY = m2(ππ) vs cosθ)
//   - TF2 fit per slice: polynomial basis x^i y^j (orders polyNx, polyNy)
//   - exports coefficients to CSV (consolidated + per-slice)
//   - saves zEdges (TVectorD), slice TH2Ds, and fitted TF2s to out_file
//   - provides helpers to reconstruct a TF3 from the TF2s and to sample
//     into a TH3F (visualization/validation without OpenGL)
// Defaults:
//   * normalize_by_max = false  (does not normalize slices)
//   * linear_z = false          (stepwise in t/τ)
//   * linear_xy = false         (use bin value for acc3_sampled)
//   * swap_xy_input = true      (uses Project3D("yx") => X=m2, Y=cos)
// =====================================================================


// How to run:

// root -l -q 'make_time_dependent_acceptance_v3.C+(
//  "final_eijk_f_y.root",
// "epsilon_D0_Hlt1TwoTrackMVADecision_TOS_y_0.0065_block_4",
//  "acc_time_dep_v3.root",
//  10,      /* nslices_z */
//  false,   /* normalize_by_max */
//  false,   /* linear_xy  */
//  false,   /* linear_z   */
//  true,    /* fit_slices */
//  2, 2,    /* polyNx, polyNy */
//  "poly",
//  60,60,40,
//  true     /* swap_xy_input: usa Project3D("yx") -> X=m2, Y=cos */
//  )'



#include <TFile.h>
#include <TH3.h>
#include <TH2.h>
#include <TF2.h>
#include <TF3.h>
#include <TString.h>
#include <TMath.h>
#include <TVectorD.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TStyle.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>

// ============================
// Aux: polinomial base for the TF2
// ============================
TString BuildPolyForm(int Nx,int Ny) {
  TString form; int k=0;
  for (int i=0;i<=Nx;++i)
    for (int j=0;j<=Ny;++j,++k) {
      TString term = TString::Format("[%d]*pow(x,%d)*pow(y,%d)",k,i,j);
      form += (k==0 ? term : " + "+term);
    }
  return form;
}

// =====================================================================
// Simple structure to materialize a 3D sample (acc3_sampled)
// from the TH2D slices (bin value in XY; stepwise in Z)
// =====================================================================
struct TDAcc {
  std::vector<TH2D*> slices;
  std::vector<double> zEdges;
  double xmin,xmax,ymin,ymax,zmin,zmax;
  bool linear_xy=false, linear_z=false;

  double valueXY(const TH2D* h2, double x, double y, bool linear) const {
    if (!h2) return 0.0;
    if (linear) return h2->Interpolate(x,y);
    int ix=h2->GetXaxis()->FindBin(x), iy=h2->GetYaxis()->FindBin(y);
    if (ix<1||ix>h2->GetNbinsX()||iy<1||iy>h2->GetNbinsY()) return 0.0;
    return h2->GetBinContent(ix,iy);
  }
  int findSlice(double z) const {
    if (z<=zEdges.front()) return 0;
    if (z>=zEdges.back())  return (int)slices.size()-1;
    for (size_t s=0;s+1<zEdges.size();++s)
      if (z>=zEdges[s] && z<zEdges[s+1]) return (int)s;
    return (int)slices.size()-1;
  }
  double eval(double x,double y,double z) const {
    if (x<xmin||x>xmax||y<ymin||y>ymax||z<zmin||z>zmax) return 0.0;
    if (!linear_z) {
      int s=findSlice(z);
      return valueXY(slices[s],x,y,linear_xy);
    } else {
      int s=findSlice(z);
      int s2=std::min(s+1,(int)slices.size()-1);
      if (s2==s) return valueXY(slices[s],x,y,linear_xy);
      double z1=zEdges[s], z2=zEdges[s+1];
      double a=valueXY(slices[s],x,y,linear_xy);
      double b=valueXY(slices[s2],x,y,linear_xy);
      double t=(z-z1)/(z2-z1); t=std::max(0.0,std::min(1.0,t));
      return (1.0-t)*a + t*b;
    }
  }
};
static TDAcc* gTD=nullptr;
double TDWrapper(double* xx,double*){ return gTD? gTD->eval(xx[0],xx[1],xx[2]) : 0.0; }

// =====================================================================
// Helper global to TF3 reconstructed following the TF2 fitted
// =====================================================================
struct FitsCtx {
  std::vector<TF2*> fits;
  std::vector<double> zEdges;
  bool linear_z=false;
};
static FitsCtx* gFitsCtx = nullptr;

static inline int findSliceFits(double z, const std::vector<double>& edges, int last) {
  if (z <= edges.front()) return 0;
  if (z >= edges.back())  return last;
  for (size_t i=0;i+1<edges.size();++i)
    if (z >= edges[i] && z < edges[i+1]) return (int)i;
  return last;
}
double FitsWrapper(double* xx, double*) {
  if (!gFitsCtx) return 0.0;
  double x=xx[0], y=xx[1], z=xx[2];
  int last = (int)gFitsCtx->fits.size()-1;
  int s = findSliceFits(z, gFitsCtx->zEdges, last);
  if (!gFitsCtx->linear_z || s==last) {
    return gFitsCtx->fits[s]->Eval(x,y);
  } else {
    int s2 = std::min(s+1, last);
    double z1=gFitsCtx->zEdges[s], z2=gFitsCtx->zEdges[s+1];
    double a = gFitsCtx->fits[s ]->Eval(x,y);
    double b = gFitsCtx->fits[s2]->Eval(x,y);
    double t = (z - z1)/(z2 - z1);
    if (t<0) t=0; if (t>1) t=1;
    return (1.0-t)*a + t*b;
  }
}

// =====================================================================
// Main
// =====================================================================
void make_time_dependent_acceptance_v3(
    const char* in_file = "final_eijk_f_y.root",
    const char* in_hist = "epsilon_D0_Hlt1TwoTrackMVADecision_TOS_y_0.0065_block_4",
    const char* out_file= "acc_time_dep_v3.root",
    int   nslices_z     = 10,
    bool  normalize_by_max = false,   // OFF by default
    bool  linear_xy     = false,      // bin-wise evaluation for acc3_sampled
    bool  linear_z      = false,      // stepwise in Z
    bool  fit_slices    = true,       // fit TF2 per slice and export coefficients
    int   polyNx        = 2,
    int   polyNy        = 2,
    const char* basis   = "poly",     // (placeholder for Legendre if desired in the future)
    int   sample_nx     = 60,
    int   sample_ny     = 60,
    int   sample_nz     = 40,
    bool  swap_xy_input = true        // if true: Project3D("yx") => X=m2, Y=cos
)
{
  std::unique_ptr<TFile> fin(TFile::Open(in_file,"READ"));
  if(!fin || fin->IsZombie()){ std::cerr<<"[ERROR] input "<<in_file<<"\n"; return; }

  TH3F* h3 = dynamic_cast<TH3F*>(fin->Get(in_hist));
  if(!h3){
    TH3D* h3d=dynamic_cast<TH3D*>(fin->Get(in_hist));
    if(h3d){
      h3=new TH3F("h3_tmp","converted",
                  h3d->GetNbinsX(),h3d->GetXaxis()->GetXmin(),h3d->GetXaxis()->GetXmax(),
                  h3d->GetNbinsY(),h3d->GetYaxis()->GetXmin(),h3d->GetYaxis()->GetXmax(),
                  h3d->GetNbinsZ(),h3d->GetZaxis()->GetXmin(),h3d->GetZaxis()->GetXmax());
      for(int ix=1;ix<=h3->GetNbinsX();++ix)
        for(int iy=1;iy<=h3->GetNbinsY();++iy)
          for(int iz=1;iz<=h3->GetNbinsZ();++iz)
            h3->SetBinContent(ix,iy,iz,h3d->GetBinContent(ix,iy,iz));
    }
  }
  if(!h3){ std::cerr<<"[ERROR] hist "<<in_hist<<" not found\n"; return; }

  // --- Preparing for the slices
  TDAcc acc;
  acc.xmin=h3->GetXaxis()->GetXmin(); acc.xmax=h3->GetXaxis()->GetXmax();
  acc.ymin=h3->GetYaxis()->GetXmin(); acc.ymax=h3->GetYaxis()->GetXmax();
  acc.zmin=h3->GetZaxis()->GetXmin(); acc.zmax=h3->GetZaxis()->GetXmax();
  acc.linear_xy=linear_xy; acc.linear_z=linear_z;

  acc.zEdges.resize(nslices_z+1);
  for(int i=0;i<=nslices_z;++i) acc.zEdges[i]=acc.zmin + (acc.zmax-acc.zmin)*(double(i)/nslices_z);

  // --- Create the slices (XY = m2 vs cos)
  for(int s=0;s<nslices_z;++s){
    double z1=acc.zEdges[s], z2=acc.zEdges[s+1];
    int binZ1=h3->GetZaxis()->FindBin(z1+1e-9);
    int binZ2=h3->GetZaxis()->FindBin(z2-1e-9);
    binZ1=std::max(1,binZ1);
    binZ2=std::min(h3->GetNbinsZ(),binZ2);
    if(binZ2<binZ1) binZ2=binZ1;
    h3->GetZaxis()->SetRange(binZ1,binZ2);

    const char* proj = swap_xy_input ? "yx" : "xy"; // "yx" => X=m2, Y=cos (if in the original file X=cos,Y=m2)
    TH2* tmp=(TH2*)h3->Project3D(proj);
    TString nm=TString::Format("acc2D_tbin_%02d",s);
    tmp->SetName(nm);
    tmp->SetTitle(TString::Format("slice %d: %.5g<=t/tau<%.5g",s,z1,z2));

    TH2D* h2=dynamic_cast<TH2D*>(tmp);
    if(!h2){
      h2=new TH2D(nm,tmp->GetTitle(),
                  tmp->GetNbinsX(),tmp->GetXaxis()->GetXmin(),tmp->GetXaxis()->GetXmax(),
                  tmp->GetNbinsY(),tmp->GetYaxis()->GetXmin(),tmp->GetYaxis()->GetXmax());
      for(int ix=1;ix<=tmp->GetNbinsX();++ix)
        for(int iy=1;iy<=tmp->GetNbinsY();++iy)
          h2->SetBinContent(ix,iy,tmp->GetBinContent(ix,iy));
      delete tmp;
    }
    if(normalize_by_max){
      double m=h2->GetMaximum(); if(m>0) h2->Scale(1.0/m);
    }
    acc.slices.push_back(h2);
    h3->GetZaxis()->SetRange(0,0);
  }
  h3->GetZaxis()->SetRange(0,0);

  // --- Output ROOT
  std::unique_ptr<TFile> fout(TFile::Open(out_file,"RECREATE"));
  if(!fout||fout->IsZombie()){ std::cerr<<"[ERROR] output "<<out_file<<"\n"; return; }

  TVectorD zvec((int)acc.zEdges.size());
  for (int i=0;i<zvec.GetNrows();++i) zvec[i]=acc.zEdges[i];
  zvec.Write("zEdges");

  // save slices
  for(auto* h2: acc.slices) h2->Write();

  // --- save parameters
  {
    std::ofstream mfs("acc_parameters.txt");
    mfs << "file="<<out_file<<"\n";
    mfs << "basis="<<basis<<"\n";
    mfs << "polyNx="<<polyNx<<"\n";
    mfs << "polyNy="<<polyNy<<"\n";
    mfs << "nslices="<<nslices_z<<"\n";
    mfs << "xrange="<<acc.xmin<<","<<acc.xmax<<"\n";
    mfs << "yrange="<<acc.ymin<<","<<acc.ymax<<"\n";
    mfs << "zrange="<<acc.zmin<<","<<acc.zmax<<"\n";
  }

  // --- Fits for slice + CSVs
  std::ofstream allcsv("coefficients_all.csv");
  allcsv << "slice,i,j,coef\n";

  if (fit_slices) {
    for(size_t s=0;s<acc.slices.size();++s){
      TH2D* h2 = acc.slices[s];

      TString form = BuildPolyForm(polyNx,polyNy);   // polinomial base
      TString fname=TString::Format("acc2D_fit_tbin_%02d",(int)s);
      TF2 f2(fname, form,
             h2->GetXaxis()->GetXmin(), h2->GetXaxis()->GetXmax(),
             h2->GetYaxis()->GetXmin(), h2->GetYaxis()->GetXmax());

      int npar=(polyNx+1)*(polyNy+1);
      for(int ip=0; ip<npar; ++ip) f2.SetParameter(ip, 1e-3);

      h2->Fit(&f2, "QN0"); 
      f2.Write();

      // print formulation format
      std::cout << "\n[Slice " << s << "] " << h2->GetTitle() << "\n";
      std::cout << "f_"<<s<<"(x,y) = ";
      int k=0;
      for(int i=0;i<=polyNx;++i)
        for(int j=0;j<=polyNy;++j,++k){
          double c=f2.GetParameter(k);
          std::cout << (k==0? "" : " + ")
                    << Form("(%.6g)*x^%d*y^%d", c, i, j);
        }
      std::cout << "\n";

      // CSV for slice
      TString csvname=TString::Format("coeff_slice_%02d.csv",(int)s);
      std::ofstream ofs(csvname.Data());
      ofs << "i,j,coef\n";
      k=0;
      for(int i=0;i<=polyNx;++i)
        for(int j=0;j<=polyNy;++j,++k){
          double c=f2.GetParameter(k);
          ofs << i << "," << j << "," << c << "\n";
          allcsv << s << "," << i << "," << j << "," << c << "\n";
        }
    }
  }
  allcsv.close();

// --- 3D sample (acc3_sampled) built from the slices (useful for inspection)
  gTD=&acc;
  TF3 acc3("acc3_session", TDWrapper,
           acc.xmin,acc.xmax, acc.ymin,acc.ymax, acc.zmin,acc.zmax, 0);

  TH3F sampled("acc3_sampled","A(x,y,z) sampled",
               sample_nx,acc.xmin,acc.xmax,
               sample_ny,acc.ymin,acc.ymax,
               sample_nz,acc.zmin,acc.zmax);
  for (int ix=1; ix<=sampled.GetNbinsX(); ++ix){
    double x=sampled.GetXaxis()->GetBinCenter(ix);
    for (int iy=1; iy<=sampled.GetNbinsY(); ++iy){
      double y=sampled.GetYaxis()->GetBinCenter(iy);
      for (int iz=1; iz<=sampled.GetNbinsZ(); ++iz){
        double z=sampled.GetZaxis()->GetBinCenter(iz);
        double v=acc3.Eval(x,y,z);
        sampled.SetBinContent(ix,iy,iz,v);
      }
    }
  }
  sampled.Write();

  fout->Close();
  fin->Close();

  std::cout << "\n[OK] Gerado " << out_file
            << " com " << acc.slices.size() << " fatias 2D.\n"
            << "Interps: Z="<<(linear_z?"linear":"degrau")
            << ", XY="<<(linear_xy?"bilinear":"bin")<<"\n"
            << "Normalização por fatia: "<<(normalize_by_max?"ON":"OFF")<<"\n";
  gTD=nullptr;
}

// =====================================================================
// Reconstructs a TF3 from the fitted TF2s saved in the ROOT file
// (stepwise in Z by default; can interpolate linearly if desired).
// =====================================================================
TF3* rebuild_TF3_from_fits(const char* acc_file, bool linear_z=false)
{
  TFile* f=TFile::Open(acc_file,"READ");
  if(!f||f->IsZombie()){ std::cerr<<"[ERROR] cannot open "<<acc_file<<"\n"; return nullptr; }

  TVectorD* zA = (TVectorD*)f->Get("zEdges");
  if(!zA){ std::cerr<<"[ERROR] zEdges missing\n"; return nullptr; }

  std::vector<TF2*> fits;
  for (int s=0;;++s) {
    TString nm=TString::Format("acc2D_fit_tbin_%02d", s);
    TF2* f2=(TF2*)f->Get(nm);
    if(!f2) break;
    fits.push_back(f2);
  }
  if(fits.empty()){ std::cerr<<"[ERROR] no fitted TF2 found\n"; return nullptr; }

  double xmin=fits[0]->GetXmin(), xmax=fits[0]->GetXmax(); // X = m2
  double ymin=fits[0]->GetYmin(), ymax=fits[0]->GetYmax(); // Y = cos
  double zmin=(*zA)[0], zmax=(*zA)[zA->GetNrows()-1];

  delete gFitsCtx; gFitsCtx = new FitsCtx();
  gFitsCtx->fits     = fits;
  gFitsCtx->zEdges.resize(zA->GetNrows());
  for (int i=0;i<zA->GetNrows();++i) gFitsCtx->zEdges[i]=(*zA)[i];
  gFitsCtx->linear_z = linear_z;

  TF3* A = new TF3("acc3_fits", FitsWrapper, xmin,xmax, ymin,ymax, zmin,zmax, 0);
  A->SetTitle("Time-dependent acceptance A(x,y,z) from TF2 fits");
  return A;
}

// =====================================================================
// Samples the reconstructed TF3 from the TF2s into a TH3F and saves it to a ROOT file
// =====================================================================
TH3F* make_acc3_from_fits_sample(const char* accroot="acc_time_dep_v3.root",
                                 const char* outroot="acc3_from_fits.root",
                                 int nx=80,int ny=80,int nz=60,
                                 bool linear_z=false, bool clamp_neg=true)
{
  TF3* Af = rebuild_TF3_from_fits(accroot, linear_z);
  if(!Af){ ::Error("make_acc3_from_fits_sample","TF3 nulo"); return nullptr; }

  TH3F* H = new TH3F("acc3_from_fits","A(x,y,z) from TF2 fits",
                     nx, Af->GetXmin(), Af->GetXmax(),
                     ny, Af->GetYmin(), Af->GetYmax(),
                     nz, Af->GetZmin(), Af->GetZmax());
  for(int ix=1; ix<=nx; ++ix){
    double x=H->GetXaxis()->GetBinCenter(ix);
    for(int iy=1; iy<=ny; ++iy){
      double y=H->GetYaxis()->GetBinCenter(iy);
      for(int iz=1; iz<=nz; ++iz){
        double z=H->GetZaxis()->GetBinCenter(iz);
        double w=Af->Eval(x,y,z);
        if (clamp_neg && w<0) w=0;
        H->SetBinContent(ix,iy,iz,w);
      }
    }
  }
  TFile fout(outroot,"RECREATE"); H->Write(); fout.Close();
  return H;
}

// =====================================================================
// Optional wmax scanner (global and per-slice) for accept–reject
// =====================================================================
void scan_wmax_from_fits(const char* acc_root="acc_time_dep_v3.root",
                         int nx=120,int ny=120,int nz=80)
{
  TF3* Af = rebuild_TF3_from_fits(acc_root, /*linear_z=*/false);
  if(!Af){ ::Error("scan_wmax_from_fits","TF3 nulo"); return; }

  double xmin=Af->GetXmin(), xmax=Af->GetXmax();
  double ymin=Af->GetYmin(), ymax=Af->GetYmax();
  double zmin=Af->GetZmin(), zmax=Af->GetZmax();

  double wmax = 0.0;

  TFile f(acc_root);
  auto zv=(TVectorD*)f.Get("zEdges");
  std::vector<double> zedges(zv->GetNrows());
  for (int i=0;i<zv->GetNrows();++i) zedges[i]=(*zv)[i];
  std::vector<double> wmax_slice(zedges.size()-1, 0.0);

  for (int ix=0; ix<nx; ++ix){
    double x = xmin + (xmax-xmin)*(ix+0.5)/nx;
    for (int iy=0; iy<ny; ++iy){
      double y = ymin + (ymax-ymin)*(iy+0.5)/ny;
      for (int iz=0; iz<nz; ++iz){
        double z = zmin + (zmax-zmin)*(iz+0.5)/nz;
        double w = Af->Eval(x,y,z);
        if (w<0) w=0;
        if (w>wmax) wmax=w;
        int s = (z<=zedges.front()) ? 0 :
                (z>=zedges.back())  ? (int)wmax_slice.size()-1 :
                (int)(std::upper_bound(zedges.begin(), zedges.end(), z) - zedges.begin()) - 1;
        if (s<0) s=0;
        if (s>=(int)wmax_slice.size()) s=(int)wmax_slice.size()-1;
        if (w>wmax_slice[s]) wmax_slice[s]=w;
      }
    }
  }

  std::ofstream og("wmax_global.txt"); og<<wmax<<"\n";
  std::ofstream os("wmax_per_slice.csv"); os<<"slice,wmax\n";
  for (size_t s=0; s<wmax_slice.size(); ++s) os<<s<<","<<wmax_slice[s]<<"\n";

  std::cout<<"[scan] wmax_global="<<wmax<<"\n";
}
