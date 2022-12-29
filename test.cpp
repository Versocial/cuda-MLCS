#include <ostream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include <time.h>
// #include <thrust/random.h>
// #include <cuda_runtime.h>
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <cooperative_groups.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>
using namespace std;

#define SETATTRINT(point, val) (point->attr = (void*)val)
#define ATTRINT(point) ((unsigned long long)point->attr)

struct Point{
    int *cord{NULL};
    void *attr{NULL};
    Point(int num){
        cord = new int[num];
        for(int i=0;i<num;i++)
            cord[i]=0;
        attr=NULL;
    }
};

class QUICKDP{

public:
    QUICKDP(){}
    QUICKDP(vector<string> seqs, string alphabets);
    ~QUICKDP(){}

    vector< Point* > Minima(vector< Point* >& points, int dim);
    void run();
    string get_lcs() const {return lcs;}

private:
    inline bool analyze_points(vector< Point* >& points, int dim);
    vector<Point*> Divide(vector< Point* >& points, int d);
    vector<Point*> Union(vector< Point* >& points, int d);
    inline int CharIndex(const Point *p){return cmap[seqs[0][p->cord[0] - 1]];}

    vector<string> seqs;
    map<char, int> cmap;
    int alphaSize;
    vector< vector< vector<int> > > SucTabs;
    string lcs;

};

map<char, int> build_alphabet_map(string& alphabets){
    map<char, int> cmap;
    for(int i = 0; i < alphabets.length(); i++){
        cmap.insert(make_pair(alphabets[i], i));
    }
    return cmap;
}

vector< vector<int> > cal_suc_tab(string& seq, map<char, int>& cmap, bool disp){
    int len = seq.length();
    vector< vector<int> > SucTab(cmap.size(), vector<int>(len + 1));

    // calculate successor table
    for (int i = 0; i < cmap.size(); i++) {
        SucTab[i][len] = -1;
    }

    for (int j = len - 1; j >= 0; j--) {
        for (int i = 0; i < cmap.size(); i++) {
            SucTab[i][j] = SucTab[i][j + 1];
        }
        SucTab[cmap[seq[j]]][j] = j + 1;
    }

    if(disp){
        cout << "\n  | \t\t";
        for(int i = 0; i < len; i++) cout << seq[i] << "\t";
        cout << endl;
        for(int i = 0; i < len + 2; i++) cout << "--" << "\t";
        cout << endl;
        for (auto m : cmap) {
            cout << m.first << " | \t";
            for (int j = 0; j <= len; j++) {
                cout << SucTab[m.second][j] << "\t";
            }
            cout << endl;
        }
        cout << endl;
    }
    return SucTab;
}

vector< vector< vector<int> > > cal_suc_tabs(vector<string>& seqs, map<char, int>& cmap, bool disp){
    
    vector< vector< vector<int> > > SucTabs;
    for(auto& seq : seqs){
        SucTabs.push_back(cal_suc_tab(seq, cmap, disp));
    }
    return SucTabs;
    
}

Point* successor(Point* p, vector< vector< vector<int> > >& SucTabs, int i){
    Point* q = new Point(SucTabs.size());
    for(unsigned int j = 0; j < SucTabs.size(); ++j){
        q->cord[j] = SucTabs[j][i][p->cord[j]];
        if(q->cord[j] < 0){ // dominants do not exists.
            delete q;
            return NULL;
        }
    }
    return q;
}

vector<Point*> set2vec(set<Point*>& pset){
    vector<Point*> pvec;
    for(auto p : pset) pvec.push_back(p);
    return pvec;
}


struct Key_equal{
    bool operator() (const Point *p1, const Point *p2, int pointsize) const;
};
bool Key_equal::operator() (const Point *p1, const Point *p2, int pointsize) const {
    for(int i = 0; i < pointsize; i++){
        if(p1->cord[i] != p2->cord[i]) return false;
    }
    return true;
}

void Qsort(vector< Point* >& arr, int low, int high, int dim){
    if (high <= low) return;
    int i = low;
    int j = high + 1;
    int key = arr[low]->cord[dim];
    while (true)
    {
        while (arr[++i]->cord[dim] < key)
        {
            if (i == high){
                break;
            }
        }
        while (arr[--j]->cord[dim] > key)
        {
            if (j == low){
                break;
            }
        }
        if (i >= j) break;
        swap(arr[i], arr[j]);
    }
    swap(arr[j], arr[low]);
    Qsort(arr, low, j - 1, dim);
    Qsort(arr, j + 1, high, dim);
}

int Qselect(vector< Point* >& arr, int k, int dim){

    if (arr.size() <= 1){
        return arr[0]->cord[dim];
    }

    int pivot = arr[arr.size()/2]->cord[dim];
    vector<Point*> first;
    vector<Point*> last;
    vector<Point*> pivots;

    for(auto p : arr){
        if(p->cord[dim] < pivot){
            first.push_back(p);
        }
        else if(p->cord[dim] > pivot){
            last.push_back(p);
        }
        else{
            pivots.push_back(p);
        }
    }

    if(first.size() > k){
        return Qselect(first, k, dim);
    }
    else if(first.size() + pivots.size() > k){
        return pivot;
    }
    else{
        return Qselect(last, k - first.size() - pivots.size(), dim);
    }

}

int Qmedian(vector< Point* >& arr, int dim){

    if(arr.size() % 2 == 0){
        return (Qselect(arr, arr.size() / 2 - 1, dim) + Qselect(arr, arr.size() / 2, dim)) / 2;
    }
    else{
        return Qselect(arr, arr.size() / 2, dim);
    }

}

int vmax(vector< Point* >& arr, int dim){
    int res = arr[0]->cord[dim];
    for(auto p : arr){
        if(p->cord[dim] > res) res = p->cord[dim];
    }
    return res;
}

vector<Point*> mergeSortedVecter(vector<Point*>& A, vector<Point*>& B, int dim){

    int i = 0, j = 0, k = 0;
    int M = A.size();
    int N = B.size();
    vector<Point*> res(M + N);

    while(i < M && j < N){
        if(A[i]->cord[dim] < B[j]->cord[dim]){
            res[k++] = A[i++];
        }
        else if(A[i]->cord[dim] > B[j]->cord[dim]){
            res[k++] = B[j++];
        }
        else{
            res[k++] = A[i++];
            res[k++] = B[j++];
        }
    }

    while(j < N) res[k++] = B[j++];
    while(i < M) res[k++] = A[i++];

    return res;

}

vector<Point*> mergeSortedSet(vector<Point*>& A, vector<Point*>& B, int dim, int pointsize){

    int i = 0, j = 0;
    int M = A.size();
    int N = B.size();
    vector<Point*> res;

    while(i < M && j < N){
        if(A[i]->cord[dim] < B[j]->cord[dim]){
            res.push_back(A[i++]);
        }
        else if(A[i]->cord[dim] > B[j]->cord[dim]){
            res.push_back(B[j++]);
        }
        else{
            char flag = 0; // 0: A = B 1: A < B 2: A > B
            for(int k = dim + 1; k < pointsize; k++){
                if(A[i]->cord[k] < B[j]->cord[k]){flag = 1; break;}
                else if(A[i]->cord[k] > B[j]->cord[k]){flag = 2; break;}
            }
            if(flag == 1){
                res.push_back(A[i++]);
            }
            else if(flag == 2){
                res.push_back(B[j++]);
            }
            else{
                res.push_back(A[i++]);j++;
            }
        }
    }

    while(j < N) res.push_back(B[j++]);
    while(i < M) res.push_back(A[i++]);

    return res;

}

QUICKDP::QUICKDP(vector<string> seqs, string alphabets)
 : seqs(seqs),alphaSize(alphabets.length())
{
    cmap = build_alphabet_map(alphabets);
    SucTabs = cal_suc_tabs(seqs, cmap, true);
}

vector< Point* > QUICKDP::Minima(vector< Point* >& points, int dim){

    Qsort(points, 0, points.size() - 1, 0); // sorted by x-axis
    return Divide(points, dim);

}

void QUICKDP::run(){
    //初始点坐标（0，0，……，0）
    Point* p0 = new Point(seqs.size());
    //0级主导点
    int k = 0;
    vector<set<Point*>> D;
    
    D.push_back(set<Point*>{p0});

    while(D[k].size() != 0){
        if(k==3)
            ;
        vector< set<Point*> > Pars = vector< set<Point*> >(alphaSize);
        set<Point*> trash;
        // 对k级主导点中的所有点
        for(Point *point : D[k]){
            vector<Point*> Par;
            // generate successors
            for(int i = 0; i < alphaSize; i++){
                Point *suc = successor(point, SucTabs, i);
                if(suc){
                    if(trash.find(suc) == trash.end())
                        Par.push_back(suc);
                    else delete suc;
                }
            }
            // minimal Par
            vector<Point*> minPar = Minima(Par, seqs.size());
            // put all the points belong to minPar in Pars correspondingly, meanwhile, put all the points
            // belong to (Par - minPar) in trash.
            sort(minPar.begin(), minPar.end());
            sort(Par.begin(), Par.end());
            for(int i = 0, j = 0; i < Par.size(); i++){
                if(j < minPar.size() && Par[i] == minPar[j]){
                    auto res = Pars[CharIndex(minPar[j])].insert(minPar[j]);
                    if(!res.second) delete minPar[j];
                    j++;
                }
                else{
                    int idx = CharIndex(Par[i]);
                    auto iter = Pars[idx].find(Par[i]);
                    if(iter != Pars[idx].end()) {
                        delete *iter;
                        Pars[idx].erase(iter);
                    }
                    auto res = trash.insert(Par[i]);
                    if(!res.second) delete Par[i];
                }
            }
        }
        // calculate Minimal Pars
        for(int i = 0; i < alphaSize; i++){
            vector<Point*> Pars_vec = set2vec(Pars[i]);
            Pars_vec = Minima(Pars_vec, seqs.size());
            for(auto p : Pars_vec) {
                D.push_back(set<Point*>{});
                D[k + 1].insert(p);
            }
                
            for(auto p : Pars[i]){ // delete dominated point
                if(D[k + 1].find(p) == D[k + 1].end()) delete p;
            }
        }
        // cout << k <<endl;
        // for(auto p : D[k]){
        //     for(int i=0;i<seqs.size();i++){
        //         cout<<p->cord[i]<<" ";
        //     }
        //     cout <<endl;
        // }
        // cout<<endl;
        k++;
    }

    //get a lcs
    k--;
    Point *curPoint = *D[k].begin();
    char c = seqs[0][curPoint->cord[0] - 1];
    lcs = "";
    lcs += c;
    set<int> idx;
    while(k > 1){
        for(auto point : D[k - 1]){
            Point *suc = successor(point, SucTabs, cmap[c]);
            if(suc == nullptr) continue;
            if(Key_equal()(suc, curPoint, seqs.size())){
                curPoint = point;
                c = seqs[0][curPoint->cord[0] - 1];
                // cout<<curPoint->cord[0] - 1<<":"<<c<<" ";
                // idx.insert(curPoint->cord[0] - 1);
                delete suc; break;
            }
            else delete suc;
        }
        lcs = c + lcs;
        k--;
    }
    // for(int i=0;i<seqs[0].size();i++){
    //     if(idx.find(i)!=idx.end()){
    //         cout<<seqs[0][i];
    //     }
    //     else
    //     cout<<(char)(seqs[0][i]+32);
    // }
    // free memory
    for(auto s : D){
        for(auto p : s) delete p;
    }
}

inline bool QUICKDP::analyze_points(vector< Point* >& points, int dim){
    int refer = points[0]->cord[dim];
    for(auto point : points){
        if(point->cord[dim] != refer) return true;
    }
    return false; // has no relation of domination.
}

vector<Point*> QUICKDP::Divide(vector< Point* >& points, int d){

    vector<Point*> res;

    // only a or zero point
    if ( points.size() <= 1 ){
        return points;
    }

    if (d == 2){ // conquer
        int ymin = INT32_MAX;
        for(auto point : points){
            if(point->cord[1] < ymin){
                ymin = point->cord[1];
                if(res.size() > 0 && point->cord[0] == res.back()->cord[0])
                    res.pop_back();
                res.push_back(point);
            }
        }

        return res;
    }

    if(!analyze_points(points, d - 1)){
        return Divide(points, d - 1);
    }

    // divide
    vector<Point*> A;
    vector<Point*> B;
    // find median number   每组点d-1维度数值的中间和最大
    int median = Qmedian(points, d - 1);
    int maxval = vmax(points, d - 1);
    // record the index of every point and place it in A or B
    bool flag = (median == maxval) ? true : false;
    for(auto point : points){
        if(point->cord[d - 1] < median){
            A.push_back(point);
        }
        else if(point->cord[d - 1] > median){
            B.push_back(point);
        }
        else{
            if(flag) B.push_back(point);
            else A.push_back(point);
        }
    }

    vector<Point*> Ares = Divide(A, d);
    vector<Point*> Bres = Divide(B, d);

    // merge
    // set label
    for(auto p : Ares){
        SETATTRINT(p, 1);
    }
    for(auto p : Bres){
        SETATTRINT(p, 0);
    }
    Ares = mergeSortedVecter(Ares, Bres, 0);
	
    // delete the point in Ares dominated by any point originally in Bres
    res = Union(Ares, d - 1);

    return res;

}

vector<Point*> QUICKDP::Union(vector< Point* >& points, int d){

    vector<Point*> res;

    // only a or zero point
    if ( points.size() <= 1){
        return points;
    }

    if (d == 2){ // conquer
        int ymin = INT32_MAX;
        for(auto point : points){
            if(ATTRINT(point) != 0){ // label is A
                if(point->cord[1] < ymin){
                    ymin = point->cord[1];
                    if(res.size() > 0 &&
                       point->cord[0] == res.back()->cord[0] &&
                       point->cord[1] <= res.back()->cord[1] &&
                       ATTRINT(res.back()) == 0){
                        res.pop_back();
                    }
                }
                res.push_back(point);
            }
            else{
                if(point->cord[1] < ymin){
                    res.push_back(point);
                }
            }
        }
        return res;
    }

    if(!analyze_points(points, d - 1)){
        return Union(points, d - 1);
    }

    // divide
    vector<Point*> A;
    vector<Point*> B;
    // find median number
    int median = Qmedian(points, d - 1);
    int maxval = vmax(points, d - 1);
    // record the index of every point and place it in A or B
    bool flag = (median == maxval) ? true : false;
    for(auto point : points){
        if(point->cord[d - 1] < median){
            A.push_back(point);
        }
        else if(point->cord[d - 1] > median){
            B.push_back(point);
        }
        else{
            if(flag) B.push_back(point);
            else A.push_back(point);
        }
    }

    vector<Point*> Ares = Union(A, d);
    vector<Point*> Bres = Union(B, d);

    // delete the point labeled 0 in Bres dominated by any point labeled 1 in Ares
    vector<Point*> UA;
    vector<Point*> UB;
    vector<Point*> Blabeled1; // the point labeled 1 in Bres
    for(auto p : Bres){
        if(ATTRINT(p) == 0) UB.push_back(p);
        else Blabeled1.push_back(p);
    }
    for(auto p : Ares){
        if(ATTRINT(p) == 1) UA.push_back(p);
    }
    UA = mergeSortedVecter(UA, UB, 0);
    vector<Point*> Ures = Union(UA, d - 1);

    // merge consequence
    Ares = mergeSortedSet(Ares, Blabeled1, 0, seqs.size());
    Ares = mergeSortedSet(Ares, Ures, 0, seqs.size());

    return Ares;

}



int main(){
    vector<string> seqs;
    ifstream file;
    file.open("3.txt",ios_base::in);
    if (!file.is_open())
    {
        cout << "open file error\n";
    }
    string s;
    while (getline(file, s))
    {
        seqs.push_back(s);
    }
    
    
    string alphabet_set = "ACTG";
    QUICKDP quickdp(seqs, alphabet_set);
    string lcs;
    // cudaEvent_t start_event, stop_event;
    // float run_time;
    // int eventflags =
    //   (cudaEventBlockingSync);

    // checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
    // checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));
    // checkCudaErrors(cudaEventRecord(start_event, 0));
    clock_t start,end;
    start = clock();
    quickdp.run();
    end = clock();
    cout<<"rum_time = "<<double(end-start)*1000/CLOCKS_PER_SEC<<"ms"<<endl;
    // checkCudaErrors(cudaEventRecord(stop_event, 0));
    // checkCudaErrors(cudaEventSynchronize(stop_event));
    // checkCudaErrors(cudaEventElapsedTime(&run_time, start_event, stop_event));
    // printf("run_time:\t%f\n", run_time);
    lcs = quickdp.get_lcs();
    cout << "Result(by " << "quickdp" << "):\n";
	// os << "time(us) : " << end_t - start_t << "\n";
	cout << "the length of lcs : " << lcs.length() << "\n";
	cout << "a lcs : " << lcs << "\n";
    return 0;
}
