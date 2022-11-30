// Written Assignment KNN
// leave 1 out (1.a 1.b)

#include <iostream>
#include <vector>
using namespace std;

int main() {
    // input
    int n, k, fold;
    cin>>n>>k>>fold;
    int num_leave = n/fold;
    vector<pair<float, int>> table(n);
    // input data into vector
    for (int i = 0; i<n; i++){
        pair<float, int> temp;
        float temp1; char temp2;
        cin>>temp1>>temp2;
        temp.first = temp1;
        if (temp2 == '+'){
            temp.second = 1;
        }
        else{
            temp.second = 0;
        }
        table[i] = temp;
    }

    int total_misclassification = 0;
    for (int iter_fold = 1; iter_fold <= fold; iter_fold++){
        cout<<"Iteration "<<iter_fold<<"\n";
        cout<<"#\tx\ty\tchoose0\tchoose1\tpredict\tmisclassified\n";

        vector<pair<float, int>> temp_table;
        for (int i = 0; i<n; i++){
            if (i>=num_leave*(iter_fold-1) && i<num_leave*iter_fold) {
                continue;
            }
            temp_table.push_back(table[i]);
        }
        for (int i=0; i<n-num_leave; i++){
            int choose_0 = 0, choose_1 = 0;
            int left = i-1, right = i+1;
            // count the value of the nearest neighbors
            while (choose_0 + choose_1 < k){
                if (left <0 && right>=n-num_leave){
                    cout<<"Not enough data";
                    return 1;
                }
                else if (left<0){
                    if (temp_table[right].second == 0){
                        choose_0++;
                    }
                    else {
                        choose_1++;
                    }
                    right++;
                }
                else if (right>=n-num_leave){
                    if (temp_table[left].second == 0){
                        choose_0++;
                    }
                    else {
                        choose_1++;
                    }
                    left--;
                }
                else {
                    if (abs(temp_table[left].first - temp_table[i].first)<= abs(temp_table[right].first - temp_table[i].first)){
                        if (temp_table[left].second == 0){
                            choose_0++;
                        }
                        else {
                            choose_1++;
                        }
                        left--;
                    }
                    else {
                        if (temp_table[right].second == 0){
                            choose_0++;
                        }
                        else {
                            choose_1++;
                        }
                        right++;
                    }
                }
            }
            int classified = (choose_1>choose_0)?1:0;

            if (classified != temp_table[i].second){
                total_misclassification++;
            }
            cout<<i<<"\t"<<temp_table[i].first<<"\t"<<temp_table[i].second<<"\t"<<choose_0<<"\t"<<choose_1<<"\t"<<classified<<"\t"<<total_misclassification<<"\n";
        }
    }
    cout<<total_misclassification;
}
