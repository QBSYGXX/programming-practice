class Solution {
public:
    int minDeletion(string s, int k)
    {
        int answer = 0;
        unordered_map<char, int>haxi;
        for (auto ss : s)
        {
            haxi[ss]++;
        }
        vector<pair<char, int>>vec(haxi.begin(),haxi.end());
        sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {return a.second < b.second; });//sort自定义排序lamada隐函数形式
        if (vec.size() > k)
        {
            for (int i = 0; i < vec.size() - k; i++)//要高度重视数组越界访问造成的缓冲区溢出等问题
            {
                answer += vec[i].second;
            }
        }
        return answer;
    }
};//pair头文件include<utility>
