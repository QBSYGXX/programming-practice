//KMP算法
class Solution {
public:
    void getNext(int* next, string s)//获取next数组
    {
        int i = 0;//前缀末端指针
        next[0] = 0;
        for (int j = 1; j < s.size(); j++)//后缀末端指针
        {
            while (i > 0 && s[i] != s[j])
            {
                i = next[i - 1];
            }
            if (s[i] == s[j])//前后缀匹配
            {
                i++;
                next[j] = i;
            }
            else
            {
                next[j] = 0;
            }
        }
    }
    int strStr(string haystack, string needle) 
    {
        if (needle.size() == 0)
        {
            return 0;//空字符串返回0
        }
        int i = 0, j = 0;//i为主串指针，j为子串指针
        vector<int>next(needle.size());
        getNext(&next[0], needle);//构造next数组
        for (i = 0; i < haystack.size();) {
            if (haystack[i] == needle[j]) 
            {
                i++;
                j++;
            }
            else 
            {
                if (j > 0)
                    j = next[j - 1];//查前一个元素next值得到子串指针回溯位置（不是查自己的next值因为自己已经不匹配了，要查前面匹配的前缀）
                else//回溯到子串第一个位置也匹配失败，因此移动主串指针
                    i++;
            }
            if (j == needle.size()) {//全部匹配成功
                return i - j;
            }
        }
        return -1;//匹配失败
    }
};
