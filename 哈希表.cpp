#include <iostream>
#include <unordered_set>
#include <string>

class Solution {
public:
    int numJewelsInStones(std::string jewels, std::string stones) {
        // 使用unordered_set创建哈希表，存储jewels中的字符
        std::unordered_set<char> jewelSet(jewels.begin(), jewels.end());
        int count = 0;
        // 遍历stones中的每个字符
        for (char stone : stones) {
            // 如果该字符在哈希表中，说明是宝石，计数器加1
            if (jewelSet.find(stone) != jewelSet.end()) {
                count++;
            }
        }
        return count;
    }
};//力扣771题
