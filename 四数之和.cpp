class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) 
    {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end()); // 双指针法排序必不可少
        
        // 难点在剪枝和去重
        for (int i = 0; i < nums.size(); i++)
        {
            // 剪枝优化：当 target 为负数且 nums[i] 为正数时，不能直接跳过
            if (nums[i] > target && (nums[i] >= 0 || target >= 0))
            {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) // 去重 a
            {
                continue;
            }
            
            for (int j = i + 1; j < nums.size(); j++)
            {
                // 剪枝优化
                if (nums[i] + nums[j] > target && (nums[i] + nums[j] >= 0 || target >= 0))
                {
                    break;
                }
                if (j > i + 1 && nums[j] == nums[j - 1]) // 去重 b
                {
                    continue;
                }
                
                int left = j + 1, right = nums.size() - 1; // 双指针
                while (left < right)
                {
                    // 关键修复：提前转换为 long 避免溢出
                    long sum = (long)nums[i] + nums[j] + nums[left] + nums[right];
                    
                    if (sum > target) right--;
                    else if (sum < target) left++;
                    else
                    {
                        result.push_back({nums[i], nums[j], nums[left], nums[right]});
                        while (left < right && nums[left] == nums[left + 1]) left++; // c 去重
                        while (left < right && nums[right] == nums[right - 1]) right--; // d 去重
                        left++;
                        right--; // 找到后双边缩进
                    }
                }
            }
        }
        return result;
    }
};
