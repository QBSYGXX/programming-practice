#include<algorithm>
#include<string>
#include<vector>
#include<sstream>
using namespace std;
//一个数组的 异或总和 定义为数组中所有元素按位 XOR 的结果；如果数组为 空 ，则异或总和为 0 。
//例如，数组[2, 5, 6] 的 异或总和 为 2 XOR 5 XOR 6 = 1 。
//给你一个数组 nums ，请你求出 nums 中每个 子集 的 异或总和 ，计算并返回这些值相加之 和 。
//注意：在本题中，元素 相同 的不同子集应 多次 计数。
//数组 a 是数组 b 的一个 子集 的前提条件是：从 b 删除几个（也可能不删除）元素能够得到 a 。
//示例 1：
//输入：nums = [1, 3]
//输出：6
//解释：[1, 3] 共有 4 个子集：
//- 空子集的异或总和是 0 。
//- [1] 的异或总和为 1 。
//- [3] 的异或总和为 3 。
//- [1, 3] 的异或总和为 1 XOR 3 = 2 。
//0 + 1 + 3 + 2 = 6
class Solution {
public:
	int subsetXORSum(vector<int>& nums)
	{
		int total = 0;
		backtrack(nums, 0, 0, total);
		return total;
	}
	void backtrack(vector<int>nums, int index, int currentXOR, int& total)
	{
		if (index == nums.size())//递归边界
		{
			total += currentXOR;
			return;
		}
		backtrack(nums, index + 1, currentXOR ^ nums[index], total);//选择
		backtrack(nums, index + 1, currentXOR, total);//不选
	}
};
