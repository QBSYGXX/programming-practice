//二进制手表顶部有 4 个 LED 代表 小时（0 - 11），底部的 6 个 LED 代表 分钟（0 - 59）。
//每个 LED 代表一个 0 或 1，最低位在右侧。
//给你一个整数 turnedOn ，表示当前亮着的 LED 的数量，返回二进制手表可以表示的所有可能时间。你可以 按任意顺序 返回答案。
//小时不会以零开头：
//例如，"01:00" 是无效的时间，正确的写法应该是 "1:00" 。
//分钟必须由两位数组成，可能会以零开头：
//例如，"10:2" 是无效的时间，正确的写法应该是 "10:02" 。
//示例 1：
//输入：turnedOn = 1
//输出：["0:01", "0:02", "0:04", "0:08", "0:16", "0:32", "1:00", "2:00", "4:00", "8:00"]

class Solution {
public:
    vector<string> readBinaryWatch(int turnedOn) {
        vector<string> result;
        backtrack(turnedOn, 0, 0, 0, result);
        return result;
    }

private:
    void backtrack(int turnedOn, int hour, int minute, int ledIndex, vector<string>& result) {
        // 终止条件：当LED点亮数量达到要求时
        if (turnedOn == 0) {
            if (hour < 12 && minute < 60) {
                string time = to_string(hour) + ":" +
                    (minute < 10 ? "0" : "") + to_string(minute);
                result.push_back(time);
            }
            return;
        }

        // 所有LED都已经考虑过
        if (ledIndex >= 10) return;

        // 选择点亮当前LED
        if (ledIndex < 4) { // 小时LED
            backtrack(turnedOn - 1, hour + (1 << ledIndex), minute, ledIndex + 1, result);
        }
        else { // 分钟LED
            int minVal = 1 << (ledIndex - 4);
            backtrack(turnedOn - 1, hour, minute + minVal, ledIndex + 1, result);
        }

        // 不选择当前LED
        backtrack(turnedOn, hour, minute, ledIndex + 1, result);
    }
};
