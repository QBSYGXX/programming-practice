#include<iostream>
#include<cmath>
#include<algorithm>
#include<string>
#include<vector>
#include<sstream>
#include<map>
#include<unordered_set>
#include<unordered_map>
#include<utility>
using namespace std;
class MyLinkedList {//编写一个单链表类，并实现成员函数功能
public:
    typedef struct node
    {
        int val;
        node* next;
        node() :val(0), next(nullptr) {}
        node(int val):val(val),next(nullptr){}//构造函数
    }node;//定义节点结构体
    MyLinkedList() //初始化一个虚拟头节点
    {//insert快捷键切换改写/插入模式
        _size = 0;
        _dummpyhead = new node();
    }

    int get(int index)//根据下标索引返回元素值（如果存在的话），否则返回-1 
    {
        if (index<0 || index>_size - 1)//合法性检查
        {
            return -1;
        }
        node* p = _dummpyhead->next;//工作指针
        int num = 0;
        while (num < index)
        {
            num++;
            p = p->next;
        }
        return p->val;
    }

    void addAtHead(int val) 
    {
        node* p = new node(val);
        p->next = _dummpyhead->next;
        _dummpyhead->next = p;
        _size++;
    }

    void addAtTail(int val) 
    {
        if (_dummpyhead->next == nullptr)//链表为空
        {
            _dummpyhead->next = new node(val);
            _size++;
            return;
        }
        node* tail = _dummpyhead->next;//尾指针
        while (tail->next != nullptr)//遍历到尾部
        {
            tail = tail->next;
        }
        tail->next = new node(val);//尾插
        _size++;
    }

    void addAtIndex(int index, int val) //将一个值为 val 的节点插入到链表中下标为 index 的节点之前
    {//如果 index 等于链表的长度，那么该节点会被追加到链表的末尾。如果 index 比长度更大，该节点将 不会插入 到链表中。
        //索引合法性检查
        if (index<0 || index>_size)//合法性检查
        {
            return ;
        }
        if (index == _size)//尾插
        {
            this->addAtTail(val);
            return;
        }
        else
        {
            node* p = _dummpyhead;//工作指针
            int num = 0;
            while (num < index)
            {
                num++;
                p = p->next;
            }
            node* temp = new node(val);
            temp->next = p->next;
            p->next = temp;
            _size++;
        }
    }

    void deleteAtIndex(int index) //如果下标有效，则删除链表中下标为 index 的节点。
    {
        //索引合法性检查
        if (index<0 || index>_size - 1)//合法性检查
        {
            return;
        }
        node* p = _dummpyhead;//工作指针
        int num = 0;
        while (num < index)
        {
            num++;
            p = p->next;
        }
        node* temp = p->next;
        p->next = temp->next;
        delete temp;
        _size--;
    }
    // 调试用：打印链表所有元素
    void printList() {
        node* p = _dummpyhead->next;
        cout << "链表元素: ";
        while (p != nullptr) {
            cout << p->val << " -> ";
            p = p->next;
        }
        cout << "nullptr (长度: " << _size << ")" << endl;
    }

    ~MyLinkedList() {
        node* current = _dummpyhead;
        while (current != nullptr) {
            node* temp = current;  // 保存当前节点
            current = current->next;  // 移动到下一个节点
            delete temp;  // 释放当前节点内存
        }
    }

private:
    int _size ;//内置链表大小
    node* _dummpyhead ;//内置头指针
};

int main() {
    // 测试用例执行步骤
    cout << "=== 初始化链表 ===" << endl;
    MyLinkedList list;
    list.printList();

    cout << "\n=== 执行 addAtTail(1) ===" << endl;
    list.addAtTail(1);
    list.printList();

    cout << "\n=== 执行 addAtTail(3) ===" << endl;
    list.addAtTail(3);
    list.printList();

    cout << "\n=== 执行 get(1) ===" << endl;
    int res = list.get(1);
    cout << "get(1) 返回值: " << res << endl;

    return 0;
}
