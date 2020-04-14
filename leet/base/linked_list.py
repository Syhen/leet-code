# -*- coding: utf-8 -*-
"""
create on 2020-04-14 15:05
author @66492
"""
from leet.base.base import BaseList


class Node(object):
    def __init__(self, val, next_node):
        self.val = val
        self.next_node = next_node

    def __repr__(self):
        return "<Node: %s>" % self.val


class LinkedList(BaseList):
    head_node = Node(None, None)

    def __init__(self, length=0):
        super(LinkedList, self).__init__(length)
        if length == 0:
            self.head_node.next_node = None

    @classmethod
    def from_list(cls, ls: list):
        if not ls:
            return cls(0)
        counter = 0
        node = cls.head_node
        for i in ls:
            node.next_node = Node(i, None)
            node = node.next_node
            counter += 1
        return cls(counter)

    def pop(self, idx):
        if idx > self.length - 1:
            raise ValueError("max index is %s" % (self.length - 1))
        node = self.head_node.next_node
        pre_node = self.head_node
        counter = 0
        idx = idx if idx != -1 else self.length - 1
        while 1:
            if idx != counter:
                counter += 1
                pre_node = node
                node = node.next_node
                continue
            pre_node.next_node = node.next_node
            self._length -= 1
            return node.val

    def _to_list(self):
        node = self.head_node.next_node
        while 1:
            if node is None:
                break
            yield node.val
            node = node.next_node

    def to_list(self):
        return list(self._to_list())

    def to_dict(self):
        return {i: v for i, v in enumerate(self._to_list())}

    def _skip(self, skip_num):
        node = self.head_node
        for _ in range(skip_num):
            node = node.next_node
        return node

    def index(self, val, start_at=0):
        node = self._skip(start_at)

        def check_and_return(node, val, return_val=0):
            if node is None:
                raise IndexError("%s is not in %s" % (val, str(self)))
            if node.val == val:
                return return_val
            return -1

        _tmp = check_and_return(node, val)
        if _tmp != -1:
            return _tmp
        idx = 1
        while 1:
            node = node.next_node
            _tmp = check_and_return(node, val, return_val=idx)
            if _tmp != -1:
                return _tmp
            idx += 1

    def insert(self, val, position=-1):
        if position == -1:
            position = self._length
        head = self._skip(position)
        head.next_node = Node(val, head.next_node)
        self._length += 1

    def remove(self, val):
        pre_node = self.head_node
        node = pre_node.next_node
        while 1:
            if node is None:
                raise ValueError("%s not in %s" % (val, str(self)))
            if node.val == val:
                pre_node.next_node = node.next_node
                self._length -= 1
                break
            pre_node = node
            node = pre_node.next_node

    def reverse(self):
        node = self.head_node.next_node
        pre_node = None
        while 1:
            if node is None:
                break
            next_node = node.next_node
            node.next_node = pre_node
            pre_node, node = node, next_node
        self.head_node.next_node = pre_node

    def count(self, val):
        counter = 0
        node = self.head_node.next_node
        while 1:
            if node is None:
                break
            if node.val == val:
                counter += 1
            node = node.next_node
        return counter

    def sort(self, ascending=True):
        raise NotImplementedError("implement later")

    def clear(self):
        self.head_node.next_node = None
        self._length = 0

    def __repr__(self):
        return r"<LinkedList: %s>" % self.length


if __name__ == '__main__':
    linked_list = LinkedList.from_list([1, 2, 3, 4, 5, 6, 7])
    print(linked_list.index(3, start_at=0))
    linked_list.insert(10)
    linked_list.insert(20, position=2)
    print(linked_list.length)
    print(linked_list)
    linked_list.remove(4)
    print(linked_list.to_list())
    print(linked_list.to_dict())
    linked_list.reverse()
    print(linked_list.count(3))
    print(linked_list.to_list())
    print(linked_list.pop(3))
    print(linked_list.to_list())

    linked_list2 = LinkedList()
    print("insert length:", linked_list2.length)
