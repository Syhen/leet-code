# -*- coding: utf-8 -*-
"""
create on 2020-04-14 16:37
author @66492
"""
from leet.base.linked_list import LinkedList


class Stack(object):
    _stack = LinkedList()

    def __init__(self):
        self._stack = LinkedList(0) if self._stack.length == 0 else self._stack

    @classmethod
    def from_list(cls, ls: list):
        for i in ls:
            cls._stack = LinkedList.from_list(ls)
        return cls()

    def pop(self):
        if self.length == 0:
            return None
        return self._stack.pop(-1)

    def push(self, val):
        self._stack.insert(val, position=-1)

    def clear(self):
        self._stack.clear()

    @property
    def length(self):
        return self._stack.length

    @length.setter
    def length(self, val):
        raise RuntimeError("can't set length")

    def to_list(self):
        return self._stack.to_list()

    def __repr__(self):
        return r"<Stack: %s>" % self.length


if __name__ == '__main__':
    stack = Stack.from_list(list(range(9)))
    print("length of stack:", stack.length)
    stack.push(10)
    print(stack.to_list())
    stack.push(11)
    print(stack.to_list())
    stack.push(12)
    print(stack.pop())
    print(stack.to_list())
    stack.clear()
    print(stack)
    print(stack.to_list())
    print(stack.pop())
