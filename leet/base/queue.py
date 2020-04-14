# -*- coding: utf-8 -*-
"""
create on 2020-04-14 16:37
author @66492
"""
from leet.base.linked_list import LinkedList


class Queue(object):
    MAX_LENGTH = 0

    _queue = LinkedList()

    def __init__(self, max_length):
        self.MAX_LENGTH = max_length
        self._queue = LinkedList(0) if self._queue.length == 0 else self._queue

    @classmethod
    def from_list(cls, ls: list, max_length):
        for i in ls:
            cls._queue = LinkedList.from_list(ls)
        return cls(max_length)

    def pop(self):
        if self.length == 0:
            return None
        return self._queue.pop(0)

    def push(self, val):
        if self._queue.length == self.MAX_LENGTH:
            self._queue.pop(0)
            self._queue.insert(val, position=-1)
            return
        self._queue.insert(val, position=-1)

    def clear(self):
        self._queue.clear()

    @property
    def length(self):
        return self._queue.length

    @length.setter
    def length(self, val):
        raise RuntimeError("can't set length")

    def to_list(self):
        return self._queue.to_list()

    def __repr__(self):
        return r"<Queue: %s>" % self.length


if __name__ == '__main__':
    queue = Queue.from_list(list(range(9)), max_length=10)
    print("length of queue:", queue.length)
    queue.push(10)
    print(queue.to_list())
    queue.push(11)
    print(queue.to_list())
    queue.push(12)
    print(queue.to_list())
    queue.clear()
    print(queue)
    print(queue.to_list())
    print(queue.pop())
