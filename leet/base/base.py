# -*- coding: utf-8 -*-
"""
create on 2020-04-14 15:13
author @66492
"""


class BaseList(object):
    def __init__(self, length=0):
        """

        :param length: int. don't set this to none-zero when you create `LinkedList`
        """
        self._length = length

    def insert(self, val, position=0):
        raise NotImplementedError()

    def pop(self, idx):
        raise NotImplementedError()

    def remove(self, val):
        raise NotImplementedError()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, val):
        raise RuntimeError("can't set property length")

    def count(self, val):
        raise NotImplementedError()

    def sort(self, ascending=True):
        raise NotImplementedError()

    def index(self, val, start_at=0):
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    def to_list(self):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()
