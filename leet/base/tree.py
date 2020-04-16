# -*- coding: utf-8 -*-
"""
create on 2020-04-14 16:37
author @66492
"""


class TreeNode(object):
    def __init__(self, key: str, val, n: int, *children):
        self.key: str = key
        self.val = val
        self.N = n
        self.children = children


class BinaryTreeNode(TreeNode):
    def __init__(self, key, val, n, left=None, right=None):
        super(BinaryTreeNode, self).__init__(key, val, n, *(left, right))
        self.left = left
        self.right = right


class BaseTree(object):
    def __init__(self):
        self.root: TreeNode = None

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError()

    def get(self, key):
        raise NotImplementedError()

    def put(self, key, val):
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()

    def keys(self):
        raise NotImplementedError()

    def rank(self):
        raise NotImplementedError()

    def delete(self, key):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()


class BinaryTree(BaseTree):
    # TODO: 为什么查找时间复杂度是lgN
    KEY_CHILDREN = "children"

    def __init__(self, root=None):
        super(BinaryTree, self).__init__()
        self.root: BinaryTreeNode = root

    def _from_dict(self, d):
        if not d:
            return None
        keys = list(d.keys())
        if self.KEY_CHILDREN in d:
            keys.remove(self.KEY_CHILDREN)
        key = keys[0]
        node = BinaryTreeNode(key, d[key], 1)
        if self.KEY_CHILDREN not in d:
            return node
        node.left = self._from_dict(d[self.KEY_CHILDREN][0])
        node.right = self._from_dict(d[self.KEY_CHILDREN][1])
        node.N = self._size(node.left) + self._size(node.right) + 1

    @classmethod
    def from_dict(cls, d):
        # TODO: valid判断
        def _from_dict(d):
            if not d:
                return None

            def get_key(d):
                if not d:
                    return "0"
                keys = list(d.keys())
                if cls.KEY_CHILDREN in d:
                    keys.remove(cls.KEY_CHILDREN)
                key = keys[0]
                return key

            key = get_key(d)
            node = BinaryTreeNode(key, d[key], 1)
            if cls.KEY_CHILDREN not in d:
                return node
            child_d = list(sorted(d[cls.KEY_CHILDREN], key=get_key))
            node.left = _from_dict(child_d[0])
            node.right = _from_dict(child_d[1])
            node.N = cls._size(node.left) + cls._size(node.right) + 1
            return node

        root = _from_dict(d)
        return cls(root)

    def _get(self, node: BinaryTreeNode, key):
        if node is None:
            return None
        if key == node.key:
            return node.val
        if key > node.key:
            return self._get(node.right, key)
        return self._get(node.left, key)

    def get(self, key):
        return self._get(self.root, key)

    def _put(self, node: BinaryTreeNode, key, val):
        # TODO: 理解
        if node is None:
            return BinaryTreeNode(key, val, 1)
        if key == node.key:
            node.val = val
        elif key > node.key:
            node.right = self._put(node.right, key, val)
        else:
            node.left = self._put(node.left, key, val)
        node.N = self._size(node.left) + self._size(node.right) + 1
        return node

    def put(self, key, val):
        # TODO: 是否进行平衡调整
        self.root = self._put(self.root, key, val)

    @staticmethod
    def _size(node: BinaryTreeNode):
        if node is None:
            return 0
        return node.N

    def size(self):
        return self._size(self.root)

    def keys(self):
        return list(self.to_dict().keys())

    def rank(self):
        raise NotImplementedError()

    def _delete(self, node: BinaryTreeNode, key):
        if node is None:
            raise ValueError("key %s is not in" % key)
        if node.key == key:
            pass

    def delete(self, key):
        # TODO: 有点复杂
        raise NotImplementedError()

    def _to_dict(self, node: BinaryTreeNode):
        if node is None:
            return {}
        d = {
            node.key: node.val,
            self.KEY_CHILDREN: [self._to_dict(node.left), self._to_dict(node.right)]
        }
        return d

    def to_dict(self):
        return self._to_dict(self.root)


if __name__ == '__main__':
    binary_tree = BinaryTree()
    binary_tree.put("A", 1)
    binary_tree.put("B", 2)
    binary_tree.put("C", 3)
    print(binary_tree.size())
    print(binary_tree.to_dict())
    d = {'A': 1,
         'children': [{}, {'B': 2, 'children': [{'D': 4, 'children': [{}, {}]}, {'C': 3, 'children': [{}, {}]}]}]}
    binary_tree = BinaryTree.from_dict(d)
    print(binary_tree.to_dict())
