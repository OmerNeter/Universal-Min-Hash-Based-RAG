#!/usr/bin/env pypy
from itertools import combinations
from datasketch import MinHash, WeightedMinHashGenerator
from numpy import zeros, put

def flatten_tree(d, keys=[]):
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            result.extend(flatten_tree(v, keys + [k]))
        else:
            result.append(tuple(keys + [k, v]))
    return result

class SimilarityHashing:
    def __init__(self, structure, size, weighted_option='None'):
        structures = {'set': self.set_hashing,
                      'sequence': self.sequence_hashing,
                      'tree': self.tree_hashing,
                      'weighted_set': self.weighted_set_hashing}
        if structure not in structures.keys():
            raise Exception("Unsupported structure! please use one of the following: ['set', 'sequence', 'tree', 'weighted_set']")

        self.structure_name = structure
        self.structure_function = structures.get(structure)
        self.num_hash = size #4096
        sample_size = size #64
        weighted_set_options = {'ports': 65536, 'english':100, 'sfc': 10}
        if structure == 'weighted_set':
            if weighted_option == False:
                raise Exception("need to determine support for weighted hashing")
            else:
                if weighted_option in weighted_set_options.keys():
                    self.weighted_set_option = weighted_set_options.get(weighted_option)
                    self.weighted_hash_gen = WeightedMinHashGenerator(self.weighted_set_option ,
                                                                      sample_size=sample_size)
                else:
                    raise Exception("unsupported option for weighted sets")


    def set_hashing(self, data):
        hash_gen = MinHash(num_perm=self.num_hash)
        for obj in data:
            hash_gen.update(str(obj).encode('utf8'))
        return hash_gen

    def weighted_set_hashing(self, data):
        vector = zeros(self.weighted_set_option)
        put(vector , [ int(key) for key in data.keys() ] , list(data.values()))
        hash = self.weighted_hash_gen.minhash(vector.tolist())
        return hash

    def sequence_hashing(self, data):
        """
        A possible generaliztion of normalized kendall tau similarity.
        works best on long sequences.
        """
        data = combinations(data, 2)
        hash = self.set_hashing(data)
        return hash

    def tree_hashing(self, data):
        """
        A set of tree branches (sequences), which in turn becomes a set of tuples.
        """
        root_lst = list(data.keys())
        if len(root_lst) == 0:
            return []
        if len(root_lst) > 1:
            raise Exception("A tree has only 1 root node!")
        else:
            list_of_sequences = flatten_tree(data)
            list_of_pairs = [pair for branch in list_of_sequences for pair in combinations(branch, 2)]
            tree_hash = self.set_hashing(list_of_pairs)
        return tree_hash

    def hash_data(self, data):
        hash_value = self.structure_function(data)
        return hash_value


if __name__ == '__main__':
    SIZE = 4096
    set_hash = SimilarityHashing('set', size=SIZE).hash_data({1, 2, 3})
    weighted_set_hash = SimilarityHashing('weighted_set', size=64, weighted_option='ports').hash_data({'1':1, '2':2, '4':3})
    sequence_hash = SimilarityHashing('sequence', size=SIZE).hash_data((1, 2, 3))
    tree_hash = SimilarityHashing('tree', size=SIZE).hash_data({1: {9:10, 2: {3:6}, 4:{5:{7:8}}}})

    # test
    print(set_hash.jaccard(SimilarityHashing('set', size=SIZE).hash_data({1, 2, 4})), ' ', # 1 different object
          set_hash.jaccard(SimilarityHashing('set', size=SIZE).hash_data({1, 2, 3})), ' ', # identical
          set_hash.jaccard(SimilarityHashing('set', size=SIZE).hash_data({1, 4, 5})), ' ', # 2 different object
          set_hash.jaccard(SimilarityHashing('set', size=SIZE).hash_data({0, 4, 5}))) # 3 different object

    print(weighted_set_hash.jaccard(SimilarityHashing('weighted_set', size=64, weighted_option='ports').hash_data({'1':1, '2':2, '4':3})), ' ', # identical
          weighted_set_hash.jaccard(SimilarityHashing('weighted_set', size=64, weighted_option='ports').hash_data({'1':2, '2':2, '3':1})))

    print(sequence_hash.jaccard(SimilarityHashing('sequence', size=SIZE).hash_data((1, 2, 4))), ' ', # 1 different object
          sequence_hash.jaccard(SimilarityHashing('sequence', size=SIZE).hash_data((1, 3, 2))), ' ', # 1 order change
          sequence_hash.jaccard(SimilarityHashing('sequence', size=SIZE).hash_data((2, 1, 3))), ' ', # 1 order change
          sequence_hash.jaccard(SimilarityHashing('sequence', size=SIZE).hash_data((3, 2, 1)))) # total order change

    print(tree_hash.jaccard(SimilarityHashing('tree', size=SIZE).hash_data({1: {9:10, 2: {3:6}, 4:{5:{7:8}}}})), ' ', # identical
          tree_hash.jaccard(SimilarityHashing('tree', size=SIZE).hash_data({1: {2: {3:6}, 4:{5:{7:8}}}})), ' ', # remove edge
          tree_hash.jaccard(SimilarityHashing('tree', size=SIZE).hash_data({1: {9:{10:11}, 2: {3:6}, 4:{5:{7:8}}}})), ' ', # add edge
          tree_hash.jaccard(SimilarityHashing('tree', size=SIZE).hash_data({1: {4:{5:{7:8}}}}))) # remove multiple edges
