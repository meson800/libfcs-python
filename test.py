import _libfcs_ext

if __name__ == '__main__':
    print('Small test file:')
    _libfcs_ext.loadFCS('../libfcs/tests/fcs_files/attune_small.fcs')
    print('Large test file:')
    _libfcs_ext.loadFCS('../libfcs/tests/fcs_files/attune.fcs')
