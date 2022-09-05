import _libfcs_ext

if __name__ == '__main__':
    print('Small test file:')
    fcs = _libfcs_ext.FCS('../libfcs/tests/fcs_files/attune_small.fcs')
    print(fcs.uncompensated)
    del fcs
    print('Large test file:')
    fcs = _libfcs_ext.FCS('../libfcs/tests/fcs_files/attune.fcs')
    print(fcs.uncompensated)
