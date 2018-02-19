class HugeTpf(object):
    """keeps ids of the largest TPF files; one typically removes the largest 
    files to limit the usage of RAM memory"""
    def __init__(self, n_huge=None, campaign=91):
        if campaign in [91, 92]:
            list_huge = ['200070438', '200070874', '200069673', '200071158']
            if n_huge is None:
                n_huge = 2
        elif campaign in [111, 112]:
            list_huge = ['235785614']
            if n_huge is None:
                n_huge = 0
        else:
            raise ValueError('Campaign {:} not yet coded in HugeTpf.__init__()'.format(campaign))
        if n_huge > len(list_huge):
            raise IndexError('wrong parameter of n_huge in HugeTpf.__init__() of {:} (max.: {:})'.format(n_huge, len(list_huge)))
        self.huge_ids = list_huge[:n_huge]
        self.huge_ids_int = [int(epic) for epic in self.huge_ids]
