from MCPM import hugetpf

def test_1():
    htpf_1 = hugetpf.HugeTpf(n_huge=2, campaign=91)
    htpf_2 = hugetpf.HugeTpf(n_huge=4, campaign=92)
    
    assert htpf_1.huge_ids == ['200070438', '200070874']
    assert htpf_2.huge_ids_int == [200070438, 200070874, 200069673, 200071158]
    