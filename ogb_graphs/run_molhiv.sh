# python main_molhiv.py --config 'MOLHIV_LapPE_Hyperboloid_HGCN.json' | tee molhiv1.txt
# python main_molhiv.py --config 'MOLHIV_LapPE_Hyperboloid_HNN.json' | tee molhiv2.txt
# python main_molhiv.py --config 'MOLHIV_LapPE_PoincareBall_HGCN.json' | tee molhiv3.txt
# python main_molhiv.py --config 'MOLHIV_LapPE_PoincareBall_HNN.json' | tee molhiv4.txt
# python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN.json' | tee molhiv5.txt
# python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HNN.json' | tee molhiv6.txt
# python main_molhiv.py --config 'MOLHIV_RWPE_PoincareBall_HGCN.json' | tee molhiv7.txt
# python main_molhiv.py --config 'MOLHIV_RWPE_PoincareBall_HNN.json' | tee molhiv8.txt




python main_molhiv.py --config 'MOLHIV_LapPE_Hyperboloid_HGCN_mean.json' | tee molhiv_mean.txt
python main_molhiv.py --config 'MOLHIV_LapPE_Hyperboloid_HGCN_max.json' | tee molhiv_max.txt
python main_molhiv.py --config 'MOLHIV_LapPE_Hyperboloid_HGCN_sum.json' | tee molhiv_sum.txt