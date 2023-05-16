# python main_molhiv.py --config 'MOLHIV_LapPE_Hyperboloid_HGCN.json'
# python main_molhiv.py --config 'MOLHIV_LapPE_Hyperboloid_HNN.json'
# python main_molhiv.py --config 'MOLHIV_LapPE_PoincareBall_HGCN.json'
# python main_molhiv.py --config 'MOLHIV_LapPE_PoincareBall_HNN.json'

# python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN.json'
# python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HNN.json' 
# python main_molhiv.py --config 'MOLHIV_RWPE_PoincareBall_HGCN.json' 
# python main_molhiv.py --config 'MOLHIV_RWPE_PoincareBall_HNN.json' 

# python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN_1.json' | tee molhiv_ablation_1.txt
python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN_2.json' | tee molhiv_ablation_2.txt
python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN_3.json' | tee molhiv_ablation_3.txt
python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN_4.json' | tee molhiv_ablation_4.txt
python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN_5.json' | tee molhiv_ablation_5.txt
python main_molhiv.py --config 'MOLHIV_RWPE_Hyperboloid_HGCN_6.json' | tee molhiv_ablation_6.txt