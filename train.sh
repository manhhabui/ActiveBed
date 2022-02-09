# CUDA_VISIBLE_DEVICES="1" python main.py --config "algorithms/RandOD/configs/PascalVOC.json" --exp_idx "1" 
tensorboard --logdir "/home/ubuntu/source_code/ActiveBed/algorithms/LLAL/results/tensorboards/CIFAR_10_1"
# python main.py --config "algorithms/LLALOD/configs/PascalVOC.json" --exp_idx "0"
# for i in {1..5}; do
#     CUDA_VISIBLE_DEVICES="1" python main.py --config "algorithms/LLALOD/configs/PascalVOC.json" --exp_idx $i
# done