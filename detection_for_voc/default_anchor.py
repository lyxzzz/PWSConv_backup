import math
 
min_dim = 320   #######维度
mbox_source_layers = ["block4", "block5", "block6", "block7", "block8", "block9"] 
# in percent %
min_ratio = 10 
max_ratio = 90
small_object = [5, 10]

step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in range(min_ratio, max_ratio + 1, step): 
  min_sizes.append(min_dim * ratio / 100.)
  print(ratio, ":", min_sizes)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * small_object[0] / 100.] + min_sizes
max_sizes = [min_dim * small_object[1] / 100.] + max_sizes
# steps = [8, 16, 32, 64, 100, 300]  
# aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
 
print(min_sizes)
print(max_sizes)