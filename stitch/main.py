import stitchscan

# parent = './testdata/seagrass120223_processed'
parent = input('Input parent directory: ')
# data = ['Leaf2side1_n', 'Leaf2side2_n', 'Leaf3side1_y', 'Leaf3side2_y', 'Leaf4side1_y', 'Leaf4side2_y', 'Leaf5side1_n', 'Leaf5side2_n']
# data = ['groundtruth1_y']
data = input('Input list of directories corresponding to objects to analyze, separated by comma: ').split(',')
diagnostic = input('Use diagnostic mode? (y/n) ')
stitchscan.stitchscans(parent,data,diagnose=diagnostic=='y')