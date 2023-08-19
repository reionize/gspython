import stitchscan
import argparse 
import os

def getargs():
    parser = argparse.ArgumentParser(prog='stitch.py', description='Stitches together GelSight scans. See stitchscan module documentation for more details.')
    parser.add_argument('object_dirs', metavar='object', type=str, nargs='+',help='parent directory (-ies) containing all scan folders corresponding to a single object')
    parser.add_argument('-a', '--allowance', type=int, default=20, help='allowance for sampling background variation (default: 20)')
    parser.add_argument('-t', '--stack_tmd', action='store_true', help='stacks raw heightmap data when stitching instead of normalizing and blending seams')
    parser.add_argument('-r', '--ratio', type=float, default=0.7, help='ratio for filtering matches (default: 0.7)')
    parser.add_argument('-n', '--n_bins', type=int, default=10, help='number of bins to use for selecting regions with high density of matches (default: 10)')
    parser.add_argument('-l', '--levels', type=int, default=0, help='determines whether and number of levels to use for pyramid blending (default: 0)')
    parser.add_argument('-o', '--reverse_order', action='store_true', help='assume reversed order of scans (default: bottom up)')
    parser.add_argument('-d', '--diagnose', action='store_true', help='use in diagnostic mode')
    parser.add_argument('--orb', action='store_true', help='use ORB instead of SIFT to detect and compute matches (not recommended)')
    parser.add_argument('--tag', type=str, default='', help='adds a tag to the output filename')

    args = parser.parse_args()
#     print(args.object_dirs, args.allowance)
    
    return args.object_dirs, args.allowance, args.stack_tmd, args.ratio, args.n_bins, args.levels, args.reverse_order, args.diagnose, args.orb, args.tag

# main script execution
if __name__ == '__main__':
    # read args from command line
    objects, allowance, stack_tmd, ratio, nbins, levels, reverse_order, diagnose, use_orb, tag = getargs()
    normalize_height = not stack_tmd
    mode = 'orb' if use_orb else 'sift'
    
    # check if output folder exists
    if not os.path.exists('output/'):
        print('Created /output folder')
        os.mkdir('output/')
    
    # stitch scans together
    stitchscan.stitchscans(objects, allowance=allowance, normalize_height=normalize_height, ratio=ratio, nbins=nbins, levels=levels, reverse_order=reverse_order, diagnose=diagnose, mode=mode, tag=tag)