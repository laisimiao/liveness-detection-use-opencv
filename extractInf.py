import os
import os.path as osp
import argparse
import shutil
ap = argparse.ArgumentParser()
# ap.add_argument("-op","--origen_path",help = 'path to origen_path')
ap.add_argument("-tp","--target_path",required=True,help = 'path to taget_path')
ap.add_argument("-t","--txt",required = True,help = "train or test")
args = vars(ap.parse_args())
target_parh = args["target_path"]
# origen_path = args["origen_path"]


# print(value, ...,)
with open(args["txt"]) as f:
	lines = f.readlines()
	imlists = [osp.join(osp.realpath('.'),'ClientRaw/',line) for line in lines]
	image_path =[]
	for imlist in imlists:
		image_path.append(imlist.replace('\\','/').rstrip("\n"))
	for image in image_path:
		if os.path.exists(target_parh):
			shutil.move(image,target_parh)
		else:
			os.makedirs(target_parh)
			shutil.move(image,target_parh)
