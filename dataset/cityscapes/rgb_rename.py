import os
path1 = []
rootdir = os.getcwd()
for file in os.listdir(rootdir):
 d = os.path.join(rootdir, file)
 if os.path.isdir(d):
  path1.append(d)

for dirs in path1:
 folder = os.path.abspath(dirs)
 files = os.listdir(folder)
 for filename in files:
  if '_leftImg8bit' in filename:
   newname = filename.replace('_leftImg8bit','')
   os.rename(os.path.join(folder, filename), os.path.join(folder, newname))
