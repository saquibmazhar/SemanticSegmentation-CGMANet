import os

#path = 'bonn/'
#folder = os.path.abspath(os.path.join(os.getcwd(), path))

path1 = []
#rootdir = os.getcwd()
rootdir = './train'
for file in os.listdir(rootdir):
 d = os.path.join(rootdir, file)
 if os.path.isdir(d):
  path1.append(d)
print(path1)

for dirs in path1:
 folder = os.path.abspath(dirs)
 files = os.listdir(folder)
 for filename in files:
  #print(filename)
  #if '_gtFine_labelTrainIds' in filename:
  if '.png' in filename: 
   newname = filename.replace('.png','_leftImg8bit.png')
   os.rename(os.path.join(folder, filename), os.path.join(folder, newname))
  else:
   os.remove(os.path.join(folder, filename))

