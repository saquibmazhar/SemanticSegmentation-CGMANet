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
 #print(folder)
 files = os.listdir(folder)
 #print(files)
 for filename in files:
  #print(filename)
  #if '_gtFine_labelTrainIds' in filename:
  if '' in filename: 
   newname = filename.replace('.png','_gtFine_labelTrainIds.png')
   #print(newname)
   os.rename(os.path.join(folder, filename), os.path.join(folder, newname))
  else:
   os.remove(os.path.join(folder, filename))

