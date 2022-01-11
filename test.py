  
with open('C:/Users/rhjva/photoswitches_255k_new.txt',) as f:
  for i, line in enumerate(f.readlines()):
    print(line.strip().split()[0])