import os, random, shutil

def train_split_validate(dir=r'../dataset/train',valpath=r'../dataset/validate'):
    
    dir1 = dir+'/a'
    dir2=dir+'/b'
    valpath1=valpath+'/a'
    vallabelpath = valpath+'/b'
    
    files = os.listdir(dir1)
    filelength = len(files) 
    print("filelength = %d " % filelength)
     
    picklength = 48
    sample = random.sample(files, picklength)
    print("len-sample = %d " % len(sample))
     
     
    isExists = os.path.exists(valpath1)
    if not isExists:
        os.makedirs(valpath1)           
    
    for name in sample:
        labelname=os.path.splitext(name)[0] + "_segmentation.png"
        shutil.move(os.path.join(dir1, name), os.path.join(valpath1, name))
        shutil.move(os.path.join(dir2, labelname), os.path.join(vallabelpath, labelname))
    



