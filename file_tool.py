import os

def create_folder_withsub(folder_name,sub_folder_list=[],path_to_folder=""):
    if not os.path.isdir(os.path.join(path_to_folder,folder_name)):
        os.makedirs(os.path.join(path_to_folder,folder_name))
    for sub_folder in sub_folder_list:
        if not os.path.isdir(os.path.join(path_to_folder,folder_name,sub_folder)):
            os.makedirs(os.path.join(path_to_folder,folder_name,sub_folder))

def read_folder(folder_path,name_start=0,name_end=0):
    path_list=[]
    name_list=[]
    for file in os.listdir(folder_path):
        path_list.append(os.path.join(folder_path,file))
        if(name_start==name_end):
            name_list.append(file)
        else:       
            name_list.append(file[name_start:name_end])
            
    return sorted(path_list),sorted(name_list)

def check_if_list_name_match(lists_name=[]):
    flag=True
    for i in range(0,len(lists_name)-1):
        for j in range(i+1,len(lists_name)):
            setij=set(lists_name[i])-set(lists_name[j])
            setji=set(lists_name[j])-set(lists_name[i])
            setunion=setij.union(setji)
            
            if len(setunion)!=0: 
                flag=False
                print("list",i," do not match",j,setij,setji)
            else:
                print("list",i," match",j)
                
    return flag,setunion
