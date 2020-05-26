"""
Created on Fri May 22 09:50:07 2020
@author: Gael MIRAMOND & Faustine MICHEL
All the usefull  functions 
"""

# import all the usefull libraries
import os
import numpy as np
from pathlib import Path


#%% Data pre-processing

def separate_text(path,trgt_folder):
    src_obj_file=open(path,"r")# open a .obj file in reading mode
    strings_to_keep="" # Usefull information from teh .obj file
    name=""
    flag=0
    compteur =0 

    for line in src_obj_file :
        if (line[0] == 'g' and flag == 0): #the program keeps only strings startinf with a 'g'
            if (line[-3]== '_'):   
                name=line[2:-3]
            else:
                name=line[2:-4]      
        if (line[0] == 'v' and flag == 0): #the program keeps only strings startinf with a 'f' or a 'v'
            strings_to_keep=strings_to_keep+line
        if (line[0] == 'f' ): #the program keeps only strings startinf with a 'f' or a 'v'
            strings_to_keep=strings_to_keep+line
            flag = 1    
        if (line[0]=='g' and flag == 1):
            trgt_txt_file=open(trgt_folder+name+".txt",'w')
            trgt_txt_file.writelines(strings_to_keep) #paste the important lines in the new .txt file
            trgt_txt_file.close() #close the .txt file
            compteur=compteur+1
            strings_to_keep=""
            if (line[-3]== '_'):
                name=line[2:-3]
            else:
                name=line[2:-4]
            flag=0       
    trgt_txt_file=open(trgt_folder+name+".txt",'w')
    trgt_txt_file.writelines(strings_to_keep) #paste the important lines in the new .txt file
    trgt_txt_file.close() #close the .txt file
    compteur=compteur+1
    flag=0
     
    return compteur 

#%% Data post-processing
    
def list_files(src,trgt):
    """
    arg :src : path of the source directory, trgt : path of the file in wich names will be written 
    """
    liste=sorted(os.listdir(src))
    trgt_file=open(trgt,"w")
    for line in liste:
        trgt_file.write(line[0:-4]+'\n')
    trgt_file.close()
    return


def pick_name(path,line_number,class_number):
    """
    arg :path (str of the file to read) int line number, int class_number 
    Numer of lines starts with  1
    """
    with open(path,"r") as src_file :
        for i in range(line_number):
            to_write=src_file.readline()
        src_file.close()
    trgt_file=open(os.path.join(Path(r"predictions\\"),Path(str(class_number)))+".txt","a")
    trgt_file.write(to_write)
    trgt_file.close()
    return
        


def supress_class_files(path,number_of_classes):
    """
    arg :path of the folder in wich class files have to be deleted, number_of_classes (ex :0->4 = 5 classes )
    ex : supress_class_files("test/",1)
    """
    list_of_files=os.listdir(path)
    list_classes=[j for j in range(number_of_classes)] #make a list with all possible filenames made by the pickname function 
    for name in list_classes:
        if(str(name)+".txt" in list_of_files)==True:
            os.remove(path+str(name)+".txt")
    return

def collect_results_file_to_tab(path_file_predicted_labels):
    #path_file: path where we can find the predicted labels (need to finish with a .txt)
    src_file=open(path_file_predicted_labels,"r")
    tab="" 
    tab2=[] 

    for line in src_file : 
        tab=tab+line
           
    src_file.close() # close the .obj file   

    for j in range(1, len(tab)-1, 3):
        tab2.append(int(tab[j]))

    return tab2

def results_processing(path_file_names,path_file_predicted_labels):
    tab=collect_results_file_to_tab(path_file_predicted_labels)
    
    for i in range(0,len(tab)):
        pick_name(path_file_names,i+1,tab[i])
    
    return
