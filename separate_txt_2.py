# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:16:00 2020

@author: ASUS
"""

import pandas as pd
import os
import glob
def separate_text(path):
    src_obj_file=open(path,"r")# open a .obj file in reading mode
    strings_to_keep="" # Usefull information from teh .obj file
    name=""
    flag=0
    compteur =0 

    for line in src_obj_file :

        if (line[0] == 'g' and flag == 0): #the program keeps only strings startinf with a 'g'
            # if (line[-2]== '_'):
            #     name=line[2:-3]
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
                
            trgt_txt_file=open(name+".txt",'w')
            trgt_txt_file.writelines(strings_to_keep) #paste the important lines in the new .txt file
            trgt_txt_file.close() #close the .txt file
            compteur=compteur+1
            strings_to_keep=""
            if (line[-3]== '_'):
                name=line[2:-3]
            else:
                name=line[2:-4]
            
            flag=0

        
    trgt_txt_file=open(name+".txt",'w')
    trgt_txt_file.writelines(strings_to_keep) #paste the important lines in the new .txt file
    trgt_txt_file.close() #close the .txt file
    compteur=compteur+1
    flag=0
     
    return compteur 

path=r"C:\Users\ASUS\Anaconda3\envs\tensorflow_cpu\program\test.obj"   

separate_text(path)    
    