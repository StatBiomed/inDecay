
import time
import pandas as pd
import glob
import os, sys
from inDecay import PATH

from inDecay.zygote import getselftarget, genepro_dec, decide_duplicate, decide_r2
import shutil
from Bio import SeqIO, Seq
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord
import csv
from datetime import date
import subprocess
today = date.today()
formatted_date = today.strftime("%Y%m%d")
pj=os.path.join
def create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


os.chdir(pj(PATH.embryo_raw_dir,'process'))
ana_dir = "DEC_analysis"
ice_ana_dir="ICE_analysis"
ice_st_dir="ICE_sangertraining"
raw_dir = "decodr/"
selftargetpath= "SelfTarget/"
sanger_dir="DEC_sangertraining/"
def exp_to_guide(filepath):
    df=pd.read_excel(filepath,index_col=0)
    return df.to_dict()["Guide Sequence(s)"]
def guide_to_exp(exp_to_guide):
    return {e:g for g,e in exp_to_guide.items()}


guidedisc={}
results=glob.glob(f'{raw_dir}/*.xlsx')
for result in results:
    guidedisc.update(exp_to_guide(result))
print("All data extracted:", guidedisc)




"""
write gene seq

"""
mark=[]
data=[]
for i in guidedisc.keys():
    if not getselftarget(guidedisc, i).process_guide()==False and i.split('---')[0] not in mark:
        mark.append(i.split('---')[0])
        data.append([str(i.split('---')[0]), str(getselftarget(guidedisc, i).shorten_ref), str(getselftarget(guidedisc, i).guide)])
    else:
        continue
       
with open('gene_seq_all.csv', "w") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['folder','seq','gDNA'])
    writer.writerows(data)

"""write run_selftarget.sh"""
create(selftargetpath)
script_name = f"{formatted_date}_run_selftarget.sh"
with open(script_name, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"cd {PATH.Indelana}\n")  # Change directory if needed
    commands=[]
    for file in guidedisc.keys():
        commands.append(getselftarget(guidedisc, file).selftarget+"\n")
    
    f.writelines(set(commands))
    f.write(f"cd -\n")

# Make the script executable
os.chmod(script_name, 0o755)

subprocess.run(['bash', script_name])

create(sanger_dir)
"""transform decodr results (for other species, filter low r2) to inDecay format """
final_fail=[]
for i in [j for j in os.listdir(ana_dir) if '.DS_Store' not in j]:
    dfr2 = pd.read_csv(pj(ana_dir, i))['rSquared'].iloc[-1]
    if  not i.startswith('m') and dfr2 < 0.8:
        print("Dropped files by r2:", i, dfr2)
    else:
        genepro_dec(guidedisc, i.replace('.csv',''), ana_dir, selftargetpath, sanger_dir)
        # except:
        #     final_fail.append(i)  
 
# addice={}
# for i in set(os.listdir(ice_ana_dir))-set(os.listdir(ana_dir)):
#     r2=pd.read_csv(pj(ice_ana_dir,i)).iloc[-1]['r2']
#     if pd.read_excel(pj(raw_dir, i.split('---')[0]+'.xlsx'))['Sample Type'].iloc[-1]=='Bulk' or r2>=0.95:
#         addice[i]=r2
#         print("ADD: ", i)
#         shutil.copy2(pj(ice_st_dir, i.replace('.csv','_SelfTarget.csv')), pj(sanger_dir, i.replace('.csv','_SelfTarget.csv')))

    

    
# print('failed merged files:', final_fail)

sumfolder={}
folder=list(set([i.split('---')[0] for i in os.listdir(sanger_dir) if '.DS_Store' not in i and '*' not in i]))

for i in folder:
    sumfolder.update({i: [j for j in os.listdir(sanger_dir) if i in j]})



"""merged files by folder"""

create('DEC_sum')
seq='Sanger'
san_all_dir='DEC_sum/'+seq+'/'
create(san_all_dir)
with open('DEC_sum/NUM_'+seq+'.csv', 'w') as fp:
    writer=csv.writer(fp,delimiter=',')
    writer.writerow(['folder','num','files'])
    for k, v in sumfolder.items():
        df=[]
        df_files=[]
        r2=[]
        num=0
        for c, iv in enumerate(v):
            dfi=pd.read_csv(sanger_dir+iv)
            df.append(dfi)
            df_files.append(iv.replace('_SelfTarget',''))
            if os.path.exists(ana_dir+'/'+iv.replace('_SelfTarget','')):
                r2i=pd.read_csv(ana_dir+'/'+iv.replace('_SelfTarget',''))
                r2.append(r2i['rSquared'].iloc[-1])
            else:
                r2.append(addice[iv.replace('_SelfTarget','')])
            num=num+1
        if not len(df)==0:
            df=pd.concat(df)
            df_all=df.groupby([df['N_gt'], df['loc'], df['indel_size'], df['Indelgen_seq'], df['Identifier']])['Count'].sum()
            df_all.to_csv(san_all_dir+k+'.csv')
            df_write=[[df_files[i], r2[i]] for i in range(len(r2))]
            data=[k, num]
            data.append(df_write)
            writer.writerow(data)


"""archive all genes to csv"""
all2024=pd.read_csv('gene2024_archive_SampleType.csv',index_col=0)
all2024=all2024.drop('guide',axis=1)
all2025=pd.read_csv('gene2025.csv')
all2025['SampleType']=all2025['SampleType'].apply(lambda x: x[0].upper()+x[1:])
allsum=pd.concat([all2024,all2025])
allsum['count']=allsum['count'].fillna(0).astype('int')
gene_seq=pd.read_csv('gene_seq_all.csv',index_col=0)


allsum['guide']=allsum['folder'].apply(lambda x: x.split('_')[0])
allsum['date']=allsum['folder'].apply(lambda x: x.split('_')[-1])
species_dict={'m':'mouse','p':'porcine','s':'goat','c':'cattle'}
allsum['species']=allsum['folder'].apply(lambda x: species_dict[x[0]])
allsum['seq']=allsum['folder'].apply(lambda x: gene_seq['seq'].loc[x])
select_mark=['R','F','M']
allsum['duplicate']=allsum['date'].apply(lambda x: 1 if x[-1] in select_mark else '-')


dfolder=allsum[allsum['duplicate']==1]
dfolder['duplicate']=dfolder['date'].apply(lambda x:x[-1])
dfolder['datemark']=dfolder['folder'].apply(lambda x:x[0:-1])
dfolder=dfolder.set_index('folder')
allsum=allsum.set_index('folder')

print('duplicated:', dfolder)
current_files=pd.read_csv('DEC_sum/NUM_'+seq+'.csv',index_col=0)
for i in current_files.index:
    if allsum['duplicate'].loc[i]==1 and i not in decide_duplicate(dfolder, seq).keys():
        print('Dropped by duplicate:', current_files.loc[i])
        current_files=current_files.drop(index=i)
allsum.to_csv('geneall.csv')
current_merge=pd.merge(current_files, allsum, left_index=True, right_index=True)
current_merge['num']=current_merge['num'].astype('str')
current_merge['count']=current_merge['count'].astype('str')
current_merge=current_merge.reset_index()
current_files=current_files.reset_index()
# lost.update({seq: allsum[~allsum['folder'].isin(current_merge['folder'])]})
grouped=current_merge.groupby('guide')

result = grouped.agg({
'count': lambda x: ','.join(x),
'files': lambda x: ','.join(x.replace('[','').replace(']','')),
'num': lambda x: ','.join(x),
# 'r2': lambda x: ','.join(x),
# 'SampleType': lambda x: '+'.join(x.unique()),
'SampleType': lambda x: ','.join(x),
'date': lambda x: ','.join(x),
'folder': lambda x: ','.join(x),
'seq': lambda x: x.unique(),
'gDNA': lambda x: x.unique()
})
current_selected= [i for i in result.index]
current_summary=result.loc[current_selected]
current_summary[['seq','gDNA']]=current_summary[['seq','gDNA']].applymap(lambda x: str(x).replace('[','').replace(']',''))
current_summary.to_csv('DEC_sum/SUM_'+seq+'.csv')

sum_all_dir='DEC_sumall_final/'
create(sum_all_dir)
    
for j in [k for k in os.listdir('DEC_sum/') if '.DS_Store' not in k and '.csv' not in k]:
    create(sum_all_dir+j)
    gene_sum_folder={}
    gene_folder_num={}
    summary=pd.read_csv('DEC_sum/SUM_'+j+'.csv',index_col=0)

    for i in summary.index:
        gene_sum_folder.update({i:[k+'.csv' for k in summary['folder'].loc[i].split(',')]})
        if summary['SampleType'].loc[i]=='Clonal':
            num=[int(summary['num'].loc[i])]
        elif summary['SampleType'].loc[i] =='Bulk':
            num=[int(summary['count'].loc[i])]#*int(summary['num'].loc[i])
        else:
            clonalnum=0
            sample_types = summary['SampleType'].loc[i].split(',')
            counts = summary['num'].loc[i].split(',')
            bulk_counts = summary['count'].loc[i].split(',')
            num = [int(bulk_counts[k] if x == 'Bulk' else counts[k]) 
                    for k, x in enumerate(sample_types)]
        
            # bulkloc=[k for k, x in enumerate(summary['SampleType'].loc[i].split(',')) if x == 'bulk']
            # clonalloc=[k for k, x in enumerate(summary['SampleType'].loc[i].split(',')) if x == 'clonal']

            # clonalnum= [int(summary['num'].loc[i].split(',')[k]) for k in clonalloc]
            # bulknum=[int(summary['count'].loc[i].split(',')[k]) for k in bulkloc] #*int(summary['num'].loc[i].split(',')[k]) for k in bulkloc]
            # num = [for k, x in enumerate(summary['SampleType'].loc[i].split(','))]
        gene_folder_num.update({i:num})
        # print(gene_folder_num)
    with open('DEC_sum/DICT_'+j+'.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(['Item','Files','SampleType','Nums','Events'])
        for key, value in gene_sum_folder.items():
            if not len(value)==0:
                writer.writerow([key, value,summary['SampleType'][key], gene_folder_num[key]])#,summary['n_dec_embryos'][key], ])


count_sum_all_dir='DEC_sumall_count/'
create(count_sum_all_dir)
count={}

# for r2 in [0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0]:
r2=0
for j in [k for k in os.listdir('DEC_sum/') if '.DS_Store' not in k and '.csv' not in k]:
    if not os.path.exists(f"{count_sum_all_dir}{j}_{r2}"):
        os.mkdir(f"{count_sum_all_dir}{j}_{r2}")
    checkcsv=pd.read_csv('DEC_sum/DICT_'+j+'.csv',index_col=0)
    count_list=[]
    for i in range(checkcsv.shape[0]):
        df=[]
        s=checkcsv['Files'][i].replace("'","").replace('[','').replace(']','').replace(' ','')
        sfolder=s.split(',')[0:]
        n=checkcsv['Nums'][i].replace("'","").replace('[','').replace(']','').split(',')[0:]
        n=[float(u) for u in n]
        nfolder= sum(n)
        for idx, iv in enumerate(sfolder):
            dfi=pd.read_csv('DEC_sum/'+j+'/'+iv)
            if checkcsv['SampleType'][i].split(',')[idx] == 'Bulk':
                dfi['Count']=dfi['Count'] * n[idx]  # bulk * embryocount, clonal * filecount
            df.append(dfi)
        df=pd.concat(df)
        df_all=df.groupby([df['N_gt'], df['loc'], df['indel_size'], df['Indelgen_seq'], df['Identifier']])['Count'].sum().reset_index()
        df_all=df_all.sort_values(by=['Count'],ascending= False)
        count_list.append(df_all[df_all['Identifier']!='Not Present']['Count'].sum())
        # if df_all[df_all['Identifier']!='Not Present']['Count'].sum()<500 and dropcount==1:
        #     print('failed count:', checkcsv.index[i], df_all[df_all['Identifier']!='Not Present']['Count'].sum())
        # if checkcsv.index[i] in decide_r2(seq=j, r2=r2).keys():
        #     print('failed r2:', checkcsv.index[i], decide_r2(seq=j, r2=r2)[checkcsv.index[i]])
        if checkcsv.index[i] in decide_r2(seq=j, r2=r2).keys():
            print(f"Drop {checkcsv.index[i]} in {r2}")
        else:
            df_all.to_csv(f"{count_sum_all_dir}{j}_{r2}/{checkcsv.index[i]}_SelfTarget.csv")
    count[f"{r2}-{j}"]=count_list

second_dir='zygote_noice'
embryodir=pj(PATH.data_dir, second_dir)
create(embryodir)
training_csv= pd.read_csv('DEC_sum/SUM_Sanger.csv')[['guide','seq']].replace("'","")
training_csv['r2']=decide_r2('Sanger',1).values()
training_csv['count']=count['0-Sanger']
training_csv['seq']=training_csv['seq'].apply(lambda x: x.replace("'",""))
training_csv.to_csv(pj(embryodir, 'gene_seq.csv'))
# print(training_csv)

final_select='Sanger_0'
species_dict={'m':'mouse','p':'porcine','s':'goat','c':'cattle'}

for j in os.listdir('DEC_sumall_count/'+final_select):
    shutil.copy2(pj('DEC_sumall_count/'+final_select,j),pj(embryodir,j))
