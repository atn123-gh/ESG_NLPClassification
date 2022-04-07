"""
Generate pickle and tsv files with extra features.
Used multiprocessing for faster processing.

Example:

Original data sample from train.tsv :

sid      |   sentence   |  html_id    | label
---------------------------------------------
0001     |   sentence1  |  Form10k_xx | 0


After adding new features :

sid      |   sentence   |  html_id    | label  | isTitle  | title      | paragraph_id
-------------------------------------------------------------------------------
0001     |   sentence1  |  Form10k_xx | 0      |  1       | Title Name | PaRa_xx

"""

import os
import urllib
import pandas as pd
import re
import concurrent
import numpy as np
from bs4 import BeautifulSoup
import numpy as np
import pdb


try:
       os.makedirs("fntestoutputtsv")
except:
       pass

    
class FormClazz:
    def __init__(self,htmlid,htmldf):
        print(f"---start---{htmlid}")
        self.htmlid=htmlid
        self.htmldf = htmldf
#         self.htmldf = htmldf[:1]  # for debugging
        self.outputfolder="final_testtsv"
        self.htmlfolder="final_html"
#         self.faildf=pd.DataFrame(columns=["sid","html_id","sentence","msentence"])
        with open(f"{self.htmlfolder}/{self.htmlid}.htm") as fp:
            self.soup = BeautifulSoup(fp, "lxml")
        self.block=self.assignblock()
        self.outputfilepath=f"{self.outputfolder}/{self.htmlid}.tsv"
        self.lastmatchbid="0"
        
    #remove previous block
    def decomposeBefore(self,strid):
        pid=int(strid.split("_")[1])
        for b in self.soup.find_all(self.block):
            current_id = int(b['id'].split("_")[1])
            if current_id < pid:
                b.decompose()

    def cleantext(self,txt):
        txt=(''.join(e for e in txt if e.isalnum())).lower()
#         txt=re.sub(' +', ' ',txt)
#         txt=(re.sub('[^A-Za-z0-9.?:,;!\-()_\[\]\—\"\' ]+', '', txt)).lower()
        return txt
        
    def makedataFrame(self):
        self.cleanSoup()
        self.htmldf[["isTitle","title","paragraph_id"]] =self.htmldf.apply(self.createNewFeatures,axis=1, result_type ='expand')
        self.htmldf['title'].fillna(method='ffill',inplace=True)
        self.htmldf.to_csv(self.outputfilepath, sep='\t')
        if self.htmldf['isTitle'].sum()<1:
            print(f'Notitle~ :{self.htmlid}')
        
        print(f"{self.htmlid} finished.") 
        return self.htmldf
    
    def createNewFeatures(self,rw):
        isTitle = 0
        paragraph_id="NoID"
        title = np.nan
        
        mrw=self.cleantext(rw['sentence'])
        matchb=self.soup.find(self.block, text=re.compile(mrw))
#         if len(matchbs)>2:
#             print("find -->:",matchb)
#             matchbids=[int(b['id'].split("_")[1]) for b in matchbs ]
#             print(matchbids)
#             matchbid=min(matchbids)
#             print(matchbid)
#             matchb=self.soup.find(self.block,id='PaRa_'+str(matchbid),text=re.compile(mrw))
#             print("findAlls -->:",matchb)
        
        
        #not empty array
        if matchb is not None:
            if matchb.has_attr('style'):
                #if "bold" in matcht['style'] or matcht.findAll('u') or matcht.findAll('i') or  matcht.findAll('b') :
                if "bold" in matchb['style'] :
                    isTitle = 1
                    title=rw['sentence']
            paragraph_id = matchb["id"]
            self.lastmatchbid=matchb["id"]
            self.decomposeBefore(paragraph_id)   
#             print(f"Current bid:{paragraph_id} Remaining paragraph to search {rw['html_id']} : {len(self.soup.find_all(self.block))}")
#         else:
#             print("------------------")
# #             print(f"sid:{self.soup.findAll(self.block)} \n")
#             print(f"Last found paragraph id : {self.lastmatchbid}")
#             print(f"sid:{rw['sid']} \n {rw['html_id']} \n {rw['sentence']} \n {mrw}")
#             pdb.set_trace()
#             self.faildf.append({'sid': rw['sid'],
#                                     'html_id': rw['html_id'],
#                                     'sentence': rw['sentence'],
#                                     'msentence':mrw,
#                                    },
#                                    ignore_index=True)   
        return isTitle,title,paragraph_id
        
    
    def assignblock(self):
        #remove non html tags and tags with no text
        print(f"before #remove tags with no text : {len(self.soup.find_all())}")
        
#         for x in self.soup.find_all(re.compile(":")):
#                 x.decompose()
                
        for x in self.soup.find_all():
            if len(x.get_text(strip=True)) == 0:
                x.decompose()
            if x.name == "table":
                x.decompose()
                
        print(f"after #remove tags with no text : {len(self.soup.find_all())}")

        block='p'
        ptext = ''
        dtext= ''
        divpcount=0
        # count p in div
        for ds in self.soup.find_all('div'):
            if len(ds.find_all('p')):
                divpcount=divpcount +1
            if divpcount > 30:
                break
            
         # if div has many p child tags
        if divpcount > 30:
            block = 'p'
        else:
            if len(self.soup.find_all('p'))<100:  
                block = 'div' # if p tags count is small
            elif len(self.soup.find_all('div'))<100:
                block ='p'  # if div tags count is small1
            else:
                for p in self.soup.find_all('p')[:5000]:
                    ptext=ptext+p.get_text()
                ptext=self.cleantext(ptext)
                for d in self.soup.find_all('div')[:5000]:
                    dtext=dtext+d.get_text()
                dtext=self.cleantext(dtext)
                if len(dtext) > len(ptext):  # compare contents of p and div
                    block ='div'
                else:
                    block = 'p'
        print(f"assignblock return  : {block}")
         
        return block
    

    def cleanSoup(self):
        print(f"before cleanSoup : {len(self.soup.find_all())}")
        fontweight=0
        # unwrap nested tags but preserve bold style
        print(f"before # unwrap nested tags but preserve bold style : {len(self.soup.find_all())}")
        for b in self.soup.findAll(self.block):
            # unwarp outer block which has inner div
            for tag in b.findAll():
                if tag.name == 'div':
                    b.unwrap()
                    break
                    
       # unwrap for missing tags (assume tag's closed tag is missing if there are too many child tags)
        print("before missing tags")
        for b in self.soup.findAll(self.block):
            if len(b.findAll()) > len(self.soup.body.findAll())/3:
                b.unwrap()
                      
        print("after missing tags")
        for b in self.soup.findAll(self.block):
            # making consistent bold styles for title
            for tag in b.findAll():
#                 pdb.set_trace()
                if tag.name in ['span','p'] and tag.has_attr('style'):
                    bstyle = tag['style'].replace(" ","")
                    if "bold" in tag['style']:
                        tag.parent['style'] ="font-weight: bold;"
                    elif "font-weight" in bstyle:
                        fontweight =re.findall("font-weight:(\d+)", bstyle)
                        try:
                            fontweight=int(fontweight[0])
                            if fontweight > 500:
                                tag.parent['style'] ="font-weight: bold;"
                        except:
                            pass
                        
                if tag.name in ['b','u','i','strong','a']:
                    tag.parent['style'] ="font-weight: bold;"
                tag.unwrap()
        print(f"after # unwrap nested tags but preserve bold style : {len(self.soup.find_all())}")
        # assign id to all div
        idno=1
        for b in self.soup.find_all(self.block):
            b['id']='PaRa_'+str(idno)
            idno=idno+1
#             print(b)
            
        assert idno > 100, "Too small blocks count : {idno}"
        
        
        beforecounts=len(self.soup.find_all(self.block))
        #remove special character
        for b in self.soup.find_all(self.block):
            txt=(str(b.get_text()))
            b.string=self.cleantext(txt)
        
#         pdb.set_trace()
                
        aftercounts=len(self.soup.find_all(self.block))
        assert beforecounts == aftercounts, f"Removed special character -> Before counts{beforecounts} and After counts{aftercounts} should match ."
    
        # remove unnecessary front pages
        parti = self.soup.find_all(self.block, text="parti")
        try:
            pid=int(parti[-1]['id'].split("_")[1])
        except:
            raise ValueError("# remove unnecessary front pages", self.htmlid)
        self.decomposeBefore('PaRa_'+str(pid))
        
        print(f"finish cleanSoup : {len(self.soup.find_all())}")



originaltsv="final_test.tsv"

df=pd.read_csv(originaltsv, sep='\t')


### For debugging

# class FormClazz:
#     def __init__(self,htmlid,htmldf):
#         self.htmldf=htmldf
#         print(f"---start---{htmlid}")
#     def makedataFrame(self):
#         return self.htmldf


# htmlIds=[(htmlId,df[df.html_id == htmlId]) for htmlId in df.html_id.value_counts().index.to_list()]
# for htmlid,htmldf in htmlIds:
#     print(htmlid)
#     fc=FormClazz(htmlid,htmldf)
#     dfs=fc.makedataFrame()

# fcd1=FormClazz("Form10k_54",df[df.html_id =="Form10k_54"]).makedataFrame()



# with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
#     dflist=list(executor.map(FormClazz(htmlid,df[df.html_id == htmlid]),htmlIds))


### Multiprocessing for faster processing
# ___MAIN_

values = ((htmlId,df[df.html_id == htmlId]) for htmlId in df.html_id.value_counts().index.to_list())

def runAll(htlmId,htmlDf):
    return FormClazz(htlmId,htmlDf).makedataFrame()

from multiprocessing import Pool, cpu_count
with Pool() as pool:
    results=pool.starmap(runAll, values)
    pool.close()
    pool.join()
    print(results)
    print('end')
    
    
testdf=pd.concat(results)
testdf.to_csv("final_testdf.tsv")
testdf.to_pickle("final_testdf.pkl")


###  uncomment this for sanity check for created dataset

# readdf = pd.read_pickle("testdf.pkl")
# readdf=readdf[['sid','sentence','html_id','label']]
# ##check new df against orignal df
# newdf=pd.concat([df,readdf])
# newdf = newdf.reset_index(drop=True)
# df_gpby = newdf.groupby(list(newdf.columns))
# idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1] # check if any unique records exist
# newdf.reindex(idx)  # if empty ok
   
