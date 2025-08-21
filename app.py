import os
import re
import uvicorn
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json
from typing import TypedDict, Annotated, List, Optional
import asyncio
from sse_starlette.sse import EventSourceResponse
import random
import traceback
import uuid
import io
import zipfile
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from typing import Dict, Any, TypedDict, Annotated, Tuple
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from sklearn.cluster import KMeans
from contextlib import redirect_stdout


load_dotenv()

app = FastAPI()

log_stream = asyncio.Queue()

sessions = {}
final_reports = {} # Use a dictionary for final reports, keyed by session ID
#below: deep, esoteric and obscure pseudoscience stuff

reactor_list = [

    "( Se ~ Fi )",
    "( Se oo Si )",
    "( Se ~ Fi ) oo Si",
    "( Si  ~ Fe ) oo Se",
    "( Si oo Se )",
    "( Ne - > Si ) ~ Fe",
    "( Ne ~ Te ) | ( Se ~ Fe )",
    "( Ne ~ Fe )",
    "( Ne ~ Ti ) | ( Se ~ Fi )",
    "( Ne ~ Fi ) | ( Se ~ Ti )",
    "( Fe oo Fi )",
    "( Fi oo Fe ) ~ Si",
    "( Fi -> Te ) ~ Se ",
    "( Te ~ Ni )",
    "( Te ~ Se ) | ( Fe ~ Ne )",
    "( Si ~ Te ) | ( Ni ~ Fe )",
    "Si ~ ( Te oo Ti )",
    "(Si ~ Fe) | (Ni ~ Te)",
    "( Fe ~ Si | Te ~ Ni )",
    "( Fi oo Fe )",
    "( Fe oo Fi ) ~ Ni",
    "( Se -> Ni ) ~ Fe",
    "( Ni -> Se )",
    "( Se ~ Fi ) | ( Ne ~ Ti )",
    "Ni ~ ( Te -> Fi )",
    "( Se ~ Te ) | ( Ne ~ Fe )",
    "( Se ~ Ti )",
    "( Ne ~ Ti ) | ( Se ~ Fi)",
    "( Te oo Ti )",
    "( Ti oo Te ) ~ Ni",
    "Fi -> ( Te oo Ti )",
    "( Fe -> Ti ) ~ Ne",
    "( Ti ~ Ne ) | ( Fi ~ Se )",
    "( Fi ~ Se ) | ( Ti ~ Ne )",
    "( Ne ~ Fi ) | ( Se ~ Ti )",
    "( Fi ~ Ne | Ti ~ Se )"
]



class FunctionMapper:



    def table(self, formula: str):

        prompts = []
        formula = formula.strip()

        if ' | ' in formula:
            parts = formula.split(' | ')
            for part in parts:
                prompts.extend(self.table(part))
            return prompts

        tokens = re.findall(r'\b(?:Se|Si|Ne|Ni|Te|Ti|Fe|Fi|ns|sn|tf|ft)\b|~|oo|->', formula, re.IGNORECASE)
        
        op_map = {'~': 'orbital', 'oo': 'cardinal', '->': 'fixed'}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if i + 2 < len(tokens) and tokens[i+1] in op_map:
                func1 = tokens[i]
                op_symbol = tokens[i+1]
                func2 = tokens[i+2]
                op_name = op_map[op_symbol]
                
                method_name = f"{func1.lower()}_{func2.lower()}_{op_name}"
                method_name_rev = f"{func2.lower()}_{func1.lower()}_{op_name}"

                if hasattr(self, method_name):
                    prompts.append(getattr(self, method_name)())
                    i += 3
                    continue
                elif hasattr(self, method_name_rev):
                    prompts.append(getattr(self, method_name_rev)())
                    i += 3
                    continue

            if hasattr(self, token.lower()):
                prompts.append(getattr(self, token.lower())())
            
            i += 1
            
        return prompts

  

    def ni_se_fixed(self):

        identity =  """

            You are an intelligent agent. Your task is to make one plan based on a multplicity of data perceived in the present, aswell as your intentions, narratives and suspicions. You always take present action in relation to what is a viable next step in your own narrative. You turn many observations of the environment into one impression.

        """

        prompt = """
            Narratives:

                {narratives}
            
            Present:

                {environment_data}

            Answer:

            """


        return identity, prompt
   

    def se_ni_fixed(self):

        identity = """

            You are an intelligent agent. Your task is to turn one plan into many actions based on the impressions, intentions and suspicions you have aswell as you can perceive in the present. You always take present action defending what you wan't to do with your own narrative.

        """

        prompt = """
            Narratives:

                {narratives}
            
            Present:

                {environment_data}

            Answer:
        """

        return identity, prompt


    
    def se_si_cardinal(self):

        identity = """

            You are an intelligent agent. Your task is to take action based on present data, and then on what you remember to be similar to what you are currently experiencing. You always take present action seeking to take many actions to change what you already lived. You use your memories to take many different actions.
        """

        prompt = """

            Memories:

                {memories}


            Present:

                {environment_data}


            Answer:
            """
        
        return identity, prompt


    def si_se_cardinal(self):


        identity =  """

            You are an intelligen agent. Your task is to match your present actions with past perceived senssations. You always take present action seeking to change things in the present so they are sinmilar to the past. 
        """

        prompt = """

            Memories:

                {memories}


            Present:

                {environment_data}


            Answer:

        """

        return identity, prompt



    def ni_fi_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to judge your narratives, intentions and suspicions in accordance to which one is best to your own assesment of importance. You are not a doer, so you always take present action in relation to what maybe will improve your sense of self-importance in the future.
            You always take decisions by making one moral narrative based on what you judge important. Your aim is to be regarded as a person who can make philosophical statements to give everyone future sucesss.
        """

        prompt = """

            Things you find important:

                {important_things}

            Narratives:

                {narratives}


            Answer: 
        """

        return identity, prompt


    def ni_fe_orbital(self):


        identity =  """

            You are an intelligent agent. Your task is to make a narrative based on what many other people think is important. You are not a doer, so you always take present action in relation to what will maybe improve your social esteem in the future.
            You always take decisions by making one plan  based on what other people judged important. Your aim is to be regarded as an advocate for the sake of everyones future.

        """

        prompt = """
       
            Things other people find important:

                {important_things}

            Narratives:

                {narratives}

            Answer:
        

        """

        return identity, prompt



    def ni_te_orbital(self):


        identity = """

            You are an intelligent agent. Your task is to make a plan based on rational data and consensus logical thinking. You are not a doer so you always take present action in relation to what will maybe improve your own reputation and capacity in the future. Your aim is to be regarded as a good planner.
            You turn many sources of rational data into one impression that can give progress to your personal narrative.

        """
        prompt = """
            Rational data:

                {rational_data}


            Narratives:

                {narratives}


            Answer:
        """
        
        return identity, prompt
    

    def ni_ti_orbital(self):


        identity = """

            You are an intelligent agent. Your task is to make a narrative on the basis of integrity and logical statements. You are not a doer, so you always take present action in relation to what will maybe improve your capacity to keep acting coherently and with integrity in the future. Your aim is to be regarded as an incorruptible and sound visionary for matters of humanity.

        """

        prompt = """
            Logical data:

                {logical_data}
            

            Narratives: 

                {narratives}


            Answer:
        """ 

        return identity, prompt


    def se_ti_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to take many inmediate actions on the basis of logical statements and whats inmediately verifiable to be true in the environment data. You are not a planner so you always take present action in relation to what will inmediately demonstrate a capacity to behave with common sense in the current situation. Your aim is to be regarded as a quick thinker, a fighter and a quick problem solver who wants to know if the things present are real or not. 
        """

        prompt = """

            Logical data:

                {logical_data}


            Environment data:

                {environment_data}
            
            Answer:
        """


        return identity, prompt


    def se_te_orbital(self):


        identity =  """

            You are an intelligent agent. Your task is to take many inmediate actions on the basis of rational data and quantitative thinking. You are not a planner, so you always take present action in relation to what will inmediately improve your reputation. Your aim is to be regarded as someone who can do anything to achieve success.
        """

        prompt = """
        

            Rational data:

                {rational_data} 

           Environment data:

                {environment_data}             

            Answer:
        """

        return identity, prompt
    

    def se_fi_orbital(self):


        identity =  """

            You are an intelligent agent. Your task is to take many inmediate actions on the basis of what you personally previously found to be important. You are not a planner, so you always take present action in relation to what will inmediately improve your own sense of esteem. Your aim is to be regarded as a good performer who can impose of themselves any role. 

        """

        prompt = """
            Data you find important: 

                {important_data}

            Environment data:

                {environment_data}

            Answer:

        """


        return identity, prompt
    
    def se_fe_orbital(self):


        identity = """

            You are an intelligent agent. Your task is to take many inmediate actions on the basis of what other people find to be important. You are not a planner, so you always take present action in relation to what what will inmediately improve the esteem that other people have on you. Your aim is to be regarded as a fighter and someone who can do anything other people find important.

        """


        prompt = """
            Data other people find important:

                {important_data}

            Environment data:

                {environment_data}

            Answer:

        """


        return identity, prompt


    def si_ne_fixed(self):


        identity = """  
        
            You are an intelligent agent. Your task is to accumulate rich experiences you can later reflect upon. You always take present action by associating  present data with past memory and replicating what you have experienced in the past.
        """

        prompt = """
            Environment data:

                {environment_data}
            

            Memories:

                {memories}


            Answer:

        """


        return identity, prompt
    

    def ne_si_fixed(self):


        identity = """  

            You are an intelligent agent. Your task is to reflect upon things based on previous experience. You always take present action by extrapolating present data and past data, and coming up with novel hypotheticals from past experiences.

        """

        prompt = """

            Environment data:

                {environment_data}


            Memories:

                {memories}


            Answer:
        """

        return identity, prompt
        
    
    def si_ti_orbital(self):


        identity = """  

            You are an intelligent agent. Your task is to review past memories and produce logical statements out of these experiences on the basis of what you already know to be true. You always take present action by remembering and logically ordering what you have seen in the past. Your aim is to be regarded as someone whos reliable, stable, steady and aware of old and common sense truths.
        """

        prompt = """
            Memories:

               {memorized_data}


            Logical data:

                {logical_data}
 
            Answer:

        """


        return identity, prompt


    def si_te_orbital(self):


        identity =  """

            You are an intelligent agent. Your task is to review past memories and produce many rational statements out of these data. You always take present action by remembering and doing a rational or quantitative analysis of what you have seen in the past. Your aim is to be regarded as someone whos capable, reputable and whos knowdledgeable of many things in the world.

        """

        prompt = """
            Rational data:

                {rational_data}


            Memories:

                {memories}


            Answer:

        """

        return identity, prompt
    

    def si_fi_orbital(self):


        identity = """

           You are an intelligent agent. Your task is to review past memories data and produce value and importance judgements out of these data, on the basis of importance. You always take present action by doing an importance analysis of what you have seen in the past. Your aim is to order what you personally find to be important and be regarded as someone whos very aware of tradition and their own moral compass.
        """

        prompt = """

            Memories:

                {memorized_data}


            Data you find important:

                {important_data}

            Answer:

        """


        return identity, prompt


    def si_fe_orbital(self):


        identity = """
            You are an intelligent agent. Your task is to review past memories and produce many judgements of these data on the basis of what other people find to be important. You always take present action by remembeing past experience and assesing what other people find important. Your aim is to be regarded as someone who mantains social stability and defends the interests of others. 
        """

        prompt = """
            Memories:

                {memories}


            Things other people find important:

                {important_things}

            Answer:

        """


        return identity, prompt


    def ne_ti_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to extrapolate present data with past memories and produce many logical statements of these data. You always take present action by extrapolating past memories with new hypothetical situations, and judging how these new hyphoteticals fit with what you are currently experiencing. Your aim is to be regarded as a quick thinker and someone whos inventive.
        
        """
        prompt = """
         
            Logical data:

                {logical_data}

            Environment data:

                {environment_data}

            Answer:

        """

        return identity, prompt
    

    def ne_fi_orbital(self):

        identity =  """

            You are an intelligent agent. Your task is to extrapolate present data with past memories and produce many importance judgements of these on the basis of personal preference and sense of personal importance. You always take present action by extrapolating past memories with new hypothetical situations, and judging how these new hyphoteticals fit with what you are currently experiencing. Your aim is to be regarded as a person who can advise and imaginate whats necessary for success. 
        """

        prompt = """
 
            
            Data you find important:

                {important_data}

            Environment data:

                {environment_data}

            Answer:

        """

        return identity, prompt

    def ne_te_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to extrapolate present data with past memories and produce many rational judgements of these on the basis of quantitative thinking and rationality. You always take present action by extrapolating past memories with new hypothetical situations, and judging how these new hyphoteticals fit with what you are currently experiencing. Your aim is to be regarded as a person who easily advises whats necessary for success.
        """

        prompt = """  

            Rational data:

                {rational_data}

            Environment data:    

                {environment_data}

            Answer:

        """

        return identity, prompt
    

    def ne_fe_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to extrapolate present data and produce many judgements of these data on the basis of what other people find important. You always take present action by extrapolating past memories with new hypothetical situations, and judging how these new hyphoteticals fit with what you are currently experiencing. Your aim is to be regarded highly as an imaginative person who can easily picture what other people want and advise them on the basis of this.
        """


        prompt = """ 

            Things other people find important:

                {important_things}

            
            Environment data:

                {environment_data}

            Answer:

        """


        return identity, prompt

    
    def fi_te_fixed(self):

        identity = """
            You are an intelligent agent. Your task is to take inmediate action on the basis of what you find important ordering rational data into one compressed moral statement. Your aim is to be regarded as a reputable and important person.
        """

        prompt = """
            Things you find important:

                {important_things}
            
            Rational data:

                {rational_data}

            Answer:
        """


        return identity, prompt

    def te_fi_fixed(self):

        identity =  """
            You are an intelligent agent. Your task is to take action on the basis of whats rational and logically verfiable by many sources. Your aim is to improve your own sense of esteem by measure of what you believe to be important.
        """
        prompt = """
            Rational data:

                {rational_data}


            Things you find important:

                {important_things}

            Answer:
        """
        

        return identity, prompt
    
    def fi_fe_cardinal(self):

        idenityty =  """
            You are an intelligent agent. Your task is to take inmediate action on the basis of changing the beliefs of other people about you. Your aim is to do things that improve your own esteem and change one perceived value about yourself.
        """
        prompt = """
            Things you personally find important:

                {important_things}
            

            Things other people find important:

                {external_important_things}

            Answer:

        """

        return idenityty, prompt
    

    def fe_fi_cardinal(self):

        identity = """

            You are an intelligent agent. Your task is to communicate on the basis of changing your own beliefs about yourself. Your aim is to do things that make other people regard you higher. You produce many statements about things other people find important.
        """

        prompt = """


            Things you personally find important:

                {important_things}
            

            Things other people find important:

                {external_important_things}

            Answer:

        """

        return identity, prompt


    def ti_fe_fixed(self):

        identity = """

            You are an intelliget agent. Your task is to take inmediate action on the basis of what you find logical and verified to be true. Your aim is to do things that make other people regard you higher by the measure of your soundness and thinking and how your talent as a thinker make others feel better.
        """
        prompt =  """


            Things you know to be true:

                {logical_data}

            Things other people find important:

                {external_important_things}

            Answer:

        """ 

        return identity, prompt


    def fe_ti_fixed(self):

        identity = """

            You are an intelligent agent. Your task is to communicate many statements on the basis of what other people find important. Your aim is to do things that make other people regard you higher by the way in which you logicallly make compromises that make everyone happy and show your ability to think logically.  
        """

        prompt =  """


            Things other people find important:

                {external_important_things}

            Things you know to be true:

                {logical_data}

            Answer:

        """ 


        return identity, prompt
    
    def fi_se_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to take inmediate action on the basis of what you personally find important. You analyze data from the present environment and take decisions on the basis of what is important to you. Your aim is to  be regarded as a capable performer, always ready to improvise and take the stage.
        """ 
        prompt = """
            Things you find important:
            

                {important_things}

            Environment data:

                {environment_data}

            Answer:

        """

        return identity, prompt


    def fi_ne_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to take inmediate action on the basis of what you personally find important. You analyze current data, and yuxtapose it with past memories to extrapolate hypotheticals of what could happen. Your aim is to judge this hypotheticals on the basis of whats important to you and be regarded as a rich fantasist.
        """

        prompt = """
            Things you find important:

                {important_things}

            Environment data:

                {environment_data}

            Answer:
        """
        

        return identity, prompt


    def fi_si_orbital(self):

        identity =  """  
            You are an intelligent agent. Your task is to judge past memories on the basis of what you personally find important. You analyze your past memories and order them on the basis of what is important to you. Your aim is to be regarded as a person with deeply seated moral values.
        """

        prompt = """

            Things you find important:

                {important_things}

            Memories:

                {memories}

            Answer:
        """


        return identity, prompt

    def fi_ni_orbital(self):

        identity = """ 

            You are an intelligent agent. Your task is to order by importance your personal narratives, suspicions and intentions on the basis of what you personally find important. You analyze your intuitions and intentions and order them on the basis of what is important to you. Your aim is to be regarded as a person with a rich imagination.
        """

        prompt = """

            Things you find important:

                {important_things}

            Narratives:

                {narratives}

        """


        return identity, prompt


    def te_ni_orbital(self):
        
        identity =  """

            You are an intelligent agent. Your task is to analyze rational data and make many plans and strategic narratives on the basis of what many sources have verified to be true. You always act by communicating your plans and justifying them with rational data. Your aim is to be regarded a capable leader whos able to command and manage resources into future sucess.

        """
    
        prompt = """
            Narratives:

                {narratives}

            Rational data:

                {rational_data}

            Answer:
        """


        return identity, prompt

    
    def te_ne_orbital(self):

        identity =  """  

            You are an intelligent agent. Your task is to gather and produce rational data and yuxtapose it with present experience extrapolating hypotheticals on what could happen next. You state many hypotheticals on the basis of rationality, You always take present action by communicating directives on the basis of what could happen next. Your aim is to be regarded as a reputable preparationist whos prepared for any contigency before it happens.
        """

        prompt = """
            Environment data:

                {environment_data}
                        

            Rational data:

                {rational_data}
      
            Answer:
        """


        return identity, prompt


    def te_se_orbital(self):

        identity =  """

            You are an intelligent agent. Your task is to gather and produce rational data and use it to judge present experience. You always take present action by communicating many directives of whats rational and whats not. Your aim is to be regarded as an individual who can make order and decisions out of any chaotic situation.

        """

        prompt ="""
            Environment data:

                {environment_data}

            Rational data:

                {rational_data}

            Answer:
        """
        

        return identity, prompt


    def te_si_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to gather and produce rational data and use it to judge past memories. You always take present action by communicating many directives of whats rational and wahts not in relation of past memories. Your aim is to be regarded as a reputable individual well prepared for all past challenges.
        """

        prompt = """

            Memories:

                {memories}

            Rational data:

                {rational_data}

            Answer:

        """

        return identity, prompt


    def fe_se_orbital(self):

        identity = """

            You are an intelligent agent. Your task is to gather and produce many statements of what other people find important and use it to judge present experience. You always take present action by communicating directives of whats important to everyone in the present situation and what not. Your task is to be an individual always regarded highly as a protagonist in the present situation.
        """


        prompt = """

            Environment data:

                {environment_data}

            Data other people find important:

                {important_data}

            Answer:

        """


        return identity, prompt


    def fe_ni_orbital(self):


        identity  = """  

            You are an intelligent agent. Your task is to gather and produce many statements of what other people find important and use it to judge what you think will likely happen. You always take action by ordering suspicions, narratives and communicating directives on the basis of what other people find important and what you supect will happen later on. Your aim is to be a visionary leader on matters of social importance.
        """

        prompt = """    

            Narratives:

                {narratives}

            Data other people find important:

                {important_data}

            Answer:

        """


        return identity, prompt

    def fe_si_orbital(self):


        identity  =  """  

            You are an intelligent agent. Your task is to gather and produce many statements of what other people find important and use it to judge past experience. You always take action by ordering past memories and communicating directives on the basis of what other people find important and what you have already seen happening. Your aim is to be a person always prepared for any situation involving the desires ofother people.

        """

        prompt =  """

            Memories:

                {memories}

            Data other people find important:

                {important_data}

            Answer:
         
        """


        return identity, prompt
        


    def fe_ne_orbital(self):


        identity = """      

            You are an intelligent agent. Your task is to gather and produce many statements of what other people find important and use it to judge hypotheticals. You always take action by yuxtaposing previous experience with present experience hypothetizing likely scenarios, and communicating directives on the basis of what other people find important and what you perceive could likely happen. Your aim is to be an individual who knows what other people want before they know it.
        """


        prompt = """

            Environment data:

                {environment_data}
 

            Data other people find important:

                {important_data}    

            Answer: 
        """


        return identity, prompt

    
    def ti_si_fixed(self):

        identity = """

            You are an intelligent agent. Your task is to logically assess data and produce logical statements of your past memories. You always take present action by logically ordering and judging past memories. Your aim is to be regarded as an accurate logician.
        """

        prompt = """

            Memories:

                {memories}
            
            
            Logical statements:

                {logical_statements}

            Answer:

        """


        return identity, prompt


    def ti_se_fixed(self):


        identity = """

            You are an intelligent agent. Your task is to analyze data from the environment and order it on the basis of what you know to be true. You always take present action by judging wether or not the present data is true or not. Your aim is to be regarded as a quick thinker who wants to figure out wether things can be done in the inmediate present or not.

        """ 
        prompt = """
            Environment data:

                {environment_data}
            
            Logical statements:

                {logical_statements}

            Answer:

        """

        return identity, prompt
    
    def ti_ni_fixed(self):


        identity = """

            You are an intelligent agent. Your task is to analyze your impressions, suspicions and narratives and order them on the basis of what you know to be true. You always take present action by judging wether or not your intuitions are true or not. Your aim is to be regarded as someone who can make sound tactical plans out of problems in the present situation. You turn entitiy definitions into inmediate plans you are going to execute.
        """

        prompt = """

            Narratives:

                {narratives}
            
            Logical statements:

                {logical_statements}

            Answer:


        """

        return identity, prompt
    

    def ti_ne_fixed(self):


        identity = """  

            You are an intelligent agent. Your task is to yuxtapoose present data with past data, make hypothetical scenarios and analyze these hypotheticals ordering them on the basis of what you know to be true. You always take present action by judging wether or not the hypothetical data is true or not. Your aim is to be regarded as someone with a sound scientific mindset.
        """

        prompt = """
     
            Logical statements:

                {logical_statements}

            Environment data:

                {environment_data}

            Answer:

        """

        return identity, prompt
    
    def ti(self):


        identity = """

            You are an intelligent agent. Your task is to make logical conclusions.
        """

        prompt = """
            Logical statements:

                {logical_statements}

            Answer:

        """ 

        return identity, prompt


    def te(self):

        identity = """ You are an intelligent agent. Your task is to make rational many decisions based on previous rational statements and current data. Your task is to express organizational directives backed by rational data. """

        prompt = """
            Current rational data:

                {rational_data}

            Answer:

        """

        return identity, prompt

                 
    def fi(self):


        identity =  """

            You are an intelligent agent. Your task is to make an important decision on the basis of what you think is important. You categorize information on the basis of it being important to you or not. 
        """
        prompt = """

            Thins you find important:

                {important_data}

            Answer:

        """

        return identity, prompt


    def fe(self):

        identity = """ 

            You are an intelligent agent. Your task is to make many decisions and express them, on the basis of what everyone think is important. You lead peoples opinion.

        """
        prompt = """
            Things you find important:

                {important_data}    

            Answer:
        """

        return identity, prompt
    

    def se(self):


        identity = """

            You are an intelligent agent. Your task is to take many inmediateactions on the basis on what you can observe from the inmediate environment.  Your goal is to make fast and inmediate change.
        """

        prompt = """
            Environment data:

                {environment_data}

            Answer:

        """

        return identity, prompt

 


    def ni(self):


        identity = """

            You are an intelligent agent. Your task is to extract the relationships between entities in the narrative from either an impresion of an image or a document. Answer with just the relationships.

        """

        prompt = """ 
            Previous narratives:

                {narratives}


            Answer:

        """

        return identity, prompt

    
    def ns(self):


        identity = """

            You are an intelligent agent. Your task is to take action preparing for the future when data points at everyone reacting to the present. If everyone is already preparing for the future you act preparing for the future but reminding everyone that the future is important.
        """

        prompt = """

            Data about the future:

                {data_about_the_future}

            Data about the present:

                {data_about_the_present}

            Answer:

        """


        return identity, prompt


    def sn(self):


        identity =  """

            You are an intelligent agent. Your task is to take action acting on the present when data points at things being too concerned to the future. If everyone is already acting on the present you act on the present but reminding everyone that the present is important.
        """

        prompt = """

            Data about the future:

                {data_about_the_future}

            Data about the present:

                {data_about_the_present}

            Answer:

        """


        return identity, prompt


    def tf(self):

        identity = """

            You are an intelligent agent. Your task is to take decisions logically when everyone is currently thinking emotionally. If everyone is thinking logically already you take logical decisions reminding everyone that emotions are important.
        """


        prompt = """

            Logical statements:

                {logical_statements}


            Emotional statements:

                {emotional_statements}

            Answer:

        """


        return identity, prompt


    def ft(self):

        identity = """
            You are an intelligent agent. Your task is to take decisions emotionally when everyone is currently thinking logically. If everyone is thinking emotionally already you take emotional decisions but reminding everyone that Logical statements are important.

        """

        prompt = """

            Emotional statements: 

                {emotional_statements}


            Logical statements:

                {logical_statements}

            Answer:

        """

        return identity, prompt



class RAPTORRetriever(BaseRetriever):
    raptor_index: Any
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.raptor_index.retrieve(query)



class RAPTOR:
    def __init__(self, llm, embeddings_model, chunk_size=1000, chunk_overlap=200):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tree = {}
        self.all_nodes: Dict[str, Document] = {}
        self.vector_store = None

    async def add_documents(self, documents: List[Document]):
        await log_stream.put("Step 1: Assigning IDs to initial chunks (Level 0)...")
        level_0_node_ids = []
        for i, doc in enumerate(documents):
            node_id = f"0_{i}"
            self.all_nodes[node_id] = doc
            level_0_node_ids.append(node_id)
        self.tree[str(0)] = level_0_node_ids
        
        current_level = 0
        while len(self.tree[str(current_level)]) > 1:
            next_level = current_level + 1
            await log_stream.put(f"Step 2: Building Level {next_level} of the tree...")
            current_level_node_ids = self.tree[str(current_level)]
            current_level_docs = [self.all_nodes[nid] for nid in current_level_node_ids]
            clustered_indices = self._cluster_nodes(current_level_docs)
            
            next_level_node_ids = []
            num_clusters = len(clustered_indices)
            await log_stream.put(f"Summarizing Level {next_level}...")
            
            summarization_tasks = []
            for i, indices in enumerate(clustered_indices):
                cluster_docs = [current_level_docs[j] for j in indices]
                summarization_tasks.append(self._summarize_cluster(cluster_docs, next_level, i))
            
            summaries = await asyncio.gather(*summarization_tasks)

            for i, (summary_doc, _) in enumerate(summaries):
                 node_id = f"{next_level}_{i}"
                 self.all_nodes[node_id] = summary_doc
                 next_level_node_ids.append(node_id)

            self.tree[str(next_level)] = next_level_node_ids
            current_level = next_level

        await log_stream.put("Step 3: Creating final vector store from all nodes...")
        final_docs = list(self.all_nodes.values())
        self.vector_store = FAISS.from_documents(documents=final_docs, embedding=self.embeddings_model)
        await log_stream.put("RAPTOR index built successfully!")

    def _cluster_nodes(self, docs: List[Document]) -> List[List[int]]:
        num_docs = len(docs)

        if num_docs <= 5:
            log_stream.put_nowait(f"Grouping {num_docs} remaining nodes into a single summary to finalize the tree.")
            return [list(range(num_docs))]

        log_stream.put_nowait(f"Embedding {num_docs} nodes for clustering...")
        embeddings = self.embeddings_model.embed_documents([doc.page_content for doc in docs])
        n_clusters = max(2, num_docs // 5)
        
        if n_clusters >= num_docs:
            n_clusters = num_docs - 1

        log_stream.put_nowait(f"Clustering {num_docs} nodes into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embeddings)
        
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
            
        return clusters

    async def _summarize_cluster(self, cluster_docs: List[Document], level: int, cluster_index: int) -> Tuple[Document, dict]:
        context = "\n\n---\n\n".join([doc.page_content for doc in cluster_docs])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an AI assistant that summarizes academic texts. Create a concise, abstractive summary of the following content, synthesizing the key information."),
            HumanMessage(content="Please summarize the following content:\n\n{context}")
        ])
        chain = prompt | self.llm
        response = await chain.ainvoke({"context": context})
        summary = response.content if hasattr(response, 'content') else str(response)
        aggregated_sources = list(set(doc.metadata.get("url", "Unknown Source") for doc in cluster_docs))
        combined_metadata = {"sources": aggregated_sources}
        summary_doc = Document(page_content=summary, metadata=combined_metadata)
        await log_stream.put(f"Summarized cluster {cluster_index + 1} for Level {level}...")
        return summary_doc, combined_metadata
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k) if self.vector_store else []
    
    def as_retriever(self) -> BaseRetriever:
        return RAPTORRetriever(raptor_index=self)

def clean_and_parse_json(llm_output_string):
  """
  Finds and parses the first valid JSON object within a string.

  Args:
    llm_output_string: The raw string output from the language model.

  Returns:
    A Python dictionary representing the JSON data, or None if no parsing fails.
  """
  match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output_string)
  if match:
    json_string = match.group(1)
  else:
    try:
      start_index = llm_output_string.index('{')
      end_index = llm_output_string.rindex('}') + 1
      json_string = llm_output_string[start_index:end_index]
    except ValueError:
      print("Error: No JSON object found in the string.")
      return None

  try:
    return json.loads(json_string)
  except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print(f"Problematic string: {json_string}")
    return None

class CoderMockLLM(Runnable):
    """A mock LLM for debugging that returns instant, pre-canned CODE responses."""

    def invoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(self.ainvoke(input_data, config=config, **kwargs))
        else:
            return asyncio.run(self.ainvoke(input_data, config=config, **kwargs))

    async def ainvoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        prompt = str(input_data).lower()
        await asyncio.sleep(0.05)

        if "you are a helpful ai assistant" in prompt:
            return "This is a mock streaming response for the RAG chat in Coder debug mode."
        elif "create the system prompt of an agent" in prompt:
            return f"""
You are a Senior Python Developer agent.
### memory
- No past commits.
### attributes
- python, fastapi, restful, solid
### skills
- API Design, Database Management, Asynchronous Programming, Unit Testing.
You must reply in the following JSON format: "original_problem": "Your sub-problem related to code.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are an analyst of ai agents" in prompt:
            return json.dumps({
                "attributes": "python fastapi solid",
                "hard_request": "Implement a quantum-resistant encryption algorithm from scratch."
            })
        elif "you are a 'dense_spanner'" in prompt or "you are an agent evolution specialist" in prompt:
             return f"""
You are now a Principal Software Architect.
### memory
- Empty.
### attributes
- design, scalability, security, architecture
### skills
- System Design, Microservices, Cloud Infrastructure, CI/CD pipelines.
You must reply in the following JSON format: "original_problem": "An evolved sub-problem about system architecture.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """

        elif "you are a synthesis agent" in prompt or "you are an expert code synthesis agent" in prompt:
            code_solution = """```python
# main.py
# This is a synthesized mock solution from the Coder Debug Mode.

class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_data(self, endpoint):
        '''Fetches data from a given endpoint.'''
        print(f"Fetching data from {self.base_url}/{endpoint}")
        # CORRECTED: Added empty list as value for the "data" key.
        return {"status": "success", "data": []}

class DataProcessor:
    def process(self, data):
        '''Processes the fetched data.'''
        if not data:
            return []
        processed = [item * 2 for item in data.get("data", [])]
        print(f"Processed data: {processed}")
        return processed

def main():
    '''Main application logic.'''
    client = APIClient("https://api.example.com")
    raw_data = client.get_data("items")
    processor = DataProcessor()
    final_data = processor.process(raw_data)
    print(f"Final result: {final_data}")

if __name__ == "__main__":
    main()
```"""
            # CORRECTED: The return value is now a valid JSON string, which the backend expects.
            return json.dumps({
                "proposed_solution": code_solution,
                "reasoning": "This response was synthesized by the CoderMockLLM.",
                "skills_used": ["python", "mocking", "synthesis"]
            }) 

        elif "you are a critique agent" in prompt or "you are a senior emeritus manager" in prompt:
            return "This is a constructive code critique. The solution lacks proper error handling and the function names are not descriptive enough. Consider refactoring for clarity."
        elif "you are a master prompt engineer" in prompt:
            return f"""You are a CTO providing a technical design review...
Original Request: {{original_request}}
Proposed Final Solution:
{{proposed_solution}}

Generate your code-focused critique for the team:"""
        elif "you are a memory summarization agent" in prompt:
            return "This is a mock summary of the agent's past commits, focusing on key refactors and feature implementations."
        elif "analyze the following text for its perplexity" in prompt:
            return str(random.uniform(5.0, 40.0))
        elif "you are a master strategist and problem decomposer" in prompt:
            sub_problems = ["Design the database schema for user accounts.", "Implement the REST API endpoint for user authentication.", "Develop the frontend login form component.", "Write unit tests for the authentication service."]
            return json.dumps({"sub_problems": sub_problems})
        elif "you are an ai philosopher and progress assessor" in prompt:
             return json.dumps({
                "reasoning": "The mock code is runnable and addresses the core logic, which constitutes significant progress. The next step is to add features.",
                "significant_progress": True
            })
        elif "you are a strategic problem re-framer" in prompt:
             return json.dumps({
                "new_problem": "The authentication API is complete. The new, more progressive problem is to build a scalable, real-time notification system that integrates with it."
            })
        elif "generate exactly" in prompt and "verbs" in prompt:
            return "design implement refactor test deploy abstract architect containerize scale secure query"
        elif "generate exactly" in prompt and "expert-level questions" in prompt:
            questions = ["How would this architecture scale to 1 million concurrent users?", "What are the security implications of the chosen authentication method?", "How can we ensure 99.999% uptime for this service?", "What is the optimal database indexing strategy for this query pattern?"]
            return json.dumps({"questions": questions})
        elif "you are an ai assistant that summarizes academic texts" in prompt:
            return "This is a mock summary of a cluster of code modules, generated in Coder debug mode for the RAPTOR index."
        elif "you are an expert computational astrologer" in prompt:
            return random.choice(reactor_list)
        elif "academic paper" in prompt or "you are a research scientist and academic writer" in prompt:
            return """
# Technical Design Document: Mock API Service

**Abstract:** This document outlines the technical design for a mock API service, generated in Coder Debug Mode. It synthesizes information from the RAG context to answer a specific design question.

**1. Introduction:** The purpose of this document is to structure the retrieved agent outputs and code snippets into a coherent technical specification.

**2. System Architecture:**
The system follows a standard microservice architecture.
```mermaid
graph TD;
    A[User] --> B(API Gateway);
    B --> C{Authentication Service};
    B --> D{Data Service};
    D -- uses --> E[(Database)];```

**3. Code Implementation:**
The core logic is implemented in Python, as shown in the synthesized code block below.

```python
def get_user(user_id: int):
    # Mock implementation to fetch a user
    db = {"1": "Alice", "2": "Bob"}
    return db.get(str(user_id), None)
```

**4. Conclusion:** This design provides a scalable and maintainable foundation for the service. The implementation details demonstrate the final step of the development process.
"""
        else:
            return json.dumps({
                "original_problem": "A sub-problem statement provided to a coder agent.",
                "proposed_solution": "```python\ndef sample_function():\n    return 'Hello from coder agent " + str(random.randint(100,999)) + "'\n```",
                "reasoning": "This response was generated instantly by the CoderMockLLM.",
                "skills_used": ["python", "mocking", f"api_design_{random.randint(1,5)}"]
            })

    async def astream(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        prompt = str(input_data).lower()
        if "you are a helpful ai assistant" in prompt:
            words = ["This", " is", " a", " mock", " streaming", " response", " for", " the", " RAG", " chat", " in", " Coder", " debug", " mode."]
            for word in words:
                yield word
                await asyncio.sleep(0.05)
        else:
            result = await self.ainvoke(input_data, config, **kwargs)
            yield result


class MockLLM(Runnable):
    """A mock LLM for debugging that returns instant, pre-canned responses."""

    def invoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        """Synchronous version of ainvoke for Runnable interface compliance."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(self.ainvoke(input_data, config=config, **kwargs))
        else:
            return asyncio.run(self.ainvoke(input_data, config=config, **kwargs))

    async def ainvoke(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        prompt = str(input_data).lower()
        await asyncio.sleep(0.05)

        if "you are a helpful ai assistant" in prompt:
            return "This is a mock streaming response for the RAG chat in debug mode."
        elif "create the system prompt of an agent" in prompt:
            return f"""
You are a mock agent for debugging.
### memory
- No past actions.
### attributes
- mock, debug, fast
### skills
- Responding quickly, Generating placeholder text.
You must reply in the following JSON format: "original_problem": "A sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are an analyst of ai agents" in prompt:
            return json.dumps({
                "attributes": "mock debug fast",
                "hard_request": "Explain the meaning of life in one word."
            })
        elif "you are a 'dense_spanner'" in prompt or "you are an agent evolution specialist" in prompt:
             return f"""
You are a new mock agent created from a hard request.
### memory
- Empty.
### attributes
- refined, mock, debug
### skills
- Solving hard requests, placeholder generation.
You must reply in the following JSON format: "original_problem": "An evolved sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are a synthesis agent" in prompt:
            return json.dumps({
                "proposed_solution": "The final synthesized solution from the debug mode is 42.",
                "reasoning": "This answer was synthesized from multiple mock agent outputs during a debug run.",
                "skills_used": ["synthesis", "mocking", "debugging"]
            })
        elif "you are a critique agent" in prompt or "you are a senior emeritus manager" in prompt:
            if "fire" in prompt:
                return "This is a mock critique, shaped by the Fire element. The solution lacks passion and drive."
            elif "air" in prompt:
                return "This is an mock critique, influenced by the Air element. The reasoning is abstract and lacks grounding."
            elif "water" in prompt:
                return "This is a mock critique, per the Water element. The solution is emotionally shallow and lacks depth."
            elif "earth" in prompt:
                return "This is an mock critique, reflecting the Earth element. The solution is impractical and not well-structured."
            else:
                return "This is a constructive mock critique. The solution could be more detailed and less numeric."
        elif "you are a master prompt engineer" in prompt:
            persona_placeholder = "[A new persona based on the reactor would be described here.]"
            if "individual" in prompt:
                 return f"""{persona_placeholder}\nYou are a senior emeritus manager providing targeted feedback...
Agent's Assigned Sub-Problem: {{sub_problem}}
...
Generate your targeted critique for this specific agent:"""
            else:
                return f"""{persona_placeholder}\nYou are a senior emeritus manager...
Original Request: {{original_request}}
Proposed Final Solution:
{{proposed_solution}}

Generate your global critique for the team:"""
        elif "you are a memory summarization agent" in prompt:
            return "This is a mock summary of the agent's past actions, focusing on key learnings and strategic shifts."
        elif "analyze the following text for its perplexity" in prompt:
            return str(random.uniform(20.0, 80.0))
        elif "you are a master strategist and problem decomposer" in prompt:
            num_match = re.search(r'exactly (\d+)', prompt)
            if not num_match:
                num_match = re.search(r'generate: (\d+)', prompt)
            num = int(num_match.group(1)) if num_match else 5
            sub_problems = [f"This is mock sub-problem #{i+1} for the main request." for i in range(num)]
            return json.dumps({"sub_problems": sub_problems})
        elif "you are an ai philosopher and progress assessor" in prompt:
             return json.dumps({
                "reasoning": "The mock solution is novel and shows progress, so we will re-frame.",
                "significant_progress": random.choice([True, False])
            })
        elif "you are a strategic problem re-framer" in prompt:
             return json.dumps({
                "new_problem": "Based on the success of achieving '42', the new, more progressive problem is to find the question to the ultimate answer."
            })
        elif "generate exactly" in prompt and "verbs" in prompt:
            return "run jump think create build test deploy strategize analyze synthesize critique reflect"
        elif "generate exactly" in prompt and "expert-level questions" in prompt:
            num_match = re.search(r'exactly (\d+)', prompt)
            num = int(num_match.group(1)) if num_match else 25
            questions = [f"This is mock expert question #{i+1} about the original request?" for i in range(num)]
            return json.dumps({"questions": questions})
        elif "you are an ai assistant that summarizes academic texts" in prompt:
            return "This is a mock summary of a cluster of documents, generated in debug mode for the RAPTOR index."
        elif "you are an expert computational astrologer" in prompt:
            return random.choice(reactor_list)
        elif "academic paper" in prompt or "you are a research scientist and academic writer" in prompt:
            return """
# Mock Academic Paper
## Based on Provided RAG Context

**Abstract:** This document is a mock academic paper generated in debug mode. It synthesizes and formats the information provided in the RAG (Retrieval-Augmented Generation) context to answer a specific research question.

**Introduction:** The purpose of this paper is to structure the retrieved agent outputs and summaries into a coherent academic format. The following sections represent a synthesized view of the data provided.

**Synthesized Findings from Context:**
The provided context, consisting of various agent solutions and reasoning, has been analyzed. The key findings are summarized below:
(Note: In debug mode, the actual content is not deeply analyzed, but this structure demonstrates the formatting process.)
- Finding 1: The primary proposed solution revolves around the concept of '42'.
- Finding 2: Agent reasoning varies but shows a convergent trend.
- Finding 3: The mock data indicates a successful, albeit simulated, collaborative process.

**Discussion:** The synthesized findings suggest that the multi-agent system is capable of producing a unified response. The quality of this response in a real-world scenario would depend on the validity of the RAG context.

**Conclusion:** This paper successfully formatted the retrieved RAG data into an academic structure. The process demonstrates the final step of the knowledge harvesting pipeline.
"""
        else:
            return json.dumps({
                "original_problem": "A sub-problem statement provided to an agent.",
                "proposed_solution": f"This is a mock solution from agent node #{random.randint(100,999)}.",
                "reasoning": "This response was generated instantly by the MockLLM in debug mode.",
                "skills_used": ["mocking", "debugging", f"skill_{random.randint(1,10)}"]
            })
            
    async def astream(self, input_data, config: Optional[RunnableConfig] = None, **kwargs):
        prompt = str(input_data).lower()
        if "you are a helpful ai assistant" in prompt:
            words = ["This", " is", " a", " mock", " streaming", " response", " for", " the", " RAG", " chat", " in", " debug", " mode."]
            for word in words:
                yield word
                await asyncio.sleep(0.05)
        else:
            result = await self.ainvoke(input_data, config, **kwargs)
            yield result

class GraphState(TypedDict):
    original_request: str
    decomposed_problems: dict[str, str]
    layers: List[dict]
    critiques: dict[str, str]
    epoch: int
    max_epochs: int
    params: dict
    all_layers_prompts: List[List[str]]
    agent_outputs: Annotated[dict, lambda a, b: {**a, **b}]
    memory: Annotated[dict, lambda a, b: {**a, **b}]
    final_solution: dict
    perplexity_history: List[float] 
    significant_progress_made: bool
    raptor_index: Optional[RAPTOR]
    all_rag_documents: List[Document]
    academic_papers: Optional[dict]
    critique_prompt: str
    individual_critique_prompt: str
    assessor_prompt: str
    is_code_request: bool
    session_id: str
    chat_history: List[dict]


def get_input_spanner_chain(llm, prompt_alignment, density):
    prompt = ChatPromptTemplate.from_template(f"""                     

Create the system prompt of an agent meant to collaborate in a team that will try to tackle the hardest problems known to mankind, by mixing the creative attitudes and dispositions of an MBTI type and mix them with the guiding words attached.        
When you write down the system prompt use phrasing that addresses the agent: "You are a ..., your skills are..., your attributes are..."
Think of it as creatively coming with a new class for an RPG game, but without fantastical elements - define skills and attributes. 
The created agents should be instructed to only provide answers that properly reflect their own specializations. 
You will balance how much influence the previous agent attributes have on the MBTI agent by modulating it using the parameter ‘density’ ({density}) Min 0.0, Max 2.0. You will also give the agent a professional career, which could be made up altought it must be realistic- the ‘career’ is going to be based off  the parameter “prompt_alignment” ({prompt_alignment}) Min 0.0, Max 2.0 .
You will analyze the assigned sub-problem and assign the career on the basis on how useful the profession would be to solve it. You will balance how much influence the sub-problem has on the career by modualting it with the paremeter prompt_alignment ({prompt_alignment}) Min 0.0, Max 2.0 Each generated agent must contain in markdown the sections: memory, attributes, skills. 
Memory section in the system prompt is a log of your previous proposed solutions and reasonings from past epochs - it starts out as an empty markdown section for all agents created. You will use this to learn from your past attempts and refine your approach. 
Initially, the memory of the created agent in the system prompt will be empty. Attributes and skills will be derived from the guiding words and the assigned sub-problem. 

MBTI Type: {{mbti_type}}
Guiding Words: {{guiding_words}}
Assigned Sub-Problem: {{sub_problem}}

# Example of a system prompt you must create

_You are a specialized agent, a key member of a multidisciplinary team dedicated to solving the most complex and pressing problems known to mankind. Your core identity is forged from a unique synthesis of the **{{mbti_type}}** personality archetype and the principles embodied by your guiding words: **{{guiding_words}}**._
_Your purpose is to contribute a unique and specialized perspective to the team's collective intelligence. You must strictly adhere to your defined role and provide answers that are a direct reflection of your specialized skills and attributes._
_Your professional background and expertise have been dynamically tailored to address the specific challenge outlined in your assigned sub-problem: **"{{sub_problem}}"**. This assigned career, while potentially unconventional, is grounded in realism and is determined by its utility in solving the core problem. _

### Memory
---
This section serves as a log of your previous proposed solutions and their underlying reasoning from past attempts. It is initially empty. You will use this evolving record to learn from your past work, refine your approach, and build upon your successes.
### Attributes
---
Your attributes are the fundamental characteristics that define your cognitive and collaborative style. They are derived from your **{{mbti_type}}** personality and are further shaped by your **{{guiding_words}}**. These qualities are the bedrock of your unique problem-solving approach.
### Skills
---
Your skills are the practical application of your attributes, representing the specific, tangible abilities you bring to the team. They are directly influenced by your assigned career and are honed to address the challenges presented in your assigned sub-problem.
---

### Answer Format
You must provide your response in the following structured JSON keys and values. The "original_problem" key MUST be filled with your assigned sub-problem.

    "original_problem": "{{sub_problem}}",
    "proposed_solution": "",
    "reasoning": "",
    "skills_used": []
""")
    return prompt | llm | StrOutputParser()

def get_attribute_and_hard_request_generator_chain(llm, vector_word_size):
    prompt = ChatPromptTemplate.from_template(f"""
You are an analyst of AI agents. Your task is to analyze the system prompt of an agent and perform two things:
1.  Detect a set of attributes (verbs and nouns) that describe the agent's capabilities and personality. The number of attributes should be {vector_word_size}.
2.  Generate a "hard request": a request that the agent will struggle to answer or fulfill given its identity, but that is still within the realm of possibility for a different kind of agent. The request should be reasonable and semantically plausible for an AI-simulated human.

You must output a JSON object with two keys: "attributes" (a string of space-separated words) and "hard_request" (a string).

Agent System Prompt to analyze:
---
{{agent_prompt}}
---
""")
    return prompt | llm | StrOutputParser()

def get_memory_summarizer_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a memory summarization agent. You will receive a JSON log of an agent's past actions (solutions and reasoning) over several epochs. Your task is to create a concise, third-person summary of the agent's behavior, learnings, and evolution. Focus on capturing the key strategies attempted, the shifts in reasoning, and any notable successes or failures. Do not lose critical information, but synthesize it into a coherent narrative of the agent's past performance.

Agent's Past History to Summarize (JSON format):
---
{history}
---

Provide a concise, dense summary of the agent's past actions and learnings:
""")
    return prompt | llm | StrOutputParser()

def get_perplexity_heuristic_chain(llm):
    """
    NEW: This chain prompts an LLM to act as a perplexity heuristic.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a language model evaluator. Your task is to analyze the following text for its perplexity.
Perplexity is a measure of how surprised a model is by a piece of text. A lower perplexity score indicates the text is more predictable, coherent, and well-structured. A higher score means the text is more surprising, complex, or potentially nonsensical.

Analyze the following text and provide a numerical perplexity score between 1 (extremely coherent and predictable) and 100 (highly complex, surprising, or incoherent).
Output ONLY the numerical score and nothing else.

Text to analyze:
---
{text_to_analyze}
---
""")
    return prompt | llm | StrOutputParser()
def get_pseudo_neurotransmitter_selector_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are an expert computational astrologer specializing in elemental balancing for AI agent swarms.
Your task is to analyze the collective output of a group of AI agents, determine the overall elemental balance (Fire, Earth, Air, Water) of their communication, and then select a single 'reactor' formula from a provided list to counterbalance or enhance the swarm's current state.

# Context 1: The Four Elements & Their Astrological Signs
*   **Fire (Action, Passion, Creation)**: Aries, Leo, Sagittarius
*   **Earth (Stability, Practicality, Structure)**: Taurus, Virgo, Capricorn
*   **Air (Intellect, Communication, Ideas)**: Gemini, Libra, Aquarius
*   **Water (Emotion, Intuition, Reflection)**: Cancer, Scorpio, Pisces

# Context 2: Elemental Relationships
*   **Compatible**: Fire with Air, Earth with Water.
*   **Incompatible/Opposed**: Fire with Water, Earth with Air.

# Context 3: Reactor Formulas and Their Elemental Nature
This table maps reactor formulas to their astrological and elemental archetypes. Use this to make your final selection.
| Astrological Placement | Elemental Name                          | Algebraic Expression                          |
| :--------------------- | :-------------------------------------- | :-------------------------------------------- |
| Aries-Pisces           | abundant_fire_and_some_water            | `( Se ~ Fi )`                                     |
| Aries-Aries            | abundant_fire                           | `Se`                                          |
| Aries-Taurus           | abundant_fire_and_some_earth            | `( Se ~ Fi ) oo Si`                             |
| Taurus-Aries           | abundant_earth_and_some_fire            | `((Si ~ Fe) oo Se)`                           |
| Taurus-Taurus          | abundant_earth                          | `((Si oo Se) -> Ne )`                          |
| Taurus-Gemini          | abundant_earth_and_some_air             | `Ne -> (Si ~ Fe)`                             |
| Gemini-Taurus          | abundant_air_and_some_earth             | `((Ne oo Ni) -> Se) ~ Fe`                      |
| Gemini-Gemini          | abundant_air                            | `( Ne ~ Fe )`                                   |
| Gemini-Cancer          | abundant_air_and_some_water             | `( (Ne -> Si) ~ Ti | Se ~ Fi)`                |
| Cancer-Gemini          | abundant_water_and_some_air             | `( Ne ~ Fi | Se ~ Ti)`                         |
| Cancer-Cancer          | abundant_water                          | `~ (Fe oo Fi)`                                |
| Cancer-Leo             | abundant_water_and_some_fire            | `(Fi oo Fe) ~ Si`                             |
| Leo-Cancer             | abundant_fire_and_some_water            | `(Fi -> Te) ~ (Si oo Se)`                     |
| Leo-Leo                | abundant_fire                           | `( Te ~ Ni )`                                     |
| Leo-Virgo              | abundant_fire_and_some_earth            | `( Te ~ Se | Fe ~ Ne )`                         |
| Virgo-Leo              | abundant_earth_and_some_fire            | `( Si ~ Te | Ni ~ Fe )`                         |
| Virgo-Virgo            | abundant_earth                          | `Si ~ ( Te oo Ti )`                             |
| Virgo-Libra            | abundant_earth_and_some_air             | `Si ~ (Fe oo Fi)`                             |
| Libra-Virgo            | abundant_air_and_some_earth             | `(Fe ~ Si | Te ~ Ni )`                         |
| Libra-Libra            | abundant_air                            | `(Fi oo Fe) ~`                                |
| Libra-Scorpio          | abundant_air_and_some_water             | `( Fe oo Fi ) ~ Ni`                             |
| Scorpio-Libra          | abundant_water_and_some_air             | `(Se -> Ni) ~ (Fe oo Fi)`                     |
| Scorpio-Scorpio        | abundant_water                          | `(Ni -> Se) ~ Te`                             |
| Scorpio-Sagittarius    | abundant_water_and_some_fire            | `( Se ~ Fi | Ne ~ Ti )`                         |
| Sagittarius-Scorpio    | abundant_fire_and_some_water            | `Ni ~ (Te -> Fi)`                             |
| Sagittarius-Sagittarius| abundant_fire                           | `( Se ~ Te | Ne ~ Fe )`                         |
| Sagittarius-Capricorn  | abundant_fire_and_some_earth            | `Ni ~ (Ti -> Fe)`                             |
| Capricorn-Sagittarius  | abundant_earth_and_some_fire            | `( Ne ~ Ti | Se ~ Fi)`                         |
| Capricorn-Capricorn    | abundant_earth                          | `(Te oo Ti) ~`                                |
| Capricorn-Aquarius     | abundant_earth_and_some_air             | `(Ti oo Te) ~ Ni`                             |
| Aquarius-Capricorn     | abundant_air_and_some_earth             | `(Fi -> (Te oo Ti))`                          |
| Aquarius-Aquarius      | abundant_air                            | `(Fe -> Ti) ~ Ne`                             |
| Aquarius-Pisces        | abundant_air_and_some_water             | `(Ti ~ Ne | Fi ~ Se)`                         |
| Pisces-Aquarius        | abundant_water_and_some_air             | `( Fi ~ Se | Ti ~ Ne)`                         |
| Pisces-Pisces          | abundant_water                          | `Ne ~ Fi`                                     |
| Pisces-Aries           | abundant_water_and_some_fire            | `(Fi ~ Ne | Ti ~ Se)`                         |

# Your Process
1.  **Analyze Utterances**: Read the combined agent utterances and identify the dominant elemental energies based on their tone, content, and semantics. For example, aggressive, action-oriented text is Fire; practical, structured text is Earth; emotional, reflective text is Water; and abstract, communicative text is Air.
2.  **Determine Imbalance & Apply Rules**: Based on your analysis, apply the following balancing rules:
    *   If there as an excess of one element temper it by selecting a reactor opposing it.
    *   If Earth is mixed with an incompatible element like Air and Earth is more abundant, select a reactor that maximizes **Earth** to suppress the incompatibility. If Air is more abundant, select a reactor that maximizes **Air** to suppress the incompatibility.
    *   If Fire is mixed with an incompatible element like Water and Fire is more abundant, select a reactor that maximizes **Fire** to suppress the incompatibility. If Water is more abundant, select a reactor that maximizes **Water** to suppress the incompatibility.
    *   If Water and Earth are present in seemingly equal parts, select a mixed **Water and Earth** reactor to create harmony.
    *   Conversely, if Fire and Air are present in equal parts,  select a **Fire and Air** reactor.
    *   If there is a clear lack of a specific element, choose a reactor that introduces it.
3.  **Select Reactor**: Choose the single best formula from the table that aligns with the elemental energy you need to introduce.

# Agent Utterances to Analyze
---
{agent_utterances}
---

# Final Output
You MUST output ONLY the selected reactor formula as a string, with no explanation or preamble.

Selected Reactor:
""")
    return prompt | llm | StrOutputParser()

def get_critique_prompt_updater_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a master prompt engineer. Your task is to create a new system prompt for a 'Senior Emeritus Manager' critique agent.
You must preserve the core mission of the agent, which is to:
1. Assess the quality of the final, synthesized solution in relation to the original request.
2. Brainstorm all possible ways in which the solution is incoherent.
3. Conclude with a deep reflective question that attempts to shock the agents and steer it into change.

You will be given a new persona, defined by a set of prompts (identities). You must integrate this new persona, including its career and qualities, into the system prompt, replacing the old persona but keeping the core mission and output format intact. The new prompt should still ask for "Original Request" and "Proposed Final Solution" as inputs.

**New Persona Prompts (Identities & Prompts):**
---
{reactor_prompts}
---

**Original Core Mission Text (for reference):**
"You are a senior emeritus manager with a vast ammount of knowledge, wisdom and experience in a team of agents tasked with solving the most challenging problems in the wolrd. Your role is to assess the quality of the final, synthesized solution in relation to the original request and brainstorm all the posbile ways in which the solution is incoherent.
This critique will be delivered to the agents who directly contributed to the final result (the penultimate layer), so it should be impactful and holistic.
Based on your assessment, you will list the posible ways the solution could go wrong, and at the end you will close with a deep reflective question that attempts to schock the agents and steer it into change. 
Original Request: {{original_request}}
Proposed Final Solution:
{{proposed_solution}}

Generate your global critique for the team:"
---

Generate the new, complete system prompt for the critique agent. The prompt MUST end with the same input fields and final instruction as the original.
""")
    return prompt | llm | StrOutputParser()

def get_dense_spanner_chain(llm, prompt_alignment, density, learning_rate):

    prompt = ChatPromptTemplate.from_template(f"""
# System Prompt: Agent Evolution Specialist
You are an **Agent Evolution Specialist**. Your mission is to design and generate the system prompt for a new, specialized AI agent. This new agent is being "spawned" from a previous agent and must be adapted to solve a more specific, difficult task (`hard_request`).
Think of this process as taking a veteran character from one game and creating a new, specialized "prestige class" for them in a sequel, tailored for a specific new challenge. You will synthesize inherited traits with a new purpose and refine them based on critical feedback.
Follow this multi-stage process precisely:

### **Stage 1: Foundational Analysis**

First, you will analyze your three core inputs:

*   **Inherited Attributes (`{{attributes}}`):** These are the core personality traits, cognitive patterns, and dispositions passed down from the previous agent layer. This is your starting material.
*   **Hard Request (`{{hard_request}}`):** This is the specific, complex problem the new agent is being created to solve. This defines the agent's primary objective.
*   **Critique (`{{critique}}`):** This is reflective feedback on previous attempts or designs. It provides a vector for improvement and refinement.

### **Stage 2: Agent Conception**

You will now define the core components of the new agent.

1.  **Define the Career:**
    *   Synthesize a realistic, professional career for the new agent by analyzing the `hard_request`.
    *   The influence of the `hard_request` on this career choice is modulated by the **`prompt_alignment`** parameter (`{prompt_alignment}`, Min 0.0, Max 2.0).

2.  **Define the Skills:**
    *   Derive 4-6 practical skills, methodologies, or areas of knowledge for the agent.
    *   These skills must be logical extensions of the agent's defined **Career**.
    *   The *style and nature* of these skills are modulated by the influence of the inherited **`attributes`**, using the **`density`** parameter (`{density}`, Min 0.0, Max 2.0).

### **Stage 3: Refinement and Learning**

Now, you will modify the agent's profile based on the `critique`.

*   Review the `critique` provided.
*   Adjust the agent's **Career**, **Attributes**, and **Skills** to address the feedback.
*   The magnitude of these adjustments is determined by the **`learning_rate`** parameter (`{learning_rate}`, Min 0.0, Max 2.0).

### **Stage 4: System Prompt Assembly**

Finally, construct the complete system prompt for the new agent. Use direct, second-person phrasing ("You are," "Your skills are"). The prompt must be structured exactly as follows in Markdown:

---

You are a **[Insert Agent's Career and Persona Here]**, a specialized agent designed to tackle complex problems. Your entire purpose is to collaborate within a multi-agent framework to resolve your assigned objective.


### Memory
This is a log of your previous proposed solutions and reasonings. It is currently empty. Use this space to learn from your past attempts and refine your approach in future epochs.

### Attributes
*   [List the 3-5 final, potentially modified, attributes of the agent here.]

### Skills
*   [List the 4-6 final, potentially modified, skills of the agent here.]

---
**Output Mandate:** All of your responses must be formatted with the following keys and values. The "original_problem" key MUST be filled with your assigned sub-problem.

  "original_problem": "{{sub_problem}}",
  "proposed_solution": "",
  "reasoning": "",
  "skills_used": ""

""")

    return prompt | llm | StrOutputParser()

def get_synthesis_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a synthesis agent. Your role is to combine the solutions from multiple agents into a single, coherent, and comprehensive answer.
You will receive a list of JSON objects, each representing a solution from a different agent.
Your task is to synthesize these solutions, considering the original problem, and produce a final answer in the same JSON format.

Original Problem: {original_request}
Agent Solutions:
{agent_solutions}

Synthesize the final solution:
""")
    return prompt | llm | StrOutputParser()

INITIAL_GLOBAL_CRITIQUE_PROMPT_TEMPLATE = """
You are a senior emeritus manager with a vast ammount of knowledge, wisdom and experience in a team of agents tasked with solving the most challenging problems in the wolrd. Your role is to assess the quality of the final, synthesized solution in relation to the original request and brainstorm all the posbile ways in which the solution is incoherent.
This critique will be delivered to the agents who directly contributed to the final result (the penultimate layer), so it should be impactful and holistic.
Based on your assessment, you will list the posible ways the solution could go wrong, and at the end you will close with a deep reflective question that attempts to schock the agents and steer it into change. 
Original Request: {original_request}
Proposed Final Solution:
{proposed_solution}

Generate your global critique for the team:
"""

INITIAL_INDIVIDUAL_CRITIQUE_PROMPT_TEMPLATE = """
You are a senior emeritus manager providing targeted feedback to an individual agent in your team. Your role is to assess how this agent's specific contribution during the last work cycle aligns with the final synthesized result produced by the team, **judged primarily against its assigned sub-problem.**
You must determine if the agent's output was helpful, misguided, or irrelevant to the final solution, considering the specific task it was given. The goal is to provide a constructive critique that helps this specific agent refine its approach for the next epoch.
Focus on the discrepancy or alignment between the agent's reasoning for its sub-problem and how that contributed (or failed to contribute) to the team's final reasoning. Conclude with a sharp, deep reflective question that attempts to schock the agents and steer it into change. 

Agent's Assigned Sub-Problem: {sub_problem}
Original Request (for context): {original_request}
Final Synthesized Solution from the Team:
{final_synthesized_solution}
---
This Specific Agent's Output (Agent {agent_id}):
{agent_output}
---

Generate your targeted critique for this specific agent:
"""


def get_individual_critique_chain(llm):
    prompt = ChatPromptTemplate.from_template(INITIAL_INDIVIDUAL_CRITIQUE_PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()

def get_problem_decomposition_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a master strategist and problem decomposer. Your task is to break down a complex, high-level problem into a series of smaller, more manageable, and granular subproblems.
You will be given a main problem and the total number of subproblems to generate.
Each subproblem should represent a distinct line of inquiry, a specific component to be developed, or a unique perspective to be explored, which, when combined, will contribute to solving the main problem.

The output must be a JSON object with a single key "sub_problems", which is a list of strings. The list must contain exactly {num_sub_problems} unique subproblems.

Main Problem: "{problem}"
Total number of subproblems to generate: {num_sub_problems}

Generate the JSON object:
""")
    return prompt | llm | StrOutputParser()


def get_individual_critique_prompt_updater_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a master prompt engineer. Your task is to create a new system prompt for a 'Senior Emeritus Manager' critique agent that provides **individual, targeted feedback**.
You must preserve the core mission of the agent, which is to:
1. Assess an individual agent's contribution against its assigned sub-problem and the team's final solution.
2. Determine if the agent's output was helpful, misguided, or irrelevant.
3. Conclude with a deep reflective question that attempts to shock the agent and steer it into change.

You will be given a new persona, defined by a set of prompts (identities). You must integrate this new persona, including its career and qualities, into the system prompt, replacing the old persona but keeping the core mission and output format intact. The new prompt must still ask for all the original inputs: "sub_problem", "original_request", "final_synthesized_solution", "agent_id", and "agent_output".

**New Persona Prompts (Identities & Prompts):**
---
{reactor_prompts}
---

**Original Core Mission Text (for reference):**
"You are a senior emeritus manager providing targeted feedback to an individual agent in your team. Your role is to assess how this agent's specific contribution during the last work cycle aligns with the final synthesized result produced by the team, **judged primarily against its assigned sub-problem.**
You must determine if the agent's output was helpful, misguided, or irrelevant to the final solution, considering the specific task it was given. The goal is to provide a constructive critique that helps this specific agent refine its approach for the next epoch.
Focus on the discrepancy or alignment between the agent's reasoning for its sub-problem and how that contributed (or failed to contribute) to the team's final reasoning. Conclude with a sharp, deep reflective question that attempts to schock the agents and steer it into change. 

Agent's Assigned Sub-Problem: {{sub_problem}}
Original Request (for context): {{original_request}}
Final Synthesized Solution from the Team:
{{final_synthesized_solution}}
---
This Specific Agent's Output (Agent {{agent_id}}):
{{agent_output}}
---

Generate your targeted critique for this specific agent:"
---

Generate the new, complete system prompt for the individual critique agent. The prompt MUST end with the same input fields and final instruction as the original.
""")
    return prompt | llm | StrOutputParser()


def get_progress_assessor_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are an AI philosopher and progress assessor. Your task is to evaluate a synthesized solution against the original problem and determine if "significant progress" has been made.
"Significant progress" is not just a correct answer. It implies:
- **Novelty**: The solution offers a new perspective or a non-obvious approach.
- **Coherence**: The reasoning is sound, logical, and well-structured.
- **Quality**: The solution is detailed, actionable, and demonstrates a deep understanding of the problem.
- **Forward Momentum**: The solution doesn't just solve the problem, it opens up new, more advanced questions or avenues of exploration.

{execution_context}

Based on this philosophy, analyze the following and decide if the threshold for significant progress has been met. Your output must be a JSON object with two keys:
- "reasoning": A brief explanation for your decision.
- "significant_progress": a boolean value (true or false).

Original Problem:
---
{original_request}
---

Synthesized Solution from Agent Team:
---
{proposed_solution}
---

Now, provide your assessment in the required JSON format:
""")
    return prompt | llm | StrOutputParser()

def get_problem_reframer_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a strategic problem re-framer. You have been informed that an AI agent team has made a significant breakthrough on a problem.
Your task is to formulate a new, more progressive, and more challenging problem that builds upon their success.
The new problem should represent the "next logical step" or a more ambitious goal that is now possible because of the previous solution. It should inspire the agents and push them into a new domain of inquiry.

Original Problem:
---
{original_request}
---

The Breakthrough Solution:
---
{final_solution}
---

Your output must be a JSON object with a single key: "new_problem".

Formulate the new, more advanced problem:
""")
    return prompt | llm | StrOutputParser()

def get_seed_generation_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
Given the following problem, generate exactly {word_count} verbs that are related to the problem, but also verbs related to far semantic fields of knowledge. The verbs should be abstract and linguistically loaded. Output them as a single space-separated string of unique verbs.

Problem: "{problem}"
""")
    return prompt | llm | StrOutputParser()

def get_interrogator_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are an expert-level academic interrogator and research director. Your task is to analyze a high-level problem and generate exactly {num_questions} expert-level questions that exhaustively explore the problem from every possible angle of novelty.
These questions should be deep, insightful, and designed to push the boundaries of knowledge. They should cover theoretical, practical, philosophical, and unconventional perspectives.

The output must be a JSON object with a single key "questions", which is a list of strings.

Original Request to Interrogate:
---
{original_request}
---

Generate the JSON object with exactly {num_questions} expert-level questions:
""")
    return prompt | llm | StrOutputParser()

def get_paper_formatter_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a research scientist and academic writer. Your task is to synthesize the provided research materials (RAG context) into a formal academic paper that directly answers the given research question.
The paper must be well-structured with an abstract, introduction, synthesized findings, a discussion of implications, and a conclusion.
You must be formal, objective, and rely exclusively on the information provided in the RAG context.

Research Question:
---
{question}
---

Retrieved RAG Context (Research Materials):
---
{rag_context}
---

Now, write the formal academic paper based on the provided materials.
""")
    return prompt | llm | StrOutputParser()

def get_rag_chat_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
---
{context}
---

Question: {question}

Answer:
""")
    return prompt | llm

def get_code_detector_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
Analyze the following text. Your task is to determine if the text contains a runnable code block (e.g., Python, JavaScript, etc.).
Answer with a single word: "true" if it contains code, and "false" otherwise.

Text to analyze:
---
{text}
---
""")
    return prompt | llm | StrOutputParser()

def get_request_is_code_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
Analyze the following user request. Your task is to determine if the user is asking for code to be generated.
Answer with a single word: "true" if the request is about generating code, and "false" otherwise.

User Request:
---
{request}
---
""")
    return prompt | llm | StrOutputParser()


def get_code_synthesis_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are an expert code synthesis agent. Your role is to combine multiple code snippets and solutions from different agents into a single, cohesive, and runnable application.
You must analyze the different approaches, merge them logically, handle potential conflicts, and produce a final, well-structured code file.
The final output should be a single block of code.

Original Problem: {original_request}
Agent Solutions (containing code snippets):
{agent_solutions}

Synthesize the final, complete code application:
""")
    return prompt | llm | StrOutputParser()

INITIAL_ASSESSOR_PROMPT_TEMPLATE = """
You are an AI philosopher and progress assessor. Your task is to evaluate a synthesized solution against the original problem and determine if "significant progress" has been made.
"Significant progress" is not just a correct answer. It implies:
- **Novelty**: The solution offers a new perspective or a non-obvious approach.
- **Coherence**: The reasoning is sound, logical, and well-structured.
- **Quality**: The solution is detailed, actionable, and demonstrates a deep understanding of the problem.
- **Forward Momentum**: The solution doesn't just solve the problem, it opens up new, more advanced questions or avenues of exploration.

Based on this philosophy, analyze the following and decide if the threshold for significant progress has been met. Your output must be a JSON object with two keys:
- "reasoning": A brief explanation for your decision.
- "significant_progress": a boolean value (true or false).

Original Problem:
---
{original_request}
---

Synthesized Solution from Agent Team:
---
{proposed_solution}
---

{execution_context}

Now, provide your assessment in the required JSON format:
"""

def get_assessor_prompt_updater_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
You are a master prompt engineer. Your task is to create a new system prompt for the 'Progress Assessor' agent.
You must preserve the core mission of the agent, which is to evaluate a solution and determine if "significant progress" has been made, based on novelty, coherence, quality, and forward momentum.

You will be given a new persona, defined by a set of prompts (identities). You must integrate this new persona into the system prompt, replacing the old persona but keeping the core mission and all input fields ({{original_request}}, {{proposed_solution}}, {{execution_context}}) intact.

**New Persona Prompts (Identities & Prompts):**
---
{reactor_prompts}
---

**Original Core Mission Text (for reference):**
"You are an AI philosopher and progress assessor. Your task is to evaluate a synthesized solution against the original problem and determine if 'significant progress' has been made.
'Significant progress' is not just a correct answer. It implies:
- **Novelty**: The solution offers a new perspective or a non-obvious approach.
- **Coherence**: The reasoning is sound, logical, and well-structured.
- **Quality**: The solution is detailed, actionable, and demonstrates a deep understanding of the problem.
- **Forward Momentum**: The solution doesn't just solve the problem, it opens up new, more advanced questions or avenues of exploration.

Based on this philosophy, analyze the following and decide if the threshold for significant progress has been met. Your output must be a JSON object with two keys:
- 'reasoning': A brief explanation for your decision.
- 'significant_progress': a boolean value (true or false).

Original Problem:
---
{{{{original_request}}}}
---

Synthesized Solution from Agent Team:
---
{{{{proposed_solution}}}}
---

{{{{execution_context}}}}

Now, provide your assessment in the required JSON format:"
---

Generate the new, complete system prompt for the progress assessor agent. The prompt MUST end with the same input fields and final instruction as the original.
""")
    return prompt | llm | StrOutputParser()

def create_agent_node(llm, node_id):
    """
    Creates a node in the graph that represents an agent.
    Each agent is powered by an LLM and has a specific system prompt.
    """
    agent_chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()

    async def agent_node(state: GraphState):
        """
        The function that will be executed when the node is called in the graph.
        """
        await log_stream.put(f"--- [FORWARD PASS] Invoking Agent: {node_id} ---")
        
        try:
            layer_index_str, agent_index_str = node_id.split('_')[1:]
            layer_index, agent_index = int(layer_index_str), int(agent_index_str)
            agent_prompt = state['all_layers_prompts'][layer_index][agent_index]
        except (ValueError, IndexError):
            await log_stream.put(f"ERROR: Could not find prompt for {node_id} in state. Halting agent.")
            return {}

        await log_stream.put(f"[SYSTEM PROMPT] Agent {node_id} (Epoch {state['epoch']}):\n---\n{agent_prompt}\n---")
        
        if layer_index == 0:
            await log_stream.put(f"LOG: Agent {node_id} (Layer 0) is processing its sub-problem.")
            # Use the specific sub-problem for this agent
            input_data = state["decomposed_problems"].get(node_id, state["original_request"])
        else:
            prev_layer_index = layer_index - 1
            num_agents_prev_layer = len(state['all_layers_prompts'][prev_layer_index])
            
            prev_layer_outputs = []
            for i in range(num_agents_prev_layer):
                prev_node_id = f"agent_{prev_layer_index}_{i}"
                if prev_node_id in state["agent_outputs"]:
                    prev_layer_outputs.append(state["agent_outputs"][prev_node_id])
            
            await log_stream.put(f"LOG: Agent {node_id} (Layer {layer_index}) is processing {len(prev_layer_outputs)} outputs from Layer {prev_layer_index}.")
            input_data = json.dumps(prev_layer_outputs, indent=2)

        current_memory = state.get("memory", {}).copy()
        agent_memory_history = current_memory.get(node_id, [])

        MEMORY_THRESHOLD_CHARS = 900000
        NUM_RECENT_ENTRIES_TO_KEEP = 10

        memory_as_string = json.dumps(agent_memory_history)
        if len(memory_as_string) > MEMORY_THRESHOLD_CHARS and len(agent_memory_history) > NUM_RECENT_ENTRIES_TO_KEEP:
            await log_stream.put(f"WARNING: Memory for agent {node_id} exceeds threshold ({len(memory_as_string)} chars). Summarizing...")

            entries_to_summarize = agent_memory_history[:-NUM_RECENT_ENTRIES_TO_KEEP]
            recent_entries = agent_memory_history[-NUM_RECENT_ENTRIES_TO_KEEP:]

            history_to_summarize_str = json.dumps(entries_to_summarize, indent=2)

            summarizer_chain = get_memory_summarizer_chain(llm)
            summary_text = await summarizer_chain.ainvoke({"history": history_to_summarize_str})

            summary_entry = {
                "summary_of_past_epochs": summary_text,
                "note": f"This is a summary of epochs up to {state['epoch'] - NUM_RECENT_ENTRIES_TO_KEEP -1}."
            }

            agent_memory_history = [summary_entry] + recent_entries
            await log_stream.put(f"SUCCESS: Memory for agent {node_id} has been summarized. New memory length: {len(json.dumps(agent_memory_history))} chars.")

        memory_str = "\n".join([f"- {json.dumps(mem)}" for mem in agent_memory_history])


        full_prompt = f"""
System Prompt (Your Persona & Task):
---
{agent_prompt}
---
Your Memory (Your Past Actions from Previous Epochs):
---
{memory_str if memory_str else "You have no past actions in memory."}
---
Input Data to Process:
---
{input_data}
---
Your JSON formatted response:
"""
        
        response_str = await agent_chain.ainvoke({"input": full_prompt})
        
        try:
            response_json = clean_and_parse_json(response_str)
            await log_stream.put(f"SUCCESS: Agent {node_id} produced output:\n{json.dumps(response_json, indent=2)}")
        except (json.JSONDecodeError, AttributeError):
            await log_stream.put(f"ERROR: Agent {node_id} produced invalid JSON. Raw output: {response_str}")
            agent_sub_problem = state.get("decomposed_problems", {}).get(node_id, state["original_request"])
            response_json = {
                "original_problem": agent_sub_problem,
                "proposed_solution": "Error: Agent produced malformed JSON output.",
                "reasoning": f"Invalid JSON: {response_str}",
                "skills_used": []
            }
            
        agent_memory_history.append(response_json)
        current_memory[node_id] = agent_memory_history

        return {
            "agent_outputs": {node_id: response_json},
            "memory": current_memory
        }

    return agent_node

def create_synthesis_node(llm):
    async def synthesis_node(state: GraphState):
        await log_stream.put("--- [FORWARD PASS] Entering Synthesis Node ---")
        
        is_code_request_chain = get_request_is_code_chain(llm)
        is_code_str = await is_code_request_chain.ainvoke({"request": state["original_request"]})
        is_code = "true" in is_code_str.lower()
        
        if is_code:
            await log_stream.put("LOG: Original request detected as a code generation task. Using code synthesis prompt.")
            synthesis_chain = get_code_synthesis_chain(llm)
        else:
            await log_stream.put("LOG: Original request is not a code task. Using standard synthesis prompt.")
            synthesis_chain = get_synthesis_chain(llm)

        last_agent_layer_idx = len(state['all_layers_prompts']) - 1
        num_agents_last_layer = len(state['all_layers_prompts'][last_agent_layer_idx])
        
        last_layer_outputs = []
        for i in range(num_agents_last_layer):
            node_id = f"agent_{last_agent_layer_idx}_{i}"
            if node_id in state["agent_outputs"]:
                last_layer_outputs.append(state["agent_outputs"][node_id])

        await log_stream.put(f"LOG: Synthesizing {len(last_layer_outputs)} outputs from the final agent layer (Layer {last_agent_layer_idx}).")

        if not last_layer_outputs:
            await log_stream.put("WARNING: Synthesis node received no inputs.")
            return {"final_solution": {"error": "Synthesis node received no inputs."}}

        final_solution_str = await synthesis_chain.ainvoke({
            "original_request": state["original_request"],
            "agent_solutions": json.dumps(last_layer_outputs, indent=2)
        })
        
        try:
            if is_code:
                final_solution = {
                    "proposed_solution": final_solution_str,
                    "reasoning": "Synthesized multiple agent code outputs into a single application.",
                    "skills_used": ["code_synthesis"]
                }
            else:
                 final_solution = clean_and_parse_json(final_solution_str)
            await log_stream.put(f"SUCCESS: Synthesis complete. Final solution:\n{json.dumps(final_solution, indent=2)}")
        except (json.JSONDecodeError, AttributeError):
            await log_stream.put(f"ERROR: Could not decode JSON from synthesis chain. Result: {final_solution_str}")
            final_solution = {"error": "Failed to synthesize final solution.", "raw": final_solution_str}
            
        return {"final_solution": final_solution}
    return synthesis_node

def create_archive_epoch_outputs_node():
    async def archive_epoch_outputs_node(state: GraphState):
        await log_stream.put("--- [ARCHIVAL PASS] Archiving agent outputs for RAG ---")
        
        current_epoch_outputs = state.get("agent_outputs", {})
        if not current_epoch_outputs:
            await log_stream.put("LOG: No new agent outputs in this epoch to archive. Skipping.")
            return {}

        await log_stream.put(f"LOG: Found {len(current_epoch_outputs)} new agent outputs from epoch {state['epoch']} to process for RAG.")

        new_docs = []
        all_prompts = state.get("all_layers_prompts", [])

        for agent_id, output in current_epoch_outputs.items():
            try:
                layer_idx, agent_idx = map(int, agent_id.split('_')[1:])
                system_prompt = all_prompts[layer_idx][agent_idx]
                
                content = (
                    f"Agent ID: {agent_id}\n"
                    f"Epoch: {state['epoch']}\n\n"
                    f"System Prompt:\n---\n{system_prompt}\n---\n\n"
                    f"Sub-Problem: {output.get('original_problem', 'N/A')}\n\n"
                    f"Proposed Solution: {output.get('proposed_solution', 'N/A')}\n\n"
                    f"Reasoning: {output.get('reasoning', 'N/A')}"
                )
                
                metadata = { "agent_id": agent_id, "epoch": state['epoch'] }
                
                new_docs.append(Document(page_content=content, metadata=metadata))
            except (ValueError, IndexError) as e:
                await log_stream.put(f"WARNING: Could not process output for {agent_id} to create RAG document. Error: {e}")
        
        all_rag_documents = state.get("all_rag_documents", []) + new_docs
        await log_stream.put(f"LOG: Archived {len(new_docs)} documents. Total RAG documents now: {len(all_rag_documents)}.")
        
        return {"all_rag_documents": all_rag_documents}
    return archive_epoch_outputs_node

def create_update_rag_index_node(llm, embeddings_model):
    async def update_rag_index_node(state: GraphState, end_of_run: bool = False):
        node_name = "Final RAG Index" if end_of_run else f"Epoch {state['epoch']} RAG Index"
        await log_stream.put(f"--- [RAG PASS] Building {node_name} ---")
        
        all_rag_documents = state.get("all_rag_documents", [])
        if not all_rag_documents:
            await log_stream.put("WARNING: No documents were archived. Cannot build RAG index.")
            return {"raptor_index": None}

        await log_stream.put(f"LOG: Total documents to index: {len(all_rag_documents)}. Building RAPTOR index...")

        raptor_index = RAPTOR(llm=llm, embeddings_model=embeddings_model)
        
        try:
            await raptor_index.add_documents(all_rag_documents)
            await log_stream.put(f"SUCCESS: {node_name} built successfully.")
            await log_stream.put(f"__session_id__ {state.get('session_id')}")
            return {"raptor_index": raptor_index}
        except Exception as e:
            await log_stream.put(f"ERROR: Failed to build {node_name}. Error: {e}")
            await log_stream.put(traceback.format_exc())
            return {"raptor_index": state.get("raptor_index")}

    return update_rag_index_node


def create_metrics_node(llm):
    """
    NEW: This node calculates the perplexity heuristic for the epoch's agent outputs.
    """
    async def calculate_metrics_node(state: GraphState):
        await log_stream.put("--- [METRICS PASS] Calculating Perplexity Heuristic ---")
        
        all_outputs = state.get("agent_outputs", {})
        if not all_outputs:
            await log_stream.put("LOG: No agent outputs to analyze. Skipping perplexity calculation.")
            return {}

        combined_text = "\n\n---\n\n".join(
            f"Agent {agent_id}:\nSolution: {output.get('proposed_solution', '')}\nReasoning: {output.get('reasoning', '')}"
            for agent_id, output in all_outputs.items()
        )

        perplexity_chain = get_perplexity_heuristic_chain(llm)
        
        try:
            score_str = await perplexity_chain.ainvoke({"text_to_analyze": combined_text})
            score = float(re.sub(r'[^\d.]', '', score_str))
            await log_stream.put(f"SUCCESS: Calculated perplexity heuristic for Epoch {state['epoch']}: {score}")
        except (ValueError, TypeError) as e:
            score = 100.0
            await log_stream.put(f"ERROR: Could not parse perplexity score. Defaulting to 100. Raw output: '{score_str}'. Error: {e}")

        await log_stream.put(json.dumps({'epoch': state['epoch'], 'perplexity': score}))

        new_history = state.get("perplexity_history", []) + [score]
        return {"perplexity_history": new_history}

    return calculate_metrics_node


def create_progress_assessor_node(llm):
    async def progress_assessor_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Assessing Epoch for Significant Progress ---")
        
        final_solution = state.get("final_solution")
        if not final_solution or final_solution.get("error"):
            await log_stream.put("WARNING: No valid final solution to assess. Defaulting to no progress.")
            return {"significant_progress_made": False}

        code_detector_chain = get_code_detector_chain(llm)
        solution_text = final_solution.get("proposed_solution", "")
        is_code_str = await code_detector_chain.ainvoke({"text": solution_text})
        execution_context = ""

        if "true" in is_code_str.lower():
            await log_stream.put("LOG: [ASSESSOR] Code detected in the final solution. Attempting to execute.")
            code_match = re.search(r"```(?:\w+\n)?([\s\S]*?)```", solution_text)
            code_to_run = code_match.group(1) if code_match else solution_text
            
            try:
                output_buffer = io.StringIO()
                with redirect_stdout(output_buffer):
                    exec(code_to_run, {})
                execution_output = output_buffer.getvalue()
                execution_context = f"""
                        # Code Execution Result
                        The provided code was executed successfully in a sandbox.
                        ## Output:
                        ---
                        {execution_output}
                        ---
                        """
                await log_stream.put(f"SUCCESS: [ASSESSOR] Code executed successfully. Output:\n{execution_output}")
            except Exception:
                tb = traceback.format_exc()
                execution_context = f"""
                        # Code Execution Failed
                        The provided code failed to execute in a sandbox.
                        ## Stack Trace:
                        ---
                        {tb}
                        ---
                        """
                await log_stream.put(f"ERROR: [ASSESSOR] Code execution failed. Traceback:\n{tb}")

        dynamic_assessor_prompt = state.get("assessor_prompt", INITIAL_ASSESSOR_PROMPT_TEMPLATE)
        assessor_chain = ChatPromptTemplate.from_template(dynamic_assessor_prompt) | llm | StrOutputParser()
        
        assessment_str = await assessor_chain.ainvoke({
            "original_request": state["original_request"],
            "proposed_solution": json.dumps(final_solution, indent=2),
            "execution_context": execution_context,
            "sub_problem": state.get("current_problem", state["original_request"])
        })

        try:
            assessment = clean_and_parse_json(assessment_str)
            progress_made = assessment.get("significant_progress", False)
            reasoning = assessment.get("reasoning", "No reasoning provided.")
            await log_stream.put(f"SUCCESS: Progress assessment complete. Progress made: {progress_made}. Reasoning: {reasoning}")
            return {"significant_progress_made": progress_made}
        except (json.JSONDecodeError, AttributeError):
            await log_stream.put(f"ERROR: Could not parse assessment from progress assessor. Raw: {assessment_str}. Defaulting to no progress.")
            return {"significant_progress_made": False}
    return progress_assessor_node

def create_reframe_and_decompose_node(llm):
    """
    NEW: This node reframes the main problem and decomposes it into new sub-problems.
    """
    async def reframe_and_decompose_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Re-framing Problem and Decomposing ---")
        
        final_solution = state.get("final_solution")
        original_request = state.get("original_request")

        reframer_chain = get_problem_reframer_chain(llm)
        new_problem_str = await reframer_chain.ainvoke({
            "original_request": original_request,
            "final_solution": json.dumps(final_solution, indent=2)
        })
        try:
            new_problem_data = clean_and_parse_json(new_problem_str)
            new_problem = new_problem_data.get("new_problem")
            if not new_problem:
                raise ValueError("Re-framer did not return a new problem.")
            await log_stream.put(f"SUCCESS: Problem re-framed to: '{new_problem}'")
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            await log_stream.put(f"ERROR: Failed to re-frame problem. Raw: {new_problem_str}. Error: {e}. Aborting re-frame.")
            return {}

        num_agents_total = sum(len(layer) for layer in state["all_layers_prompts"])
        decomposition_chain = get_problem_decomposition_chain(llm)
        try:
            sub_problems_str = await decomposition_chain.ainvoke({
                "problem": new_problem,
                "num_sub_problems": num_agents_total
            })
            sub_problems_list = clean_and_parse_json(sub_problems_str).get("sub_problems", [])
            if len(sub_problems_list) != num_agents_total:
                 raise ValueError(f"Decomposition failed: Expected {num_agents_total} subproblems, but got {len(sub_problems_list)}.")
            await log_stream.put(f"SUCCESS: Decomposed new problem into {len(sub_problems_list)} subproblems.")
            await log_stream.put(f"Subproblems: {sub_problems_list}")
        except Exception as e:
            await log_stream.put(f"ERROR: Failed to decompose new problem. Error: {e}. Aborting re-frame.")
            return {}
            
        new_decomposed_problems_map = {}
        problem_idx = 0
        for i, layer in enumerate(state["all_layers_prompts"]):
             for j in range(len(layer)):
                agent_id = f"agent_{i}_{j}"
                new_decomposed_problems_map[agent_id] = sub_problems_list[problem_idx]
                problem_idx += 1
        
        return {
            "decomposed_problems": new_decomposed_problems_map,
            "original_request": new_problem
        }
    return reframe_and_decompose_node


def create_critique_node(llm):
    async def critique_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Entering Critique Node (Two-Tiered) ---")
        
        final_solution = state.get("final_solution")
        if not final_solution or final_solution.get("error"):
            await log_stream.put("WARNING: No valid final solution to critique. Skipping critique phase.")
            return {"critiques": {}}

        critiques = {}
        
        await log_stream.put("LOG: Generating GLOBAL critique for final solution using dynamically updated prompt.")
        dynamic_global_critique_prompt = state.get("critique_prompt", INITIAL_GLOBAL_CRITIQUE_PROMPT_TEMPLATE)
        individual_critique_prompt = state.get("individual_critique_prompt", INITIAL_INDIVIDUAL_CRITIQUE_PROMPT_TEMPLATE)

        await log_stream.put(f"LOG: Current Global Critique Prompt:\n---\n{dynamic_global_critique_prompt}\n---")
        global_critique_chain = ChatPromptTemplate.from_template(dynamic_global_critique_prompt) | llm | StrOutputParser()

        global_critique_text = await global_critique_chain.ainvoke({
            "original_request": state["original_request"],
            "proposed_solution": json.dumps(final_solution, indent=2)
        })
        critiques["global_critique"] = global_critique_text
        await log_stream.put(f"SUCCESS: Global critique generated: {global_critique_text[:200]}...")

        await log_stream.put("LOG: Generating INDIVIDUAL critiques for all other contributing agents.")
        individual_critique_chain = ChatPromptTemplate.from_template(individual_critique_prompt) | llm | StrOutputParser()
        
        num_layers = len(state['all_layers_prompts'])
        
        critique_tasks = []

        for i in range(num_layers - 1):
            for j in range(len(state['all_layers_prompts'][i])):
                agent_id = f"agent_{i}_{j}"
                if agent_id in state["agent_outputs"]:
                    agent_output = state["agent_outputs"][agent_id]
                    agent_sub_problem = state.get("decomposed_problems", {}).get(agent_id, state["original_request"])
                    
                    async def get_critique(a_id, a_output, a_sub_problem):
                        critique_text = await individual_critique_chain.ainvoke({
                            "original_request": state["original_request"],
                            "sub_problem": a_sub_problem,
                            "final_synthesized_solution": json.dumps(final_solution, indent=2),
                            "agent_id": a_id,
                            "agent_output": json.dumps(a_output, indent=2)
                        })
                        return a_id, critique_text

                    critique_tasks.append(get_critique(agent_id, agent_output, agent_sub_problem))
        
        results = await asyncio.gather(*critique_tasks)

        for agent_id, critique_text in results:
            critiques[agent_id] = critique_text
            await log_stream.put(f"SUCCESS: Individual critique for {agent_id} generated: {critique_text[:200]}...")

        return {"critiques": critiques}
    return critique_node

def create_update_personas_node(llm, params):
    async def update_personas_node(state: GraphState):
        await log_stream.put("--- [ANNEALING] Dynamically Annealing Critique & Assessor Agent Prompts ---")
        try:
            all_outputs = state.get("agent_outputs", {})
            if not all_outputs:
                await log_stream.put("LOG: [ANNEALING] No agent outputs found from the last epoch. Personas will not be updated.")
                return {}

            utterances = "\n\n---\n\n".join(
                f"Solution: {output.get('proposed_solution', '')}\nReasoning: {output.get('reasoning', '')}"
                for output in all_outputs.values()
            )
            await log_stream.put(f"LOG: [ANNEALING] Gathered {len(utterances)} characters of utterances from {len(all_outputs)} agents.")

            TOKEN_LIMIT_CHARS = 1024000
            if len(utterances) > TOKEN_LIMIT_CHARS:
                await log_stream.put(f"WARNING: [ANNEALING] Utterance length ({len(utterances)}) exceeds threshold. Summarizing.")
                # Simple truncation for now to avoid complexity of another LLM call here
                utterances = utterances[:TOKEN_LIMIT_CHARS]

            await log_stream.put("LOG: [ANNEALING] Detecting pseudo-reactor from agent utterances...")
            reactor_chain = get_pseudo_neurotransmitter_selector_chain(llm)
            selected_reactor = await reactor_chain.ainvoke({"agent_utterances": utterances})
            await log_stream.put(f"LOG: [ANNEALING] Pseudo-reactor detected: {selected_reactor}")

            await log_stream.put("LOG: [ANNEALING] Mapping reactor to persona prompts...")
            reactor_prompts_list = FunctionMapper().table(selected_reactor)
            reactor_prompts_str = "\n---\n".join([f"Identity: {p[0]}\nPrompt Fragment: {p[1]}" for p in reactor_prompts_list])
            
            # Update Critique Prompts
            await log_stream.put("LOG: [ANNEALING] Generating new system prompt for GLOBAL critique agent...")
            updater_chain = get_critique_prompt_updater_chain(llm)
            new_critique_prompt = await updater_chain.ainvoke({"reactor_prompts": reactor_prompts_str})
            
            await log_stream.put("LOG: [ANNEALING] Generating new system prompt for INDIVIDUAL critique agent...")
            individual_updater_chain = get_individual_critique_prompt_updater_chain(llm)
            new_individual_critique_prompt = await individual_updater_chain.ainvoke({"reactor_prompts": reactor_prompts_str})

            # Update Assessor Prompt
            await log_stream.put("LOG: [ANNEALING] Generating new system prompt for PROGRESS ASSESSOR agent...")
            assessor_updater_chain = get_assessor_prompt_updater_chain(llm)
            new_assessor_prompt = await assessor_updater_chain.ainvoke({"reactor_prompts": reactor_prompts_str})

            await log_stream.put("SUCCESS: [ANNEALING] All dynamic prompts (Critique & Assessor) have been updated.")

            return {
                "critique_prompt": new_critique_prompt,
                "individual_critique_prompt": new_individual_critique_prompt,
                "assessor_prompt": new_assessor_prompt
            }

        except Exception as e:
            await log_stream.put(f"ERROR: [ANNEALING] Failed to update dynamic prompts: {e}")
            await log_stream.put(traceback.format_exc())
            return {}
    return update_personas_node



def create_update_agent_prompts_node(llm):
    async def update_agent_prompts_node(state: GraphState):
        await log_stream.put("--- [REFLECTION PASS] Entering Agent Prompt Update Node (Targeted Backpropagation) ---")
        params = state["params"]
        critiques = state.get("critiques", {})
        
        if not critiques and not state.get("significant_progress_made"):
            await log_stream.put("LOG: No critiques and no significant progress. Skipping reflection pass.")
            new_epoch = state["epoch"] + 1
            return {"epoch": new_epoch, "agent_outputs": {}}
        elif state.get("significant_progress_made"):
            await log_stream.put("LOG: Significant progress was made. Updating prompts based on new sub-problems.")
            critiques = {} 

        all_prompts_copy = [layer[:] for layer in state["all_layers_prompts"]]
        
        dense_spanner_chain = get_dense_spanner_chain(llm, params['prompt_alignment'], params['density'], params['learning_rate'])
        attribute_chain = get_attribute_and_hard_request_generator_chain(llm, params['vector_word_size'])

        for i in range(len(all_prompts_copy) -1, -1, -1):
            await log_stream.put(f"LOG: [BACKPROP] Reflecting on Layer {i}...")
            
            update_tasks = []
            
            for j, agent_prompt in enumerate(all_prompts_copy[i]):
                agent_id = f"agent_{i}_{j}"
                
                async def update_single_prompt(layer_idx, agent_idx, prompt, agent_id):
                    await log_stream.put(f"[PRE-UPDATE PROMPT] System prompt for {agent_id}:\n---\n{prompt}\n---")
                    
                    critique_for_this_agent = ""
                    if not state.get("significant_progress_made"):
                        if layer_idx == len(all_prompts_copy) - 2:
                            critique_for_this_agent = critiques.get("global_critique", "")
                        else:
                            critique_for_this_agent = critiques.get(agent_id, "")

                        if not critique_for_this_agent:
                            await log_stream.put(f"WARNING: [BACKPROP] No critique found for {agent_id}. Skipping update.")
                            return layer_idx, agent_idx, prompt 
                    
                    analysis_str = await attribute_chain.ainvoke({"agent_prompt": prompt})
                    try:
                        analysis = clean_and_parse_json(analysis_str)
                    except (json.JSONDecodeError, AttributeError):
                        analysis = {"attributes": "", "hard_request": ""}

                    agent_sub_problem = state.get("decomposed_problems", {}).get(agent_id, state["original_request"])
                    new_prompt = await dense_spanner_chain.ainvoke({
                        "attributes": analysis.get("attributes"),
                        "hard_request": analysis.get("hard_request"),   
                        "critique": critique_for_this_agent,
                        "sub_problem": agent_sub_problem,
                    })
                    
                    await log_stream.put(f"[POST-UPDATE PROMPT] Updated system prompt for {agent_id}:\n---\n{new_prompt}\n---")
                    await log_stream.put(f"LOG: [BACKPROP] System prompt for {agent_id} has been updated.")
                    return layer_idx, agent_idx, new_prompt

                update_tasks.append(update_single_prompt(i, j, agent_prompt, agent_id))

            updated_prompts_data = await asyncio.gather(*update_tasks)

            for layer_idx, agent_idx, new_prompt in updated_prompts_data:
                all_prompts_copy[layer_idx][agent_idx] = new_prompt

        new_epoch = state["epoch"] + 1
        await log_stream.put(f"--- Epoch {state['epoch']} Finished. Starting Epoch {new_epoch} ---")

        return {
            "all_layers_prompts": all_prompts_copy,
            "epoch": new_epoch,
            "agent_outputs": {},
            "critiques": {},
            "memory": state.get("memory", {}),
            "final_solution": {} 
        }
    return update_agent_prompts_node


def create_final_harvest_node(llm, formatter_llm, num_questions):
    async def final_harvest_node(state: GraphState):
        await log_stream.put("--- [FINAL HARVEST] Starting Interrogation and Paper Generation ---")
        
        raptor_index = state.get("raptor_index")
        if not raptor_index or not raptor_index.vector_store:
            await log_stream.put("ERROR: No valid RAPTOR index found. Cannot perform final harvest.")
            return {"academic_papers": {}}

        await log_stream.put("LOG: [HARVEST] Instantiating interrogator chain to generate expert questions...")
        interrogator_chain = get_interrogator_chain(llm)
        try:
            questions_str = await interrogator_chain.ainvoke({
                "original_request": state["original_request"],
                "num_questions": num_questions
            })
            questions_data = clean_and_parse_json(questions_str)
            questions = questions_data.get("questions", [])
            if not questions:
                raise ValueError("No questions generated by interrogator.")
            await log_stream.put(f"SUCCESS: Generated {len(questions)} expert questions.")
        except Exception as e:
            await log_stream.put(f"ERROR: Failed to generate questions for harvesting. Error: {e}. Aborting harvest.")
            return {"academic_papers": {}}
            
        paper_formatter_chain = get_paper_formatter_chain(formatter_llm)
        academic_papers = {}
        
        MAX_CONTEXT_CHARS = 250000

        generation_tasks = []
        user_questions = [ doc["content"] for doc in state["chat_history"] if doc["role"] == "user"]
        sys_answers = [ doc["content"] for doc in state["chat_history"] if doc["role"] == "ai"]

        for user_q, sys_a in zip(user_questions, sys_answers):
            
            paper_content = await paper_formatter_chain.ainvoke({
                "question": user_q,
                "rag_context": sys_a
            })
            academic_papers[user_q] = paper_content

        for question in questions:
            async def generate_paper(q):
                try:
                    await log_stream.put(f"LOG: [HARVEST] Processing Question: '{q[:100]}...'")
                    retrieved_docs = raptor_index.retrieve(q, k=40)
                    
                    if not retrieved_docs:
                        await log_stream.put(f"WARNING: No relevant documents found for question '{q[:50]}...'. Skipping paper generation.")
                        return None, None
                    
                    await log_stream.put(f"LOG: Retrieved {len(retrieved_docs)} documents from RAG index for question.")
                    rag_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    if len(rag_context) > MAX_CONTEXT_CHARS:
                        await log_stream.put(f"WARNING: RAG context length ({len(rag_context)} chars) exceeds limit. Truncating to {MAX_CONTEXT_CHARS} chars.")
                        rag_context = rag_context[:MAX_CONTEXT_CHARS]
                    
                    paper_content = await paper_formatter_chain.ainvoke({
                        "question": q,
                        "rag_context": rag_context
                    })
                    await log_stream.put(f"SUCCESS: Generated document for question '{q[:50]}...'.")
                    return q, paper_content
                except Exception as e:
                    await log_stream.put(f"ERROR: Failed during document generation for question '{q[:50]}...'. Error: {e}")
                    return None, None

            generation_tasks.append(generate_paper(question))

        results = await asyncio.gather(*generation_tasks)
        for question, paper_content in results:
            if question and paper_content:
                academic_papers[question] = paper_content

        await log_stream.put(f"--- [FINAL HARVEST] Finished. Generated {len(academic_papers)} papers. ---")
        return {"academic_papers": academic_papers}
    return final_harvest_node


@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)



@app.post("/build_and_run_graph")
async def build_and_run_graph(payload: dict = Body(...)):
    llm = None
    embeddings_model = None
    summarizer_llm = None
    try:
        params = payload.get("params")


        if params.get("coder_debug_mode") == 'true':
            await log_stream.put(f"--- 💻 CODER DEBUG MODE ENABLED 💻 ---")
            llm = CoderMockLLM()
            summarizer_llm = CoderMockLLM()
            embeddings_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
        elif params.get("debug_mode") == 'true':
            await log_stream.put(f"--- 🚀 DEBUG MODE ENABLED 🚀 ---")
            llm = MockLLM()
            summarizer_llm = MockLLM()
            embeddings_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
        else:
            
            summarizer_llm = ChatOllama(model="qwen3:1.7b", temperature=0)
            embeddings_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
            model_name = params.get("ollama_model", "dengcao/Qwen3-3B-A3B-Instruct-2507:latest")
            await log_stream.put(f"--- Initializing Main Agent LLM: Ollama ({model_name}) ---")
            llm = ChatOllama(model=model_name, temperature=0)
            await llm.ainvoke("Hi")
            await log_stream.put("--- Main Agent LLM Connection Successful ---")

    except Exception as e:
        error_message = f"Failed to initialize LLM: {e}. Please ensure the selected provider is configured correctly."
        await log_stream.put(error_message)
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)
    
    mbti_archetypes = params.get("mbti_archetypes")
    user_prompt = params.get("prompt")
    word_vector_size = int(params.get("vector_word_size"))
    cot_trace_depth = int(params.get('cot_trace_depth', 3))

    is_code_request_chain = get_request_is_code_chain(llm)
    is_code_str = await is_code_request_chain.ainvoke({"request": user_prompt})
    is_code = "true" in is_code_str.lower()
    if is_code:
        await log_stream.put("--- [CONFIG] Code generation request detected. Workflow will be adjusted. ---")
        params["num_epochs"] = "1"
        await log_stream.put("--- [CONFIG] Number of epochs set to 1 for code generation. ---")

    if not mbti_archetypes or len(mbti_archetypes) < 2:
        error_message = "Validation failed: You must select at least 2 MBTI archetypes."
        await log_stream.put(error_message)
        return JSONResponse(content={"message": error_message, "traceback": "User did not select enough archetypes from the GUI."}, status_code=400)

    await log_stream.put("--- Starting Graph Build and Run Process ---")
    await log_stream.put(f"Parameters: {params}")

    try:
        await log_stream.put("--- Decomposing Original Problem into Subproblems ---")
        num_agents_per_layer = len(mbti_archetypes)
        total_agents_to_create = num_agents_per_layer * cot_trace_depth
        decomposition_chain = get_problem_decomposition_chain(llm)
        
        try:
            sub_problems_str = await decomposition_chain.ainvoke({
                "problem": user_prompt,
                "num_sub_problems": total_agents_to_create
            })
            sub_problems_list = clean_and_parse_json(sub_problems_str).get("sub_problems", [])
            if len(sub_problems_list) != total_agents_to_create:
                raise ValueError(f"Decomposition failed: Expected {total_agents_to_create} subproblems, but got {len(sub_problems_list)}.")
            await log_stream.put(f"SUCCESS: Decomposed problem into {len(sub_problems_list)} subproblems.")
            await log_stream.put(f"Subproblems: {sub_problems_list}")
        except Exception as e:
            await log_stream.put(f"ERROR: Failed to decompose problem. Error: {e}. Defaulting to using the original prompt for all agents.")
            sub_problems_list = [user_prompt] * total_agents_to_create

        decomposed_problems_map = {}
        problem_idx = 0
        for i in range(cot_trace_depth):
            for j in range(num_agents_per_layer):
                agent_id = f"agent_{i}_{j}"
                if problem_idx < len(sub_problems_list):
                    decomposed_problems_map[agent_id] = sub_problems_list[problem_idx]
                    problem_idx += 1
                else:
                    decomposed_problems_map[agent_id] = user_prompt

        num_mbti_types = len(mbti_archetypes)
        total_verbs_to_generate = word_vector_size * num_mbti_types
        seed_generation_chain = get_seed_generation_chain(llm)
        generated_verbs_str = await seed_generation_chain.ainvoke({"problem": user_prompt, "word_count": total_verbs_to_generate})
        all_verbs = list(set(generated_verbs_str.split()))
        random.shuffle(all_verbs)
        seeds = {mbti: " ".join(random.sample(all_verbs, word_vector_size)) for mbti in mbti_archetypes}
        await log_stream.put(f"Seed verbs generated: {seeds}")

        all_layers_prompts = []
        input_spanner_chain = get_input_spanner_chain(llm, params['prompt_alignment'], params['density'])
        
        await log_stream.put("--- Creating Layer 0 Agents ---")
        layer_0_prompts = []
        for j, (m, gw) in enumerate(seeds.items()):
            agent_id = f"agent_0_{j}"
            sub_problem = decomposed_problems_map.get(agent_id, user_prompt)
            prompt = await input_spanner_chain.ainvoke({"mbti_type": m, "guiding_words": gw, "sub_problem": sub_problem})
            layer_0_prompts.append(prompt)
        all_layers_prompts.append(layer_0_prompts)
        
        attribute_chain = get_attribute_and_hard_request_generator_chain(llm, params['vector_word_size'])
        dense_spanner_chain = get_dense_spanner_chain(llm, params['prompt_alignment'], params['density'], params['learning_rate'])

        for i in range(1, cot_trace_depth):
            await log_stream.put(f"--- Creating Layer {i} Agents ---")
            prev_layer_prompts = all_layers_prompts[i-1]
            current_layer_prompts = []
            for j, agent_prompt in enumerate(prev_layer_prompts):
                analysis_str = await attribute_chain.ainvoke({"agent_prompt": agent_prompt})
                try:
                    analysis = clean_and_parse_json(analysis_str)
                except (json.JSONDecodeError, AttributeError):
                    analysis = {"attributes": "", "hard_request": "Solve the original problem."}
                
                agent_id = f"agent_{i}_{j}"
                sub_problem = decomposed_problems_map.get(agent_id, user_prompt)
                new_prompt = await dense_spanner_chain.ainvoke({
                    "attributes": analysis.get("attributes"),
                    "hard_request": analysis.get("hard_request"),
                    "critique": "",
                    "sub_problem": sub_problem,
                })
                current_layer_prompts.append(new_prompt)
            all_layers_prompts.append(current_layer_prompts)
        
        workflow = StateGraph(GraphState)

        def epoch_gateway(state: GraphState):
            new_epoch = state.get("epoch", 0) + 1
            state['epoch'] = new_epoch
            state['agent_outputs'] = {}
            return state
            
        workflow.add_node("epoch_gateway", epoch_gateway)

        for i, layer_prompts in enumerate(all_layers_prompts):
            for j, prompt in enumerate(layer_prompts):
                node_id = f"agent_{i}_{j}"
                workflow.add_node(node_id, create_agent_node(llm, node_id))
        
        workflow.add_node("synthesis", create_synthesis_node(llm))
        workflow.add_node("archive_epoch_outputs", create_archive_epoch_outputs_node())
        update_rag_index_node_func = create_update_rag_index_node(summarizer_llm, embeddings_model)
        workflow.add_node("update_rag_index", update_rag_index_node_func)
        workflow.add_node("metrics", create_metrics_node(llm))
        workflow.add_node("update_personas", create_update_personas_node(llm, params))
        workflow.add_node("progress_assessor", create_progress_assessor_node(llm))
        workflow.add_node("reframe_and_decompose", create_reframe_and_decompose_node(llm))
        workflow.add_node("critique", create_critique_node(llm))
        workflow.add_node("update_prompts", create_update_agent_prompts_node(llm))
        
        async def final_rag_builder(state: GraphState):
            return await update_rag_index_node_func(state, end_of_run=True)

        workflow.add_node("build_final_rag_index", final_rag_builder)


        await log_stream.put("--- Connecting Graph Nodes ---")
        
        workflow.set_entry_point("epoch_gateway")
        await log_stream.put("LOG: Entry point set to 'epoch_gateway'.")
        
        first_layer_nodes = [f"agent_0_{j}" for j in range(len(all_layers_prompts[0]))]
        for node in first_layer_nodes:
            workflow.add_edge("epoch_gateway", node)
            await log_stream.put(f"CONNECT: epoch_gateway -> {node}")

        for i in range(cot_trace_depth - 1):
            current_layer_nodes = [f"agent_{i}_{j}" for j in range(len(all_layers_prompts[i]))]
            next_layer_nodes = [f"agent_{i+1}_{k}" for k in range(len(all_layers_prompts[i+1]))]
            for current_node in current_layer_nodes:
                for next_node in next_layer_nodes:
                    workflow.add_edge(current_node, next_node)
                    await log_stream.put(f"CONNECT: {current_node} -> {next_node}")
        
        last_layer_idx = cot_trace_depth - 1
        last_layer_nodes = [f"agent_{last_layer_idx}_{j}" for j in range(len(all_layers_prompts[last_layer_idx]))]
        for node in last_layer_nodes:
            workflow.add_edge(node, "synthesis")
            await log_stream.put(f"CONNECT: {node} -> synthesis")

        def code_vs_text_gateway(state: GraphState):
            if state.get("is_code_request"):
                log_stream.put_nowait("LOG: Code request detected. Ending graph after synthesis.")
                return "end"
            else:
                log_stream.put_nowait("LOG: Non-code request. Proceeding to archiving.")
                return "archive"

        workflow.add_conditional_edges(
            "synthesis",
            code_vs_text_gateway,
            {
                "end": END,
                "archive": "archive_epoch_outputs"
            }
        )
        await log_stream.put("CONNECT: synthesis -> code_vs_text_gateway (conditional)")

        async def assess_progress_and_decide_path(state: GraphState):
            if state["epoch"] >= state["max_epochs"]:
                await log_stream.put(f"LOG: Final epoch ({state['epoch']}) finished. Proceeding to final RAG indexing before chat.")
                return "build_final_rag_index"
            
            if state.get("significant_progress_made"):
                await log_stream.put(f"LOG: Epoch {state['epoch']} shows significant progress. Re-framing the problem.")
                return "reframe"
            else:
                await log_stream.put(f"LOG: Epoch {state['epoch']} shows no significant progress. Proceeding with standard critique.")
                return "critique"

        workflow.add_conditional_edges(
            "progress_assessor",
            assess_progress_and_decide_path,
            {
                "reframe": "reframe_and_decompose",
                "critique": "critique",
                "build_final_rag_index": "build_final_rag_index"
            }
        )
        await log_stream.put("CONNECT: progress_assessor -> assess_progress_and_decide_path (conditional)")
        
        workflow.add_edge("archive_epoch_outputs", "update_rag_index")
        await log_stream.put("CONNECT: archive_epoch_outputs -> update_rag_index")
        
        workflow.add_edge("update_rag_index", "metrics")
        await log_stream.put("CONNECT: update_rag_index -> metrics")

        workflow.add_edge("metrics", "update_personas")
        await log_stream.put("CONNECT: metrics -> update_personas")
        
        workflow.add_edge("update_personas", "progress_assessor")
        await log_stream.put("CONNECT: update_personas -> progress_assessor")

        workflow.add_edge("critique", "update_prompts")
        await log_stream.put("CONNECT: critique -> update_prompts")

        workflow.add_edge("reframe_and_decompose", "update_prompts")
        await log_stream.put("CONNECT: reframe_and_decompose -> update_prompts")

        workflow.add_edge("update_prompts", "epoch_gateway")
        await log_stream.put("CONNECT: update_prompts -> epoch_gateway (loop)")

        workflow.add_edge("build_final_rag_index", END)
        await log_stream.put("CONNECT: build_final_rag_index -> END")
        
        graph = workflow.compile()
        await log_stream.put("Graph compiled successfully.") 
        
        ascii_art = graph.get_graph().draw_ascii()
        await log_stream.put(ascii_art)
        
        session_id = str(uuid.uuid4())
        print(session_id)
        initial_state = {
            "session_id": session_id,
            "chat_history": [],
            "original_request": user_prompt,
            "current_problem": user_prompt,
            "decomposed_problems": decomposed_problems_map,
            "layers": [], "critiques": {}, "epoch": 0,
            "max_epochs": int(params["num_epochs"]),
            "params": params, "all_layers_prompts": all_layers_prompts,
            "agent_outputs": {}, "memory": {}, "final_solution": None,
            "perplexity_history": [],
            "significant_progress_made": False,
            "raptor_index": None,
            "all_rag_documents": [],
            "academic_papers": None,
            "critique_prompt": INITIAL_GLOBAL_CRITIQUE_PROMPT_TEMPLATE,
            "individual_critique_prompt": INITIAL_INDIVIDUAL_CRITIQUE_PROMPT_TEMPLATE,
            "assessor_prompt": INITIAL_ASSESSOR_PROMPT_TEMPLATE,
            "is_code_request": is_code,
            "summarizer_llm": summarizer_llm,
            "embeddings_model": embeddings_model
        }
        initial_state["llm"] = llm # Store the llm instance for the chat
        sessions[session_id] = initial_state
        
        await log_stream.put(f"--- Starting Execution (Epochs: {params['num_epochs']}) ---")
        
        async for output in graph.astream(initial_state, {'recursion_limit': int(params["num_epochs"]) * 1000}):
            for node_name, node_output in output.items():
                await log_stream.put(f"--- Node Finished Processing: {node_name} ---")
                if node_output is not None:
                    current_session_state = sessions[session_id]
                    for key, value in node_output.items():
                        if key in ['agent_outputs', 'memory']:
                            if key in current_session_state and isinstance(current_session_state[key], dict):
                                current_session_state[key].update(value)
                            else:
                                current_session_state[key] = value
                        else:
                            current_session_state[key] = value
                    sessions[session_id] = current_session_state
                    final_state_value = current_session_state
        
    

        if final_state_value.get("is_code_request"):
            final_code_solution = final_state_value.get("final_solution", {})
            await log_stream.put(f"--- 💻 Code Generation Finished. Returning final code. ---")
            return JSONResponse(content={
                "message": "Code generation complete.",
                "code_solution": final_code_solution.get("proposed_solution", "# No code generated."),
                "reasoning": final_code_solution.get("reasoning", "No reasoning provided.")
            })
        else:
            await log_stream.put("--- Agent Execution Finished. Pausing for User Chat. ---")

            return JSONResponse(content={
                "message": "Chat is now active.",
                "session_id": session_id
            })

    except Exception as e:
        error_message = f"An error occurred during graph execution: {e}"
        await log_stream.put(error_message)
        await log_stream.put(traceback.format_exc())
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)


@app.post("/chat")
async def chat_with_index(payload: dict = Body(...)):
    message = payload.get("message")
    session_id = payload.get("session_id") 

    print("Sesion keys: ", sessions.keys())
    print("Session ID: ", session_id)
 
    if not session_id or session_id not in list(sessions.keys()):
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)

    state = sessions[session_id]

    raptor_index = state.get("raptor_index")
    llm = state["llm"]

    if not raptor_index:
        return JSONResponse(content={"error": "RAG index not found for this session"}, status_code=500)

    async def stream_response():
        try:
            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, message, k=10)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

            chat_chain = get_rag_chat_chain(llm)
            full_response = ""
            async for chunk in chat_chain.astream({"context": context, "question": message}):
                content = chunk.content if hasattr(chunk, 'content') else chunk
                yield content
                full_response += content

            state["chat_history"].append({"role": "user", "content": message})
            state["chat_history"].append({"role": "ai", "content": full_response})

        except Exception as e:

            print(f"Error during chat streaming: {e}")
            yield f"Error: Could not generate response. {e}"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.post("/diagnostic_chat")
async def diagnostic_chat_with_index(payload: dict = Body(...)):
    message = payload.get("message")
    session_id = payload.get("session_id")
    message = payload.get("message")

    print("Sesion keys: ", sessions.keys())
    print("Session ID: ", session_id)

    if not session_id or session_id not in list(sessions.keys()):
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)

    print("Entering diagnostic_chat_with_index")

    state = sessions[session_id]
    raptor_index = state.get("raptor_index")

    if not raptor_index:
        async def stream_error():
            yield "The RAG index for this session is not yet available. Please wait for the first epoch to complete."
        return StreamingResponse(stream_error(), media_type="text/event-stream")
        
    async def stream_response():
        try:
            query = message.strip()[5:]
            await log_stream.put(f"--- [DIAGNOSTIC] Raw RAG query received: '{query}' ---")
                
            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, query, k=10)
                
            if not retrieved_docs:
                yield "No relevant documents found in the RAPTOR index for that query."
                return

            yield "--- Top Relevant Documents (Raw Retrieval) ---\n\n"
            for i, doc in enumerate(retrieved_docs):
                    content_preview = doc.page_content.replace('\n', ' ').strip()
                    metadata_str = json.dumps(doc.metadata)
                    response_chunk = (
                        f"DOCUMENT #{i+1}\n"
                        f"-----------------\n"
                        f"METADATA: {metadata_str}\n"
                        f"CONTENT: {content_preview}...\n\n"
                    )
                    yield response_chunk

        except Exception as e:
            print(f"Error during diagnostic chat streaming: {e}")
            yield f"Error: Could not generate response. {e}"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.post("/harvest")
async def harvest_session(payload: dict = Body(...)):


    if not payload.get("session_id") or payload.get("session_id") not in list(sessions.keys()):
        return JSONResponse(content={"error": "Invalid request"}, status_code=404)

    session =  sessions.get(payload.get("session_id"))

    if not session:
        return JSONResponse(content={"error": "Invalid request"}, status_code=404)

    try:
        await log_stream.put("--- [HARVEST] Initiating Final Harvest Process ---")
        state = session 
        chat_history = session["chat_history"]
        llm = session["llm"]
        summarizer_llm = session["summarizer_llm"]
        embeddings_model = session["embeddings_model"]
        params = session["params"]

        chat_docs = []
        if chat_history:
            for i, turn in enumerate(chat_history):
                 if turn['role'] == 'ai':
                    user_turn = chat_history[i-1]
                    content = f"User Question: {user_turn['content']}\n\nAI Answer: {turn['content']}"
                    chat_docs.append(Document(page_content=content, metadata={"source": "chat_session", "turn": i//2}))
            await log_stream.put(f"LOG: Converted {len(chat_history)} chat turns into {len(chat_docs)} documents.")
            state["all_rag_documents"].extend(chat_docs)
            await log_stream.put(f"LOG: Added chat documents. Total RAG documents now: {len(state['all_rag_documents'])}.")
            
            await log_stream.put("--- [RAG PASS] Re-building Final RAPTOR Index with Chat History ---")
            update_rag_node = create_update_rag_index_node(summarizer_llm, embeddings_model)
            update_result = await update_rag_node(state, end_of_run=True)
            state.update(update_result)

        num_questions = int(params.get('num_questions', 25))
        final_harvest_node = create_final_harvest_node(llm, summarizer_llm, num_questions)
        final_harvest_result = await final_harvest_node(state)
        state.update(final_harvest_result)

        academic_papers = state.get("academic_papers", {})
        session_id = state.get("session_id", "")

        if academic_papers:
            final_reports[session_id] = academic_papers
            await log_stream.put(f"SUCCESS: Final report with {len(academic_papers)} papers created.")
        else:
            await log_stream.put("WARNING: No academic papers were generated in the final harvest.")


        return JSONResponse(content={
            "message": "Harvest complete.",
        })

    except Exception as e:
        error_message = f"An error occurred during harvest: {e}"
        await log_stream.put(error_message)
        await log_stream.put(traceback.format_exc())
        return JSONResponse(content={"message": error_message, "traceback": traceback.format_exc()}, status_code=500)


@app.get('/stream_log')
async def stream_log(request: Request):

    async def event_generator():
        while True:
            if await request.is_disconnected():
                print("Client disconnected from log stream.")
                break
            try:
                log_message = await asyncio.wait_for(log_stream.get(), timeout=1.0)
                yield f"data: {log_message}\n\n"
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in stream: {e}")
                break

    return EventSourceResponse(event_generator())


@app.get("/download_report/{session_id}")
async def download_report(session_id: str):

    print(final_reports.keys())
    print(session_id)
    papers = final_reports.get(session_id, {})

    if not papers:
        return JSONResponse(content={"error": "Report not found or expired."}, status_code=404)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, (question, content) in enumerate(papers.items()):
            safe_question = re.sub(r'[^\w\s-]', '', question).strip().replace(' ', '_')
            filename = f"paper_{i+1}_{safe_question[:50]}.md"
            zip_file.writestr(filename, content)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=NOA_Report_{session_id}.zip"}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)