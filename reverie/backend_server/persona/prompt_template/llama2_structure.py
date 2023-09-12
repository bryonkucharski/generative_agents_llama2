"""
Author: Bryon Kucharski (bryonkucharski@gmail.com)

File: gpt_structure.py
Description: Wrapper functions for calling Llama2 APIs with the HuggingFace Text Generation Server (Sagemaker version)
"""

from sentence_transformers import SentenceTransformer

from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri, HuggingFacePredictor

import json
import random
import time 

HF_ENDPOINT_NAME = 'hf-tgi-meta-llama-Llama-2-13b-chat-hf20-2023-09-10-13-26-58-895'

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message.content}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt

def build_llama2_prompt_with_completion(messages):
    """
    """
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        elif message["role"] == "completion":
            #this is if you want to steer the bots completion in a certain direction.
            prompt =  startPrompt + "".join(conversation) + endPrompt
            prompt += " "+ message["content"]
            return prompt
        else:
            conversation.append(f" [/INST] {message['content']}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt

def llm(messages, parameters):
  
  predictor = HuggingFacePredictor(HF_ENDPOINT_NAME)

  completion_start = ''
  for message in messages:
      if message['role'] == 'completion':
          completion_start = message['content']

  final_prompt = build_llama2_prompt_with_completion(messages)

  payload = {
      "inputs":  final_prompt,
      "parameters": parameters
  }
  response = predictor.predict(payload)

  generated_text = response[0]['generated_text'][len(final_prompt):]
  full_text = completion_start + generated_text

  return final_prompt, full_text
  
def llm_request(messages, parameters): 
  """
  Given a prompt, make a request to LLM server and returns the response. 
  ARGS:
    prompt: a str prompt 
    parameters: optional 
  RETURNS: 
    a str of LLM's response. 
  """
  # temp_sleep()
  try:
    final_prompt, response = llm(
        messages,
        parameters
      )
  except ValueError:
    print("Requested tokens exceed context window")
    ### TODO: Add map-reduce or splitter to handle this error.
    return final_prompt, "LLM ERROR"
  return final_prompt, response

def safe_generate_response(messages,
                           parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 

  for i in range(repeat): 
    final_prompt, curr_gpt_response  = llm_request(
      messages,
      parameter
    )
    if func_validate(curr_gpt_response, messages): 
      return final_prompt, func_clean_up(curr_gpt_response, messages)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  import pdb;pdb.set_trace()
  return 'fail safe', fail_safe_response

def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    # Initialize the SentenceTransformer model
    model = SentenceTransformer(model_name)

    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    # Get the embedding
    embedding = model.encode(text).tolist()

    return embedding

