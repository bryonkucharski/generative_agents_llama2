"""
Author: Bryon Kucharski (bryonkucharski@gmail.com)

File: run_llama2_prompt.py
Description: Defines all run llama prompt functions. These functions directly
interface with the safe_generate_response function.
"""
import re
import datetime
import sys
import ast
import json

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.llama2_structure import safe_generate_response
from utils import *

import logging 
import sys
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logging.basicConfig(handlers=[sh], )
logger.setLevel(logging.INFO)


def levenshteinDistance(s1, s2):
    #https://stackoverflow.com/questions/2460177/edit-distance-in-python

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_random_alphanumeric(i=6, j=6): 
  """
  Returns a random alpha numeric strength that has the length of somewhere
  between i and j. 

  INPUT: 
    i: min_range for the length
    j: max_range for the length
  OUTPUT: 
    an alpha numeric str with the length of somewhere between i and j.
  """
  k = random.randint(i, j)
  x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
  return x

def get_templates_and_params(folder_name):

  with open(f'persona/prompt_template/llama2/{folder_name}/generation_params.json') as f:
    params = json.load(f)
  
  system_message_file = f"persona/prompt_template/llama2/{folder_name}/system_message.txt"
  user_message_file = f"persona/prompt_template/llama2/{folder_name}/user_message.txt"
  completion_start_file = f"persona/prompt_template/llama2/{folder_name}/completion_message_start.txt"

  with open(system_message_file, "r") as f:
    system_message_template = f.read()

  with open(user_message_file, "r") as f:
    user_message_template = f.read()

  with open(completion_start_file, "r") as f:
    completion_start_template = f.read()

  return params, system_message_template, user_message_template, completion_start_template 

def get_completion_string(messages):
  for message in messages:
      if message['role'] == 'completion':
          return message['content']


def run_llama2_prompt_wake_up_hour(persona, test_input=None, verbose=False): 
  """
  Given the persona, returns an integer that indicates the hour when the 
  persona wakes up.  

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    integer for the wake up hour.
  """
  def __func_clean_up(llm_response,messages):
    completion_start = get_completion_string(messages)
    hour = llm_response.replace(completion_start,"")
    cr = int(hour.strip().lower().split("am")[0])
    return cr
  
  def __func_validate(llm_response,messages): 
    try: __func_clean_up(llm_response,messages)
    except: return False
    return True

  def get_fail_safe(): 
    fs = 8
    return fs

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('wake_up_hour')

  persona_iss = persona.scratch.get_str_iss()
  persona_lifestyle = persona.scratch.get_str_lifestyle()
  persona_name = persona.scratch.get_str_firstname()

  system_message = system_message_template
  user_message = user_message_template.format(
    persona_iss = persona_iss,
    lifestyle = persona_lifestyle
  )
  
  completion_start = completion_start_template.format(
    persona_name = persona_name
  )

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  fail_safe = get_fail_safe()
  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )



  if debug or verbose: 
    print("---"*10)
    print("> Prompt For Wakeup Hour: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)
    
  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_daily_plan(persona, wake_up_hour, test_input=None, verbose=False):

  """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form: 
  'wake up and complete the morning routine at 6:00 am', 
  'eat breakfast at 7:00 am',.. 
  Note that the actions come without a period. 

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    a list of daily actions in broad strokes.
  """
  def __func_clean_up(llm_response,messages):
    return ast.literal_eval(llm_response)
    
  def __func_validate(llm_response,messages): 
    try: __func_clean_up(llm_response, messages)
    except: 
      return False
    return True

  def get_fail_safe(): 
    fs = ['wake up and complete the morning routine at 6:00 am', 
          'eat breakfast at 7:00 am', 
          'read a book from 8:00 am to 12:00 pm', 
          'have lunch at 12:00 pm', 
          'take a nap from 1:00 pm to 4:00 pm', 
          'relax and watch TV from 7:00 pm to 8:00 pm', 
          'go to bed at 11:00 pm'] 
    return fs

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('daily_plan')

  persona_iss = persona.scratch.get_str_iss()
  persona_lifestyle = persona.scratch.get_str_lifestyle()
  persona_curr_date = persona.scratch.get_str_curr_date_str()
  persona_first_name = persona.scratch.get_str_firstname()
  wakeup_hour = f"{str(wake_up_hour)}:00 am"
  
  system_message = system_message_template
  
  user_message = user_message_template.format(
    persona_iss = persona_iss,
    lifestyle = persona_lifestyle,
    current_date = persona_curr_date,
    persona_name = persona_first_name,
  )

  completion_start = completion_start_template.format(
    wakeup_hour = wakeup_hour
  )

  fail_safe = get_fail_safe()

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  if debug or verbose: 
    print("---"*10)
    print("> Prompt For Daily Schedule: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)
    
  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_generate_hourly_schedule(
        persona, 
        curr_hour_str,
        p_f_ds_hourly_org, 
        hour_str,
        intermission2=None,
        test_input=None, 
        verbose=False
    ): 
 
  def __func_clean_up(llm_response, messages):

    completion_start = get_completion_string(messages)
    generated_text = llm_response.replace(completion_start,"")
    schedule_item = ast.literal_eval('["' + generated_text)[0]
    return schedule_item

  def __func_validate(llm_response, messages): 
    try: __func_clean_up(llm_response, messages)
    except: return False
    return True

  def get_fail_safe(): 
    fs = "asleep"
    return fs

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('hourly_schedule')

  persona_curr_date = persona.scratch.get_str_curr_date_str()
  persona_iss = persona.scratch.get_str_iss()
  persona_name = persona.scratch.get_str_firstname()
  persona_daily_req = persona.scratch.daily_req

  schedule_format = ""
  for i in hour_str: 
    schedule_format += f"[{persona_curr_date} -- {i}]"
    schedule_format += f" Activity: <START> [Fill in] <END>\n"
  schedule_format = schedule_format[:-1]

  intermission_str = f"Here the originally intended hourly breakdown of {persona_name}'s schedule today: \n\n"
  for count, i in enumerate(persona_daily_req): 
    intermission_str += f"{str(count+1)}) {i}\n"
  intermission_str += '\n'
  
  
  intermission2 = ''
  if intermission2: 
    intermission2 = f"\n{intermission2}"

  prior_schedule = ""
  if p_f_ds_hourly_org: 
    prior_schedule = "\n"
    for count, i in enumerate(p_f_ds_hourly_org): 
        prior_schedule += completion_start_template.format(
            current_date = persona_curr_date,
            current_hour = hour_str[count],
            persona_name = persona_name
        )
        prior_schedule += f'{i}"]\n'


  system_message= system_message_template.format(
    schedule_format = schedule_format
  )

  completion_start = completion_start_template.format(
    current_date = persona_curr_date,
    current_hour = curr_hour_str,
    persona_name = persona_name
  )

  user_message = user_message_template.format(
    persona_iss = persona_iss,
    prior_schedule = prior_schedule,
    intermission_str = intermission_str
  )

  fail_safe = get_fail_safe()
  
  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  
  if debug or verbose: 
    print("---"*10)
    print("> Prompt For Hourly Schedule: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)
    
  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_task_decomp(persona, 
                               task, 
                               previous_generated_decomp_list, 
                               test_input=None, 
                               verbose=False): 

  def __func_clean_up(llm_response, messages):
    completion_start = get_completion_string(messages)
    subtask = llm_response.replace(completion_start,"")

    return subtask
    
  

  def __func_validate(llm_response, messages):  
      try:
        __func_clean_up(llm_response,messages)
      except: 
        return False
      return True

  def get_fail_safe(): 
    fs = "unknown"
    return fs


  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('task_decomp')

  curr_task = str([task])
  subtask_list_str = ''
  for prev_task in previous_generated_decomp_list:
    subtask_list_str += "Subtask: " + str([f"While {task}, for the next 5 minutes, {prev_task}"]) + "\n"


  fail_safe = get_fail_safe()

  persona_iss = persona.scratch.get_str_iss()

  system_message = system_message_template

  user_message = user_message_template.format(
    persona_iss = persona_iss,
    current_task = curr_task,
    subtask_list = subtask_list_str
  )

  completion_start = completion_start_template.format(
    current_task = task
  )

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  if debug or verbose: 
    print("---"*10)
    print("> Prompt For Decomp: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)
    
  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_action_sector(action_description, 
                                persona, 
                                maze, 
                                test_input=None, 
                                verbose=False):



  def __func_clean_up(llm_response, messages):
    try:
      #attempt to clean some common errors
      completion_start = get_completion_string(messages)
      llm_response = llm_response.replace(completion_start,"")
      llm_response = llm_response.split("]")[0] + "]"
      llm_response = llm_response.replace(".",'')
      llm_response = '[\"' + llm_response
      action_sector = ast.literal_eval(llm_response)[0]
    except: 
      import pdb;pdb.set_trace()
    return action_sector

  def __func_validate(llm_response, messages):  
    #no idea why this happens honestly
    try:
      llm_response_list = __func_clean_up(llm_response, messages)
    except:
      return False
    if len(llm_response_list) < 1: 
      return False
    return True
  
  def get_fail_safe(): 
    fs = ("kitchen")
    return fs


  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('action_location_sector')

  act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"

  persona_name = persona.scratch.get_str_name()
  persona_living_area = persona.scratch.living_area.split(":")[1]

  living_accessible_sector_arenas = persona.s_mem.get_str_accessible_sector_arenas(
    f"{act_world}:{persona_living_area}"
  )
  living_accessible_sector_arenas = json.dumps(living_accessible_sector_arenas.split(","))

  current_location = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
  current_location_accessible_sector_arenas = persona.s_mem.get_str_accessible_sector_arenas(
    f"{act_world}:{current_location}"
  )

  current_location_accessible_sector_arenas = json.dumps(current_location_accessible_sector_arenas.split(","))

  accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
  curr = accessible_sector_str.split(", ")
  fin_accessible_sectors = []
  for i in curr: 
    if "'s house" in i: 
      if persona.scratch.last_name in i: 
        fin_accessible_sectors += [i]
    else: 
      fin_accessible_sectors += [i]

  accessible_sector_str = json.dumps(fin_accessible_sectors)

  action_description_1 = action_description
  action_description_2 = action_description
  if "(" in action_description: 
    action_description_1 = action_description.split("(")[0].strip()
    action_description_2 = action_description.split("(")[-1][:-1]



  system_message = system_message_template

  user_message = user_message_template.format(
    persona_name = persona_name,
    persona_living_area = persona_living_area,
    living_accessible_sector_arenas = living_accessible_sector_arenas,
    current_location = current_location,
    current_location_accessible_sector_arenas = current_location_accessible_sector_arenas,
    area_options = accessible_sector_str
  )

  completion_start = completion_start_template.format(
    persona_name = persona_name,
    persona_action1 = action_description_1,
    persona_action2 = action_description_2,
  )


  fail_safe = get_fail_safe()

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  y = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
  x = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(y).split(",")]
  if output not in x: 
    # output = random.choice(x)
    output = persona.scratch.living_area.split(":")[1]

  print ("DEBUG", random.choice(x), "------", output)

  if debug or verbose: 
    print("---"*10)
    print("> Prompt For Action Sector: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)
    
  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_action_arena(action_description, 
                                persona, 
                                maze, act_world, act_sector,
                                test_input=None, 
                                verbose=False):

  def __func_clean_up(llm_response, messages):
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    cleaned_response = llm_response.split("]")[0].replace("'","").replace('"',"")
    return cleaned_response

  def __func_validate(llm_response, messages):  
    try: 
      llm_response = __func_clean_up(llm_response,messages)
    except: return False
    return True 

  
  def get_fail_safe(): 
    fs = ("kitchen")
    return fs

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('action_location_arena')
  
  persona_name = persona.scratch.get_str_name()
  current_tile = persona.scratch.curr_tile
  current_sector = maze.access_tile(current_tile)["sector"]
  current_arena = maze.access_tile(current_tile)["arena"]

  target_sector = act_sector

  accessible_target_arenas = persona.s_mem.get_str_accessible_sector_arenas(
    f"{act_world}:{act_sector}"
  )

  curr = accessible_target_arenas.split(", ")
  fin_accessible_arenas = []
  for i in curr: 
    if "'s room" in i: 
      if persona.scratch.last_name in i: 
        fin_accessible_arenas += [i]
    else: 
      fin_accessible_arenas += [i]
  accessible_target_arenas = str(fin_accessible_arenas)

  action_description1 = action_description
  action_description2 = action_description
  if "(" in action_description: 
    action_description1 = action_description1.split("(")[0].strip()
    action_description2 = action_description2.split("(")[-1][:-1]

  fail_safe = get_fail_safe()

  system_message = system_message_template
  
  user_message = user_message_template.format(
    persona_name = persona_name,
    persona_current_arena = current_arena,
    persona_current_sector = current_sector,
    target_sector = target_sector,
    accessible_target_arenas = accessible_target_arenas
  )

  completion_start = completion_start_template.format(
    persona_action_description = action_description1,
    persona_name = persona_name,
    persona_target_sector = target_sector
  )

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  #try to avoid small issues with grammer. Sometimes the LLM adds or misses punctuation
  distances = []
  if output not in fin_accessible_arenas:
    for arena in fin_accessible_arenas:
      distances.append(levenshteinDistance(output,arena))

    closest_arena_idx = distances.index(min(distances))
    closest_arena = fin_accessible_arenas[closest_arena_idx]

    print("!!Warning: Output not in accesible arenas. LLM Output: {}, Closest Arena: {}".format(closest_arena,output))
    output = closest_arena

  
  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Action Location Object: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_action_game_object(action_description, 
                                      persona, 
                                      maze,
                                      temp_address,
                                      test_input=None, 
                                      verbose=False): 

  def __func_clean_up(llm_response, messages): 
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    llm_response = '[\"' + llm_response
    action_object = ast.literal_eval(llm_response)[0]
    return action_object

  def __func_validate(llm_response, messages): 
   try:
      __func_clean_up(llm_response, messages)
      return True
   except:
    return False
   
  def get_fail_safe(): 
    fs = ("bed")
    return fs

  
  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('action_object')
  

  fail_safe = get_fail_safe()

  if "(" in action_description: 
      action_description = action_description.split("(")[-1][:-1]

  current_objects = persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")

  system_message = system_message_template
  
  user_message = user_message_template.format(
    current_activity = action_description,
    available_objects = current_objects
  )

  completion_start = completion_start_template

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  x = [i.strip() for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(",")]
  if output not in x: 
    output = random.choice(x)

  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Action Game Object: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_emoji_generation(action_description, persona, verbose=False): 
 
  def __func_clean_up(llm_response,  messages):
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    cr = llm_response.strip()
    if len(cr) > 3:
      cr = cr[:3]
    return cr

  def __func_validate(llm_response, messages): 
    try: 
      __func_clean_up(llm_response, messages)
      if len(llm_response) == 0: 
        return False
    except: return False
    return True 

  def get_fail_safe(): 
    fs = "😋"
    return fs


  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('action_emoji_generation')

  if "(" in action_description: 
      action_description = action_description.split("(")[-1][:-1]


  system_message = system_message_template
  
  user_message = user_message_template.format(
    current_activity = action_description
  )

  completion_start = completion_start_template

  fail_safe = get_fail_safe()
  messages = [
    {"role": "system","content": system_message},
    {"role": "user","content": user_message},
    {"role": "completion","content": completion_start}
  ]
  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Action Emoji Generation: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)


def run_llama2_prompt_event_triple(action_description, persona, verbose=False): 
  
  
  def __func_clean_up(llm_response, messages): 
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    llm_response = completion_start.replace("Output: ","") + llm_response
    llm_response_list = ast.literal_eval(llm_response)
    return llm_response_list

  def __func_validate(llm_response, messages): 
    try: 
      llm_response = __func_clean_up(llm_response,messages)
      if len(llm_response) != 3: 
        print('!!Warning: More than three triplets generated: {}'.format(llm_response))
    except: return False
    return True 

  def get_fail_safe(persona): 
    fs = [persona.name, "is", "idle"]
    return fs

  persona_name = persona.name
  if "(" in action_description: 
      action_description = action_description.split("(")[-1].split(")")[0]
  
  
  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('event_triplet')

  system_message = system_message_template

  user_message = user_message_template.format(
    persona_name = persona_name,
    current_activity = action_description
  )

  completion_start = completion_start_template.format(
    persona_name = persona_name
  )

  fail_safe = get_fail_safe(persona) 

  messages = [
    {"role": "system","content": system_message},
    {"role": "user","content": user_message},
    {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  output = (output[0], output[1], output[2])


  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Action Event Triplet: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)



  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False): 
  
  
  def __func_clean_up(llm_response, messages): 
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    llm_response =  completion_start.split(": ")[1] + llm_response
    llm_response_list = ast.literal_eval(llm_response)
    return llm_response_list[0]

  def __func_validate(llm_response, messages): 
    try: 
      llm_response = __func_clean_up(llm_response,messages)
    except: return False
    return True 

  def get_fail_safe(act_game_object): 
    fs = f"{act_game_object} is idle"
    return fs
                  
  
  
  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('object_event_state')

  game_object = act_game_object
  persona_name = persona.name
  action_description = act_desp

  system_message = system_message_template

  user_message = user_message_template.format(
    object_name = game_object,
    persona_name = persona_name,
    action_description=action_description
  )

  completion_start = completion_start_template.format(
    object_name = game_object
  )

  fail_safe = get_fail_safe(act_game_object) 

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]
  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )



  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Action Object Description: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)



  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False): 
  def __func_clean_up(llm_response, messages): 
    
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    llm_response = completion_start.replace("Output: ","") + llm_response
    llm_response_list = ast.literal_eval(llm_response)
    return llm_response_list

  def __func_validate(llm_response, messages): 
    try: 
      llm_response = __func_clean_up(llm_response,messages)
      if len(llm_response) != 3: 
        return False
    except: return False
    return True 

  def get_fail_safe(act_game_object): 
    fs = [act_game_object, "is", "idle"]
    return fs

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params('event_triplet')

  act_obj_desc = act_obj_desc.replace(act_game_object, "").replace("is",'').strip()

  system_message = system_message_template

  user_message = user_message_template.format(
    persona_name = act_game_object,
    current_activity = act_obj_desc
  )

  completion_start = completion_start_template.format(
    persona_name = act_game_object
  )

  fail_safe = get_fail_safe(act_game_object)
  messages = [
    {"role": "system","content": system_message},
    {"role": "user","content": user_message},
    {"role": "completion","content": completion_start}
  ]
  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  output = (output[0], output[1], output[2])

  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Action Event Triplet: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_poignancy(poignancy_type,persona, description, test_input=None, verbose=False): 

  def __func_clean_up(llm_response, messages):
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    llm_response = ast.literal_eval('["' + llm_response)[0]
    return int(llm_response)

  def __func_validate(llm_response, messages):
    try: 
      __func_clean_up(llm_response, messages)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return 4

  if poignancy_type == "event" or poignancy_type == "thought" : 
    poignancy_type = 'event_poignancy'

  elif poignancy_type == "chat": 
    poignancy_type = 'chat_poignancy'

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params(poignancy_type)

  system_message = system_message_template

  user_message = user_message_template.format(
    description = description
  )

  completion_start = completion_start_template
  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]
  fail_safe = get_fail_safe() ########
  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )

  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Poignancy: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False): 
  
  def __func_clean_up(llm_response, messages):
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    llm_response = "1) " + llm_response.strip()
    ret = []
    for i in llm_response.split("\n"): 
      ret += [i.split(") ")[-1]]
    return ret

  def __func_validate(llm_response, messages):
    try: 
      __func_clean_up(llm_response, messages)
      return True
    except:
      return False 

  def get_fail_safe(n): 
    return ["Who am I"] * n

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params("reflect_focal_pt")

  system_message = system_message_template.format(
    number_to_generate = n
  )

  user_message = user_message_template.format(
    statements = statements
  )
  completion_start = completion_start_template
  messages = [
        {"role": "system","content": system_message},
        {"role": "user","content": user_message},
        {"role": "completion","content": completion_start}
  ]
  fail_safe = get_fail_safe(n) 
  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )
  
  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Focal Pt: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False): 

  def __func_clean_up(llm_response, messages):
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    llm_response = "1. " + llm_response.strip()
    ret = dict()
    for i in llm_response.split("\n"): 
      row = i.split(". ")[-1]
      thought = row.split("(because of ")[0].strip()
      evi_raw = row.split("(because of ")[1].split(")")[0].strip()
      evi_raw = re.findall(r'\d+', evi_raw)
      evi_raw = [int(i.strip()) for i in evi_raw]
      ret[thought] = evi_raw
    return ret

  def __func_validate(llm_response, messages):
    try: 
      __func_clean_up(llm_response, messages)
      return True
    except:
      import pdb;pdb.set_trace()
      return False 

  def get_fail_safe(n): 
    return ["I am hungry"] * n


  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params("reflect_insight_and_evidence")

  system_message = system_message_template.format(
    number_to_generate = n
  )

  user_message = user_message_template.format(
    statements = statements
  )
  completion_start = completion_start_template.format(
    number_to_generate = n
  )

  fail_safe = get_fail_safe(n) 
  messages = [
        {"role": "system","content": system_message},
        {"role": "user","content": user_message},
        {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )
  
  if debug or verbose:  
    print("---"*10)
    print("> Prompt For insight and guidance ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_decide_to_talk(persona, target_persona, retrieved,test_input=None, verbose=False): 

  def __func_validate(llm_response, system_message=None, user_message=None, completion_start=None): 
    try: 
      completion_start = get_completion_string(messages)
      llm_response = llm_response.replace(completion_start,"")
      if llm_response.split("\"")[0].strip().lower() in ["yes", "no"]: 
        return True
      return False     
    except:
      return False 

  def __func_clean_up(llm_response, messages): 
    return llm_response.split("\"")[0].strip().lower()

  def get_fail_safe(): 
    fs = "yes"
    return fs


  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params("chat_decide_to_talk")

  last_chat = persona.a_mem.get_last_chat(target_persona.name)
  last_chatted_time = ""
  last_chat_about = ""
  if last_chat: 
    last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
    last_chat_about = last_chat.description

  context = ""
  for c_node in retrieved["events"]: 
    curr_desc = c_node.description.split(" ")
    curr_desc[2:3] = ["was"]
    curr_desc = " ".join(curr_desc)
    context +=  f"{curr_desc}. "
  context += "\n"
  for c_node in retrieved["thoughts"]: 
    context +=  f"{c_node.description}. "

  curr_time = persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
  init_act_desc = persona.scratch.act_description
  if "(" in init_act_desc: 
    init_act_desc = init_act_desc.split("(")[-1][:-1]
  
  if len(persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
    init_p_desc = f"{persona.name} is already {init_act_desc}"
  elif "waiting" in init_act_desc:
    init_p_desc = f"{persona.name} is {init_act_desc}"
  else: 
    init_p_desc = f"{persona.name} is on the way to {init_act_desc}"

  target_act_desc = target_persona.scratch.act_description
  if "(" in target_act_desc: 
    target_act_desc = target_act_desc.split("(")[-1][:-1]
  
  if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc: 
    target_p_desc = f"{target_persona.name} is already {target_act_desc}"
  elif "waiting" in init_act_desc:
    target_p_desc = f"{persona.name} is {init_act_desc}"
  else: 
    target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"


  if last_chatted_time == '':
    last_chatted_time = 'None'
  
  if last_chat_about == '':
    last_chat_about = 'None'

  system_message = system_message_template

  user_message = user_message_template.format(
    context = context,
    curr_time=curr_time,
    persona_1=persona.name,
    persona_2=target_persona.name,
    last_chat_time=last_chatted_time,
    last_chat_info=last_chat_about,
    persona_1_action=init_p_desc,
    persona_2_action=target_p_desc
  )

  completion_start = completion_start_template
  fail_safe = get_fail_safe()

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )
  
  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Decide To Talk: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None, verbose=False): 
  
  def __func_clean_up(llm_response, messages): 
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    return llm_response.split('"')[0].strip()

  def __func_validate(llm_response, messages): 
    try: 
      __func_clean_up(llm_response, messages )
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "..."

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params("chat_summarize_relationship")

  system_message = system_message_template.format(
    persona_1 = persona.scratch.name,
    persona_2 = target_persona.scratch.name
  )

  user_message = user_message_template.format(
    statements = statements
  )

  completion_start = completion_start_template.format(
    persona_1 = persona.scratch.name,
    persona_2 = target_persona.scratch.name
  )

  fail_safe = get_fail_safe()

  messages = [
      {"role": "system","content": system_message},
      {"role": "user","content": user_message},
      {"role": "completion","content": completion_start}
  ]

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )
  
  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Chat Relationship Summarization: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None, verbose=False): 


  def __func_clean_up(llm_response, messages): 
 
    json_response = json.loads(llm_response.replace("```","").replace("```",""))

    cleaned_dict = dict()
    cleaned = []
    for key, val in json_response.items(): 
      cleaned += [val]
    cleaned_dict["utterance"] = cleaned[0]

    if cleaned[1].strip().lower() == 'yes':
      end = True
    else:
      end= False

    cleaned_dict["end"] = end
    # if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
    #   cleaned_dict["end"] = False

    return cleaned_dict

  def __func_validate(llm_response, messages): 

    try: 
      json_response = json.loads(llm_response.replace("```","").replace("```",""))
      return True
    except:
      return False 

  def get_fail_safe():
    cleaned_dict = dict()
    cleaned_dict["utterance"] = "..."
    cleaned_dict["end"] = False
    return cleaned_dict


  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params("chat_generate_utterance")

  #get memory
  retrieved_str = ""
  for key, vals in retrieved.items(): 
    for v in vals: 
      retrieved_str += f"- {v.description}\n"

  #get current context
  prev_convo_insert = "\n"
  if init_persona.a_mem.seq_chat: 
    for i in init_persona.a_mem.seq_chat: 
      if i.object == target_persona.scratch.name: 
        v1 = int((init_persona.scratch.curr_time - i.created).total_seconds()/60)
        prev_convo_insert += f'{str(v1)} minutes ago, {init_persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
        break

  if prev_convo_insert == "\n": 
    prev_convo_insert = ""
  if init_persona.a_mem.seq_chat: 
    if int((init_persona.scratch.curr_time - init_persona.a_mem.seq_chat[-1].created).total_seconds()/60) > 480: 
      prev_convo_insert = ""

  #get current location
  curr_sector = f"{maze.access_tile(init_persona.scratch.curr_tile)['sector']}"
  curr_arena= f"{maze.access_tile(init_persona.scratch.curr_tile)['arena']}"
  curr_location = f"{curr_arena} in {curr_sector}"

  messages = []
  
  system_message = system_message_template.format(
    persona_1 = init_persona.scratch.name,
    persona_2 = target_persona.scratch.name, 
    persona_1_iss=init_persona.scratch.get_str_iss(),
    persona_1_memory=retrieved_str,
    current_location=curr_location,
    past_context = prev_convo_insert,
    current_context = curr_context,
  )

  completion_start = completion_start_template.format(
    persona_1 = init_persona.scratch.name
  )

  messages.append({"role": "system","content": system_message})

  #get conversation_history
  
  for i in curr_chat:
    if i[0] == init_persona.scratch.name:
      messages.append({"role": "user","content": i[1]})
    else:
      messages.append({"role": "assistant","content": i[1]})


  messages.append({"role": "completion","content": completion_start})

  fail_safe = get_fail_safe()
  

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )



  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Chat Utterance Generation: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  #import pdb;pdb.set_trace()

  return output, [output, output_prompt, params, fail_safe]


def run_llama2_summarize_conversation(persona, conversation, test_input=None, verbose=False): 
  
  def __func_clean_up(llm_response, messages): 
    completion_start = get_completion_string(messages)
    llm_response = llm_response.replace(completion_start,"")
    return llm_response.split('"')[0].strip()

  def __func_validate(llm_response, messages): 
    try: 
      __func_clean_up(llm_response, messages)
      return True
    except:
      return False 

  def get_fail_safe(): 
    return "conversing with a housemate about morning greetings"
  

  params, system_message_template, user_message_template, completion_start_template = get_templates_and_params("chat_summarize_conversation")

  convo_str = ""
  for row in conversation: 
    convo_str += f'{row[0]}: "{row[1]}"\n'

  system_message = system_message_template
  user_message = user_message_template.format(conversation=convo_str)
  completion_start = completion_start_template

  messages = [
    {"role": "system","content": system_message},
    {"role": "user","content": user_message},
    {"role": "completion","content": completion_start}
  ]

  fail_safe = get_fail_safe()

  output_prompt, output = safe_generate_response(
    messages,
    params,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up
  )
  
  if debug or verbose:  
    print("---"*10)
    print("> Prompt For Chat Summarization: ")
    print(output_prompt)
    print("> LLM Response: \n", output)
    print("---"*10)

  return output, [output, output_prompt, params, fail_safe]







  