
from curses.ascii import alt
import json
import os
import cmu_course_api
import re
import pickle
import sys
sys.setrecursionlimit(1000000)
import pandas as pd

import random
random.seed(0)

def format_instructor(instructor: str):
    if ',' in instructor:
        pieces = instructor.split(', ')
        if len(pieces) == 2:
            return f"{pieces[1]} {pieces[0]}"
    return instructor

def elaborate_semester(sem: str):
    alt_names = {
        'Summer One/All 2024': 'Summer One/All 2024 (aka M24, M24 Mini 1 or M24 Full, Summer 2024 Mini 1)',
        'Summer Two 2024': 'Summer Two 2024 (aka M24, M24 Mini 2, Summer 2024 Mini 2)',
        'Spring 2024': 'Spring 2024 (aka S24)',
        'Fall 2023': 'Fall 2023 (aka F23)',
    }
    
    if sem in alt_names:
        return alt_names[sem]
    else:
        return sem

def parse_lectures(lecs):
    instructors = set()
    rooms = set()
    locations = set()
    
    for l in lecs:
        for instructor in l['instructors']:                
            instructors.add(format_instructor(instructor))
        for time in l['times']:
            if time['room'] is not None and time['building'] is not None:
                rooms.add(f"{time['building']} {time['room']}")
            elif time['building'] is not None:
                rooms.add(f"{time['building']}")
            locations.add(time['location'])
            
    return instructors, rooms, locations

def mk_course_doc(cid, c, sem):
    instructors, rooms, locations = parse_lectures(c['lectures'])
    res = \
f"""<start course metadata for {cid} {c['name']}>
Semester: {elaborate_semester(sem)}
Course Name: {c['name']}
Course Number: {cid}
Department: {c['department']}
Number of Units: {int(c['units']) if c['units'] != None else 'N/A'}
Prerequisites: {c['prereqs']}
Instructors: {', '.join(sorted(list(instructors)))}
Rooms: {'; '.join(sorted(list(rooms)))}
Locations: {'; '.join(sorted(list(locations)))}
</end course metadata for {cid} {c['name']}>

<start course description for {cid} {c['name']}>
Semester: {elaborate_semester(sem)}
Course Description: {c['desc']}
</end course description for {cid} {c['name']}>
"""
# Detailed Lectures Information (JSON): 
# ```
# {json.dumps(c['lectures'], indent=4)}
# ```
# """
    return res



def gen_course_questions(cid, c, sem):
    instructors, rooms, locations = parse_lectures(c['lectures'])
    res = [
        {
            "Q": f"Which department offers the course {cid} {c['name']}?",
            "A": f"{c['department']}",
        },
    ]
    
    if c['units'] != None:
        res.append({
            "Q": f"How many units does the course {cid} {c['name']} carry?",
            "A": f"{int(c['units'])}",
        })
        
    if '; '.join(sorted(list(instructors))) != 'Instructor TBA':
        res.append({
            "Q": f"Who taught {cid} {c['name']} in the semester {sem}?",
            "A": f"{'; '.join(sorted(list(instructors)))}",
        })
    
    if c['prereqs'] != None:
        res.append({
            "Q": f"What are the prerequisites for {cid} {c['name']}?",
            "A": f"{c['prereqs']}",
        })

    
    if '; '.join(sorted(list(rooms))) != 'DNM DNM':
        res.append({
            "Q": f"Which room or rooms does the course {cid} {c['name']} take place in {sem}?",
            "A": f"{'; '.join(sorted(list(rooms)))}",
        })

    return res


# Create the if it doesn't exist (this is by copilot)
os.makedirs('../../../collection/courses', exist_ok=True)
questions_acc = []

for (sem_id, sem_name) in [
    ('M1', 'Summer 2024 (Mini 1 and Full Summer)'), 
    ('M2', 'Summer 2024 (Mini 2)'), 
    ('F', 'Fall 2023'), 
    ('S', 'Spring 2024'), 
]:
  
    cache_file_name = f'cache/{sem_id}.json'
    if os.path.exists(cache_file_name):
        with open(cache_file_name, 'r') as f:
            sem_data = json.load(f)
    else:
        sem_data = cmu_course_api.get_course_data(sem_id)
        with open(cache_file_name, 'w') as f:
            json.dump(sem_data, f)
        
    course_ids = list(sem_data['courses'].keys())
    for cid in course_ids:
        course_record = sem_data['courses'][cid]
        doc: str = mk_course_doc(cid, course_record, sem_data['semester'])
        
        # Replace characters that can mess up file path with an underscore (regex by copilot)
        safe_course_name = re.sub(r'[<>:"/\\|?*]', '_', course_record['name'])

        # Use the safe course name in the file path
        with open(f"../../../collection/courses/{sem_name} {cid} {safe_course_name}.txt", 'w') as file:
            file.write(doc)
        
        qs = gen_course_questions(cid, course_record, sem_data['semester'])
        questions_acc.extend(qs)
        

sampled_questions = random.sample(questions_acc, 300)

df = pd.DataFrame(sampled_questions)
df['category'] = 'courses'
print(df)
df.to_json('../../../annotations/coursesQAs.json', indent=2, orient='records')

df = pd.DataFrame(questions_acc)
df['category'] = 'courses'
df.to_json('../../../annotations/coursesQAsFull.json', indent=2, orient='records')