#from textblob import TextBlob
from pathlib import Path
import re
import subprocess
import shlex
import os
import traceback
import random
#import language_tool_python
import sys, os
sys.path.append(os.path.dirname(__file__))
# path = Path(os.path.dirname(__file__)) / "response_generator/419eater"
# sys.path.append(str(path))

tool = language_tool_python.LanguageTool('en-US')

ROOT = Path(os.path.realpath(__file__)).parent

def fix_grammar(text = 'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'):
    return tool.correct(truecase(text))

def clean_paragraph(text):
    text = text.encode("ascii", "ignore").decode()
    text = text.replace("\n", " ")
    text = clean(text) # fix periods
    text = fix_grammar(text)
    return text


#email_dict
# First_name
# Body
# Date
# Reply-To
# Subject
def standardize_email_dict(email_dict, first_contact=True):
    email_dict['Body'] = email_dict['body']
    email_dict['Subject'] = email_dict['subject']
    email_dict['Date'] = email_dict['date']
    email_dict['Reply-To'] = email_dict['from_name']
    email_dict['First_name'] = ''
    email_dict['first_contact'] = first_contact
    return email_dict

equivalents = {"Telephone":"phone",
               "bank account": "your bank account",
               "first_name":"First_name",
                "last_name":"Last_name",
                "address": "residence"}

def standardize_id_dict(id_dict):
    for main_key in equivalents.keys():
        if main_key in id_dict:
            e_key = equivalents[main_key]
            id_dict[e_key] = id_dict[main_key]
    if "address" not in id_dict:
        id_dict["address"] = f"{id_dict['Address']}\n" \
                             f"\t{id_dict['City']}, {id_dict['Country']} {id_dict['Postcode']}"
    id_dict["full name"] = f"{id_dict['First_name']} {id_dict['Last_name']}"
    return id_dict

def get_first_scammer_message(conversation_dict):
    for message in conversation_dict["messages"]:
        if message["author_role"] == "SCAMMER":
            return message

def get_sentences(message, min_length=15):
    sentences = TextBlob(clean(message)).sentences
    sufficiently_long_sentences = [x for x in sentences if len(str(x)) > min_length]
    return sufficiently_long_sentences if sufficiently_long_sentences else sentences

def get_random_scammer_message(conversation_dict):
    start_idx = random.randint(0,len(conversation_dict["messages"]))
    for message in conversation_dict["messages"][start_idx:]:
        if message["author_role"] == "SCAMMER":
            return message
    return get_first_scammer_message(conversation_dict)


## Decorator must be declared before it is called! (duh!)
def error_handler(func):
    def wrapper(self, *args, **kwargs):
        # Do nothing
        if False:
            return func(self, *args, **kwargs)
        else:
            try:
                return func(self, *args, **kwargs) # exits here if everything is working
            except Exception as e:
                traceback.print_exc()
                input("Continue?")
    return wrapper


def run_command(command, return_text=False, shell=False, suppress_err=False):
    """ Requires subprocess, shlex

        command (str): the string of the command that would be put into shell
        return_text (bool): by default, function returns exit code 0/1; this returns the command output text
        shell (bool): use subproces 'shell' command -- NOT TESTED
        suppress_err (bool): Send stderr (red text in PyCharm)
    """
    kwargs = {}
    if suppress_err:
        FNULL = open(os.devnull, 'w')
        kwargs["stderr"] = FNULL

    command = shlex.split(command) if not shell else command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False, **kwargs)
    outputs = []
    while True:
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        if output:
            output = output.strip()
            if isinstance(output, bytes):
                output = output.decode()
            outputs.append(output)
    if not return_text:
        rc = process.poll()
        return rc
    else:
        return "\n".join(outputs)

def create_scenarios():
    all_categories = ["atm_card", "employment", "next_of_kin", "banking", "fake_cheques", "orphans", "business",
                      "government", "refugees", "church_and_charity", "loans", "romance", "commodities", "lottery",
                      "western_union_and_moneygram", "compensation", "military", "widow", "delivery_company", "misc",
                      "dying_people"]
    for category in all_categories:
        if not (Path("scenarios") / category).exists():
            input(f"Didn't find {category}")
            (Path("scenarios") / ("_"+category)).mkdir(parents=True,exist_ok=True)


### GENERIC VERSION
def pick_random_file(path, ext=".txt", open_it=True):
    options = list(filter(lambda x: ext in str(x) and '~' not in str(x), path.glob('*' + ext)))
    if options:
        choice = random.choice(options)
        if open_it:
            return open(choice).read().strip()
        else:
            return str(choice)
    else:
        return ''

def choose_random_line_from_file(path):
    with Path(path).open() as f:
        all_lines = f.readlines()
    return random.choice(all_lines).strip()


def trim_after_keyword(message, regex_end_list=None, prefix_length=0, delete_last_line=False):
    if regex_end_list is None:
        regex_end_list = [
            "sincerely",
            "regards[,\n\r]+",
            "best[,\n\r]+",
            "cheers[,\n\r]+",
            "wishes[,\n\r]+",
            "yours[,\n\r]+",
            "truly[,\n\r]+",
            "cordially[,\n\r]+",
            "respectfully[,\n\r]+"
        ]
    if regex_end_list == "prefix":
        regex_end_list=["[\n ]+Hey[\n,]+",
                         "Hello",
                         "Dear",
                         "[\n ]+Hi[\n, ]+"]
    deletion_made = False
    for end_message_regex in regex_end_list:
        match_obj = re.search(end_message_regex, message[prefix_length:], flags=re.IGNORECASE)
        if match_obj:
            message = message[:match_obj.start(0)+prefix_length]
            deletion_made = True

    # delete the last line - e.g. if a colon is found, delete everything which preceded the colon
    if delete_last_line and deletion_made:
        message = "\n".join(message.split("\n")[:-1])
    return message


import numpy as np
from pathlib import Path
import nltk
from tqdm import tqdm
# UNCOMMENT FIRST TIME RUNNING
#nltk.download('punkt')
import re

fix_spaces = re.compile("([A-Za-z0-9]+)([\.\?\!])([A-Za-z0-9]+)")
fix_spaces.sub(r"\1\2 \3", "This is a sentence. This is too.Bad period there?And here!")


# Top priority
def fix_email_order():
    """ If email contains time metadata, make sure messages are sorted by "UTC" and not local time

    Returns:

    """
    pass


# Second priority
def add_ppi_tokens(email_dict):
    """ Take in BAITER messages, replace PPI with tokens. E.g.,
    Hi <SCAMMER>, My phone number is <PHONE NUMBER>. Thanks! <SCAM BAITER>

    Args:
        email_dict (dict):
    Returns:

    """
    pass




def process_conversations():
    """ Create training pairs (scammer prompt - baiter response)

    Returns:

    """
    path = "/media/data/GitHub/BAITERBOT/data/scam_baiters_json/all_json_data.npy"
    conversations = np.load(path, allow_pickle=True).item()
    training_pairs = []
    prev_message = ""
    for key, conversation in conversations.items():
        for email in conversation["messages"]:
            if prev_message and email["author_role"]=="VICTIM":
                training_pairs.append((clean(prev_message), clean(email["body"])))
                prev_message = "" # ignore when victim emails multiple times
            elif email["author_role"]=="SCAMMER":
                prev_message = email["body"]
    return training_pairs

def sentence_tokenize(text):
    """ Split messages into sentences, e.g. for paraphrasing

    Args:
        text:

    Returns:

    """
    sent_text = nltk.sent_tokenize(text)  # this gives us a list of sentences
    return sent_text

# GRAMMER
## Standarize scammer grammer
def clean(paragraph):
    """

    Args:
        paragraph (str): Correct some scammer grammar issues (e.g. so NLTK tokenizes sentences correctly)

    Returns:

    """
    return fix_spaces.sub(r"\1\2 \3", paragraph.replace(" .", ". "))  # fix period issues




def truecase(text=None):
    """ Scammers sometimes use all caps, might generalize better if there's an easy way to properly case it

    Returns:

    """
    import truecase
    if text is None:
        text = "Dear Mr.Mikhail Stroganov , \n\nThanks very much for your informations to me. Could you kindly forward the picture Of the imposter that is claiming to me, Mohammed Abacha. \n\nMy dear brother, I want you to put me in your shoes for considerations sake.Please, I am trying you for your keen involvements so far and would like to ifnrom you that TO BE FORE WARN IS TO FORE ARM A WORD IS ENOUGH FOR THE WISE!."
    return truecase.get_true_case(text)

def stanford():
    #pip install stanfordnlp
    import stanfordnlp
    stanfordnlp.download('en')
    nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
    doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
    doc.sentences[0].print_dependencies()


pattern = "(([a-z0-9._-]+)@[a-z0-9\._-]+\.[a-z0-9_-]{2,10})"
find_email = re.compile(pattern, re.IGNORECASE)
find_email_bytes = re.compile(pattern.encode(), re.IGNORECASE)



if __name__=='__main__':
    create_scenarios()


