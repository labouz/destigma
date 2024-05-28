import pandas as pd
import time
import google.generativeai as genai
from threading import Lock

rate_limit = 500
rate_limit_period = 60  # seconds

# Global variables for tracking rate limit
request_count = 0
request_lock = Lock()

def get_drug_post(post, retries = 2, api_key = None):
    prompt = f"""
    USER: Carefully read the entire post, paying attention to both explicit mentions and subtle allusions to drug-related topics.
    Consider the context when evaluating the post. Categorize the post into ONE of the following classes:
        * D: The post is primarily about drugs or people who use drugs.
        * ND: The post is not primarily about drugs or people who use drugs.

    Additional Notes:
    * Slang terms and euphemisms for drugs are common. Be vigilant in identifying them.
    * Posts that mention drugs in passing or as part of a larger narrative should be classified based on the overall focus of the post.

    Example Posts and Labels:
    USER: I'm so high right now, I can't even feel my face. This is the best weed I've ever smoked.
    MODEL: D, 'I'm so high right now', 'This is the best weed I've ever smoked'. The author is explicitly mentioning being high and smoking weed.
    
    USER: I hope my junkie sister OD's or disappears out of our lives My sister is an alcoholic junkie who has 2 DUIs under her belt as well as loves taking Xanax and alcohol together and wreaking havoc for our family and even strangers.
    MODEL: D, 'junkie', 'Xanax and alcohol', The author is using the term junkie to describe their sister and mentions specific drugs like Xanax and alcohol.
    
    USER: My mom is going to kick me out. She graciously gave me the choice of getting dropped off in a shelter in either San Diego or the desert area (Palm Springs and surrounding areas). I would choose the desert because that is one of my old stomping grounds. The dope is phenomenal and cheap (3g's for $100) and the homeless population is a majority young people. I can also hustle up $350 and rent a room at a buddy's place. I have a few options that I can look at but I have to figure it out soon.
    MODEL: D, 'dope', '3g's for $100', The author is referring to drugs like heroin as 'dope' and mentions the price and availability of drugs in the desert area.
    
    USER: {post}
    """
    global request_count

    while retries > 0:
        with request_lock:
            if request_count >= rate_limit:
                time.sleep(rate_limit_period)
                request_count = 0
            request_count += 1

        try:
            response = genai.complete(prompt, api_key = api_key)
            return response.choices[0].text
        except Exception as e:
            print(e)
            retries -= 1
            time.sleep(1)
    return None

