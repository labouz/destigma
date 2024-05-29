# the script for labeling a post as durg-related or not
import pandas as pd
import openai
import time
from threading import Lock

rate_limit = 500
tpm_limit = 160000  # Tokens per minute
rate_limit_period = 60  # seconds
retry_wait_time = 5  # seconds between retries

# Global variables for tracking rate limit
request_count = 0
token_count = 0
request_lock = Lock()
token_lock = Lock()

def get_drug_post(post, retries = 2, model = "gpt-3.5-turbo-0125", openai_client=None):
# prompt
    prompt = f"""
    Instructions:

    1. Carefully read the entire post, paying attention to both explicit mentions and subtle allusions to drug-related topics.
    2. Consider the context when evaluating the post.  
    3. Categorize the post into ONE of the following classes:
        * D: The post is primarily about drugs or people who use drugs. 
        * ND: The post is not primarily about drugs or people who use drugs. 

    Additional Notes:

    * Slang terms and euphemisms for drugs are common. Be vigilant in identifying them.
    * Posts that mention drugs in passing or as part of a larger narrative should be classified based on the overall focus of the post.

    """
    example1 = "I'm so high right now, I can't even feel my face. This is the best weed I've ever smoked."
    answer1 = "D"
    example2 = "I hope my junkie sister OD's or disappears out of our lives My sister is an alcoholic junkie who has 2 DUIs under her belt as well as loves taking Xanax and alcohol together and wreaking havoc for our family and even strangers."
    answer2 = "D"
    example3 = "My mom is going to kick me out. She graciously gave me the choice of getting dropped off in a shelter in either San Diego or the desert area (Palm Springs and surrounding areas). I would choose the desert because that is one of my old stomping grounds. The dope is phenomenal and cheap (3g's for $100) and the homeless population is a majority young people. I can also hustle up $350 and rent a room at a buddy's place. I have a few options that I can look at but I have to figure it out soon."
    answer3 = "D"

    global request_count, token_count

    while retries > 0:
        try:
            with request_lock:
                if request_count >= rate_limit:
                    print("Rate limit reached. Pausing for a minute...")
                    time.sleep(rate_limit_period)
                    request_count = 0

            with token_lock:
                if token_count >= tpm_limit:
                    print("TPM limit reached. Pausing for a minute...")
                    time.sleep(rate_limit_period)
                    token_count = 0


            response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": example1

                },
                {
                    "role": "system",
                    "content": answer1
                },
                {
                    "role": "user",
                    "content": example2
                },
                {
                    "role": "system",
                    "content": answer2
                },
                {
                    "role": "user",
                    "content": example3
                },
                {
                    "role": "system",
                    "content": answer3
                },
                {
                    "role": "system",
                    "content": post
                }
            ],
            model=model,
            temperature=0
        )
            with request_lock:
                request_count += 1

            with token_lock:
                token_count += sum([len(prompt), len(response.choices[0].message.content)])

            label = response.choices[0].message.content.lower().strip()
            return label
        except openai.RateLimitError as e:
            print(f"Rate limit error: {e}. Retrying in {retry_wait_time} seconds...")
            time.sleep(retry_wait_time)
            retries -= 1
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(retry_wait_time)

    print("Retrying...")
    return "skipped"

