# the script for labeling a post as containing stigmatizing language or not
import pandas as pd
import time
import openai
from threading import Lock

rate_limit = 600
tpm_limit = 800000  # Tokens per minute
tpd_limit = 80000000  # Tokens per day
rate_limit_period = 60  # seconds
retry_wait_time = 5  # seconds between retries

# Global variables for tracking rate limit
request_count = 0
token_count = 0
daily_token_count = 0
request_lock = Lock()
token_lock = Lock()

def get_destigma(post, explanation, step, retries = 2, model = "gpt-4-turbo-2024-04-09", openai_client=None):
    # explanation in the form of Labeling: Uses the term 'junkies,' a derogatory label, Stereotyping: Portrays people who use drugs as irresponsible and a burden on society, Separation: Explicitly states that people with addiction do not belong in society, Discrimination: Advocates for stripping rights and marking individuals with addiction, suggesting they should be treated differently and excluded from societal opportunities.
    # get string between Labeling: and ,
    if step == 1:
        instruction = "Rewrite this post to remove any and all labeling."
        definition = "Labeling includes the use of derogatory or othering language related to drug use/addiction."
        try:
            explanation = explanation.split('Labeling: ')[1].split(',')[0]
        except:
            explanation = explanation
    else:
        instruction = "Rewrite this post to remove any all instances of stereotyping, insinuations of separation, and/or discriminatory language."
        definition = "Stereotyping reinforces negative generalizations about people who use drugs. Separation creates a divide between people who use drugs and those who don't. Discrimination implies or suggests unfair treatment based on drug use."
        try:
            stereotype = explanation.split('Stereotyping: ')[1].split(',')[0]
            separation = explanation.split('Separation: ')[1].split(',')[0]
            discrimination = explanation.split('Discrimination: ')[1].split(',')[0]
            explanation = stereotype + ';' + separation + ';' + discrimination
        except:
            explanation = explanation

    prompt = f"""
    {instruction}; 
    {definition};
    Only rewrite the relevant parts of the post, do not rewrite the whole post. Do not change the meaning of the post or add any new information.
    Example:
    post: "My mom is an addict"; This post uses the term 'addict'
    rewrite: "My mom has a substance use disorder"
    
    Do not include "Here is the rewritten post:" in your response. Just return the rewritten post.
    """
    ex = f"This post {explanation}"
    # example1 = "my mom is an addict."
    # answer1 = "my mom has a substance use disorder."

    global request_count, token_count, daily_token_count
    response_tokens = 0  # Initialize response_tokens before the try block

    while retries > 0:
        try:
            with request_lock:
                if request_count >= rate_limit:
                    print("Rate limit reached. Pausing for a minute...")
                    time.sleep(rate_limit_period)
                    request_count = 0
            with token_lock:
                if token_count >= tpm_limit:
                    print(token_count)
                    print("TPM limit reached. Pausing for a minute...")
                    time.sleep(rate_limit_period)
                    token_count = 0

                if daily_token_count >= tpd_limit:
                    print("TPD limit reached. Stopping processing...")
                    return "skipped"

            response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                # {
                #     "role": "user",
                #     "content": example1

                # },
                # {
                #     "role": "assistant",
                #     "content": answer1
                # },
                # {
                #     "role": "assistant",
                #     "content": ex
                # },
                {
                    "role": "user",
                    "content": post + ";" + ex
                }
            ],
            model=model,
            temperature=0
        )
            with request_lock:
                request_count += 1

            with token_lock:
                response_tokens += sum([len(prompt), len(response.choices[0].message.content)])
                token_count += response_tokens
                daily_token_count += response_tokens

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
    