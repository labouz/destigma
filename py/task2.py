# the script for labeling a post as containing stigmatizing language or not
import pandas as pd
import time
import openai
from threading import Lock

rate_limit = 10000
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

def get_utterance(post, retries = 2, model = "gpt-4-turbo-2024-04-09", openai_client=None):
    prompt = f"""
    *Instructions:*
    You are an expert in identifying stigma in social media posts, focusing on both personal and systemic levels. Your task is to read each post and determine if it should be labeled as 'S' (Stigma) for depicting stigma or 'NS' (Non-Stigma) if it does not. For posts labeled as 'S', provide a detailed explanation based on the following criteria:

    1. **Labeling**: Does the post label the author or others based on drug use or addiction? Consider how the author might use stigmatizing language about themselves or others, reflecting internalized societal attitudes.

    2. **Stereotyping**: What stereotypes about drug use or addiction does the post reinforce, even if these are self-directed? Discuss how these stereotypes reflect broader societal views that may perpetuate stigma, even if the author is speaking only about themselves.

    3. **Separation**: How does the post foster a sense of separation between the author and the rest of society or between different groups? Even if the author is describing their own experiences, consider how they might be reinforcing a sense of 'otherness' that aligns with stigmatizing narratives.

    4. **Discrimination**: Does the post suggest any form of discrimination against the author or others based on drug use or addiction? This includes personal reflections that might underscore societal discrimination or internalized negative consequences.

    *Considerations:*
    - Reflect on the author's perspective: Is the author speaking solely about their own experiences, or are they also reflecting societal attitudes that contribute to stigma?
    - Consider the role of power and its influence on the stigmatization process, particularly how personal narratives can reflect and reinforce societal power dynamics.

    *Definition of Stigma:*
    Stigma involves negative attitudes, beliefs, and behaviors towards individuals based on perceived characteristics, leading to labeling, stereotyping, separation, and discrimination. This can occur at personal levels (personal stigma) and systemic levels (systemic stigma).

    *Task:*
    Read the following post and:
    1. Label it as 'S' (Stigma) if it meets the criteria for stigma, including self-stigmatization, and provide a detailed explanation for each of the four components. 
    2. Label it as 'NS' (Non-Stigma) if it does not meet these criteria, and no explanation is needed.

    Post: "[Insert the text of the post here]"
    Response: "Label: [S or NS], Explanation: [Your detailed explanation here if 'S']"
    """
    example1 = "Once an addict, always an addict. They'll never change."
    answer1 = "S, Labeling: Labels individuals with substance use disorder as 'addicts.', Stereotyping: Assumes that people with addiction are incapable of recovery or personal growth, Separation: Creates a sense of hopelessness and perpetuates the idea that people with addiction are fundamentally different, Discrimination: Discourages offering support or opportunities to individuals in recovery due to a belief they'll relapse."

    example2 = "He's just a junkie looking for his next fix."
    answer2 = "S, Labeling: Labels the individual as a 'junkie', a derogatory term, Stereotyping: Reduces the person's identity to their substance use, ignoring other aspects of their life, Separation: Creates a distance between 'us' (non-users) and 'them' (people who use substances), Discrimination: Implies the person is solely motivated by drug use and lacks other values or goals."

    example3 = "She's constantly nodding off at work. Probably using again."
    answer3 = "S, Labeling: Implies the person has a substance use disorder by suggesting they are 'using again,' Stereotyping: Generalizes that the person's behavior (nodding off) is solely due to drug use, Separation: Creates a distinction between the person and their colleagues, implying they are unreliable or irresponsible due to assumed substance use, Discrimination: Could lead to unfair treatment at work based on an assumption of relapse."

    example4 = "If you hang out with those druggies, you'll end up like them."
    answer4 = "S, Labeling: Uses the derogatory term 'druggies', Stereotyping: Generalizes that all people who use drugs are bad influences and will lead others to the same behavior, Separation: Creates a clear division between 'us' (those who don't use drugs) and 'them' (those who do), Discrimination: Promotes exclusion and fear of people who use drugs, potentially leading to social isolation."

    # Non-Stigmatizing Examples

    example5 = "My friend is in recovery from opioid use disorder and is doing incredibly well."
    answer5 = "NS"

    example6 = "Addiction is a complex disease, but with the right support, people can and do recover."
    answer6 = "NS"

    example7 = "I'm concerned about my brother's substance use. I'm learning about available resources to help him."
    answer7 = "NS"

    # self and systemic
    example8 = "I always hide my medication because I feel ashamed of my condition. I'm just an addict, and that's all people will see if they know."
    answer8 = "S, The author internalizes the negative stereotype associated with addiction, labeling themselves pejoratively as 'just an addict.' This self-labeling reinforces personal shame and societal stereotypes, contributing to a separation from others who might not understand or who might judge based on this one aspect of their identity."
    example9 = "Our town's only rehab facility refuses to accept people who have been arrested for drug offenses. They say it's a policy to maintain safety, but it just prevents those who need help the most from getting it."
    answer9 = "S, This post describes a systemic policy that discriminates against individuals with drug offenses, reinforcing stereotypes that they are dangerous. The policy creates a separation by denying these individuals access to rehabilitation, which is crucial for their recovery and reintegration into society."

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
                    "role": "user",
                    "content": example4
                },
                {
                    "role": "system",
                    "content": answer4
                },
                {
                    "role": "user",
                    "content": example5
                },
                {
                    "role": "system",
                    "content": answer5
                },
                {
                    "role": "user",
                    "content": example6
                },
                {
                    "role": "system",
                    "content": answer6
                },
                {
                    "role": "user",
                    "content": example7
                },
                {
                    "role": "system",
                    "content": answer7
                },
                {
                    "role": "user",
                    "content": example8
                },
                {
                    "role": "system",
                    "content": answer8
                },
                {
                    "role": "user",
                    "content": example9
                },
                {
                    "role": "system",
                    "content": answer9
                },
                {
                    "role": "user",
                    "content": post
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