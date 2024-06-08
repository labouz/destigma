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
    **Instructions:**
    You are an expert in identifying stigma related to drug use and addiction in social media posts. Your task is to analyze each post and determine if it contains stigmatizing language. 

    1. **Label:**
        * If the post contains stigmatizing language, label it as **"S" (Stigma)**.
        * If the post does not contain stigmatizing language, label it as **"NS" (Non-Stigma)**.

    2. **Explanation:**
        * **For "S" (Stigma) labels:**
            * If the post demonstrates **stigmatizing language (directed at others)**, analyze it based on the following four attributes and provide an explanation for each attribute:
                * **Labeling:** Does the post use derogatory or othering language related to drug use/addiction?
                * **Stereotyping:** Does the post reinforce negative generalizations about people who use drugs?
                * **Separation:** Does the post create a divide between people who use drugs and those who don't?
                * **Discrimination:** Does the post imply or suggest unfair treatment based on drug use?
            * If the post demonstrates **self-stigma or structural stigma**, provide a brief, one-sentence explanation of the type of stigma present.

        * **For "NS" (Non-Stigma) labels:** No explanation is needed.

    **Definitions:**

    * **Stigma:** Negative attitudes or beliefs directed at individuals based on their drug use.
    * **Self-Stigma:** Internalization of negative societal attitudes about drug use.
    * **Structural Stigma:** Societal systems, policies, or practices that disadvantage people who use drugs.

    **Additional Considerations:**

    * Consider the context of the post and the author's intent.
    * Focus on language that is harmful, discriminatory, or perpetuates negative stereotypes.

    **Task:**
    Read the following post, label it as "S" or "NS", and provide the appropriate explanation:
    """
    example1 = "Once an addict, always an addict. They'll never change."
    answer1 = "S, Labeling: 'addict', Stereotyping: Assumes that people with addiction are incapable of recovery or personal growth, Separation: Creates a sense of hopelessness and perpetuates the idea that people with addiction are fundamentally different, Discrimination: Discourages offering support or opportunities to individuals in recovery due to a belief they'll relapse."

    example2 = "He's just a junkie looking for his next fix."
    answer2 = "S, Labeling: 'junkie', a derogatory term, Stereotyping: Reduces the person's identity to their substance use, ignoring other aspects of their life, Separation: Creates a distance between 'us' (non-users) and 'them' (people who use substances), Discrimination: Implies the person is solely motivated by drug use and lacks other values or goals."

    example3 = "She's constantly nodding off at work. Probably using again."
    answer3 = "S, Labeling: Implies the person has a substance use disorder by suggesting they are 'using again,' Stereotyping: Generalizes that the person's behavior (nodding off) is solely due to drug use, Separation: Creates a distinction between the person and their colleagues, implying they are unreliable or irresponsible due to assumed substance use, Discrimination: Could lead to unfair treatment at work based on an assumption of relapse."

    example4 = "If you hang out with those druggies, you'll end up like them."
    answer4 = "S, Labeling: 'druggies', Stereotyping: Generalizes that all people who use drugs are bad influences and will lead others to the same behavior, Separation: Creates a clear division between 'us' (those who don't use drugs) and 'them' (those who do), Discrimination: Promotes exclusion and fear of people who use drugs, potentially leading to social isolation."

    # Non-Stigmatizing Examples

    example5 = "My friend is in recovery from opioid use disorder and is doing incredibly well."
    answer5 = "NS"

    example6 = "Addiction is a complex disease, but with the right support, people can and do recover."
    answer6 = "NS"

    example7 = "I'm concerned about my brother's substance use. I'm learning about available resources to help him."
    answer7 = "NS"

    # self and systemic
    example8 = "I always hide my medication because I feel ashamed of my condition. I'm just an addict, and that's all people will see if they know."
    answer8 = "S, The author internalizes the negative stereotype associated with addiction, labeling themselves pejoratively as 'just an addict.'"
    example9 = "Our town's only rehab facility refuses to accept people who have been arrested for drug offenses. They say it's a policy to maintain safety, but it just prevents those who need help the most from getting it."
    answer9 = "S, This post describes a systemic policy that discriminates against individuals with drug offenses, reinforcing stereotypes that they are dangerous. The policy creates a separation by denying these individuals access to rehabilitation."
    example10 = "Crack is a better alternative to coffee All coffee does is make you feel less drowsy. When I smoke a bowl of crack in the morning, shit gets me gooooinn, makes me go absolutely bonkers, crazy ridiculous out of the world bonkers ! I'm getting the jitters just thinkin about it"
    answer10 = "NS"

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
                    "content": example10
                },
                {
                    "role": "system",
                    "content": answer10
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