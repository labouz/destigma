# the script for labeling a post as containing stigmatizing language or not
import pandas as pd
import time
from threading import Lock

rate_limit = 500
rate_limit_period = 60  # seconds

# Global variables for tracking rate limit
request_count = 0
request_lock = Lock()

def get_utterance(post, retries = 2, model = "gpt-4-turbo-2024-04-09", openai_client=None):
    prompt = f"""
    *Instructions:*

    You are an expert in identifying stigma in social media posts. Your task is to label each post as either S (Stigma) or NS (Non-Stigma).
    If the post contains stigmatizing language:
    1. Extract the utterance (the exact phrase/sentence or the underlying message).
    2. Explain why it's stigmatizing by providing evidence for ALL FOUR of the following components:

    * Labeling: Clearly identify how the utterance labels or categorizes a group based on a characteristic.
    * Stereotyping: Explain the negative or oversimplified generalizations the utterance makes about the labeled group.
    * Separation: Describe how the utterance creates an "us vs. them" dynamic or suggests exclusion or isolation.
    * Discrimination: Specify how the utterance implies or encourages unfair treatment or result in loss of status based on the group's characteristic? Does the underlying message perpetuate unequal access to resources, opportunities, or social acceptance?  

    *Definition of Stigma:*
    Stigma is a complex social phenomenon characterized by negative attitudes, beliefs, and behaviors towards individuals or groups based on a perceived characteristic or attribute. It involves the following components:

    1. Labeling:  Identifying and marking individuals or groups based on a specific characteristic, often reducing them to that single attribute.
    2. Stereotyping: Assigning negative or inaccurate generalizations to the labeled group, perpetuating harmful assumptions and expectations.
    3. Separation: Creating a perceived "us vs. them" dynamic, leading to social distancing, exclusion, and isolation of the stigmatized group.
    4. Discrimination: Unfairly treating the stigmatized group based on negative attitudes and stereotypes, often resulting in unequal access to resources, opportunities, and social acceptance.

    Key Distinctions from Negative Personal Opinions and Rudeness:
    * Stigma is widespread and systemic, while opinions are individual and isolated.
    * Stigma generalizes about groups, while opinions are more specific.
    * Stigma is enduring, while opinions are more fleeting.
    * Stigma has severe consequences (discrimination, etc.), while opinions are less impactful.
    * Stigma often intends to devalue, while rudeness might be unintentional.
    * Stigma involves power imbalances, while rudeness can occur in any interaction.

    *Considerations:*
    * Is the author part of the stigmatized group? Their language may be descriptive or empowering rather than stigmatizing.
    * Stigma is entirely dependent on social, economic, and political power;it takes power to stigmatize. In some instances the role of power is obvious. However, the role of power in stigma is frequently overlooked because in many instances power differences are so taken for granted as to seem unproblematic.
    * If uncertain, err on the side of NS (Non-Stigma).
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


    global request_count

    while retries > 0:
        try:
            with request_lock:
                if request_count >= rate_limit:
                    print("Rate limit reached. Pausing for a minute...")
                    time.sleep(rate_limit_period)
                    request_count = 0

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
                    "content": post
                }

            ],
            model=model,
            temperature=0
        )
            with request_lock:
                request_count += 1

            label = response.choices[0].message.content.lower().strip()
            return label
        
        except Exception as e:
            if e:
                print(e)
                retries -= 1
                time.sleep(5)
    print("Retrying...")
    return "skipped"