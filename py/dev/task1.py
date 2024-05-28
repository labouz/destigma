# the script for labeling a post as stigma or not, as drug related or noti 
import pandas as pd
import pickle
import os
import time
import openai

def get_stigma_drug(post, retries = 2, model = "gpt-4-turbo", openai_client=None):
# prompt
    prompt = f"""
    Instructions:

    You are an expert in identifying stigma in social media posts. Your task is to label each post as either S (Stigma) or NS (Non-Stigma).
    If the post contains stigma, extract the exact utterance(s) of stigma and provide a brief description of the stigma. 

    Definition of Stigma:

    Stigma is a complex social phenomenon characterized by negative attitudes, beliefs, and behaviors towards individuals or groups based on a perceived characteristic or attribute. It involves the following components:

    1. Labeling:  Identifying and marking individuals or groups based on a specific characteristic, often reducing them to that single attribute.
    2. Stereotyping: Assigning negative or inaccurate generalizations to the labeled group, perpetuating harmful assumptions and expectations.
    3. Separation: Creating a perceived "us vs. them" dynamic, leading to social distancing, exclusion, and isolation of the stigmatized group.
    4. Discrimination: Unfairly treating the stigmatized group based on negative attitudes and stereotypes, often resulting in unequal access to resources, opportunities, and social acceptance.

    Key Distinctions from Negative Personal Opinions:

    * Scope: Stigma is a widespread social phenomenon embedded in cultural norms and beliefs, while negative opinions are isolated views held by individuals.
    * Generalization vs. Specificity: Stigma involves broad negative generalizations about an entire group, whereas personal opinions tend to be more specific to individual experiences or interactions.
    * Persistence: Stigma is enduring and pervasive, often reinforced by social institutions and media, whereas personal opinions can be more fleeting and changeable.
    * Impact: Stigma has serious consequences for individuals and groups, leading to discrimination, social isolation, and reduced opportunities. While negative personal opinions can be hurtful, their impact is usually less widespread and systemic.

    Distinguishing from Rudeness:

    * Intent: Stigma often involves an underlying intent to devalue or marginalize a group, while rudeness can be unintentional or simply a result of poor social skills.
    * Power Dynamics: Stigma is often associated with power imbalances, where dominant groups perpetuate negative attitudes towards marginalized groups. Rudeness can occur in any social interaction, regardless of power dynamics.

    Labeling:

    Label each post as either:

    * S: Stigma 
    * NS: Non-Stigma
    
    If you are unsure, provide your best guess.
    """

    # few-shot examples
    example1 = "I can't believe they're letting addicts use drugs in that safe injection site. It's just enabling them."
    answer1 = "S, 'addicts', The author is using the term 'addicts' to describe people who use drugs and is implying that providing a safe injection site is enabling their drug use."
    example2 = "I hope my junkie sister OD's or disappears out of our lives My sister is an alcoholic junkie who has 2 DUIs under her belt as well as loves taking Xanax and alcohol together and wreaking havoc for our family and even strangers."
    answer2 = "S, 'junkie', The author is using the term junkie to describe their sister and is perpetuating the stereotype that people who use drugs are dangerous and harmful."
    example3 = "That person is so annoying. They always talk too loud."
    answer3 = "NS (rudeness)"
    example4 = "Led Zeppelin isn't good I genuinely believe they only got popular because people listened to them while stoned."
    answer4 = "NS (opinion)"
    example5 = "I don't like people who are overweight. They're lazy and undisciplined." 
    answer5 = "S, 'They're lazy and undisciplined'. The author is using negative stereotypes to describe people who are overweight."
    example6 = "I think I genuinely may have developed some sort of PTSD due to the incident. I have had an anxiety disorder of some kind for my entire life but over the past few years it has been getting steadily worse and I wonder if it is due to that incident."
    answer6 = "NS"
    example7 = "My brother is in recovery, and I'm so proud of his progress. It's not easy, but he's doing it."
    answer7 = "NS"
    example8 = "I don't like people who use drugs. They're all losers."
    answer8 = "NS (opinion)"
    while retries > 0:
        try:
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
                    "content": post
                }

            ],
            model=model,
            temperature=0
        )
            label = response.choices[0].message.content.lower().strip()
            return label
        except Exception as e:
            if e:
                print(e)
                retries -= 1
                time.sleep(5)
            else:
                raise e
    print("Retrying...")
    return "skipped"

